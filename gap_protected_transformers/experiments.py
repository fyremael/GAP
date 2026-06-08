"""Minimal experiment harness for protected-mode sanity checks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import (
    GapProtectedBlock,
    MLPCore,
    OutputProjectionWrapper,
    ProjectedResidualBlock,
    ResidualBlock,
    TinyMoECore,
    TinySelfAttentionCore,
    TokenTransformerCore,
)
from .datasets import (
    TensorDatasetBundle,
    make_edge_flow_complement_dataset,
    make_edge_flow_dataset,
    make_grid_divergence_complement_dataset,
    make_grid_divergence_dataset,
)
from .diagnostics import (
    commutator_error,
    complement_jacobian_spectral_proxy,
    complement_leakage,
    constraint_violation_stats,
    gap_proxy,
    protected_leakage,
    rollout_diagnostics,
    routing_commutator_error,
)
from .logging_utils import write_csv, write_json
from .operators import LinearConstraint
from .hierarchy import CompatibleRestriction, UnconstrainedRestriction
from .routing import CompatibleRouter, UnconstrainedRouter
from .toy_complexes import identity_hierarchy
from .toy_complexes import (
    componentwise_router,
    cycle_graph_hierarchy,
    random_router,
)


VARIANTS = (
    "vanilla",
    "soft_penalty",
    "output_projection",
    "layerwise_projection",
    "split_gap_protected",
)


@dataclass
class ExperimentConfig:
    """Configuration for the small deterministic sanity experiments."""

    experiment: str
    epochs: int = 60
    lr: float = 2e-3
    depth: int = 2
    hidden_dim: int | None = None
    core_type: str = "mlp"
    penalty_weight: float = 10.0
    seed: int = 0
    output_dir: Path = Path("runs")
    dtype: torch.dtype = torch.float64


def build_model(
    variant: str,
    dim: int,
    constraint: LinearConstraint,
    *,
    depth: int = 2,
    hidden_dim: int | None = None,
    core_type: str = "mlp",
    split_complement_residual: bool = False,
) -> nn.Module:
    """Build one ablation variant from the run matrix."""

    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Expected one of {VARIANTS}.")
    hidden_dim = hidden_dim or max(32, 2 * dim)

    if variant in {"vanilla", "soft_penalty", "output_projection"}:
        model = nn.Sequential(
            *[
                ResidualBlock(_build_core(dim, hidden_dim, core_type))
                for _ in range(depth)
            ]
        )
        if variant == "output_projection":
            return OutputProjectionWrapper(model, constraint)
        return model

    if variant == "layerwise_projection":
        return nn.Sequential(
            *[
                ProjectedResidualBlock(
                    dim,
                    constraint,
                    _build_core(dim, hidden_dim, core_type),
                )
                for _ in range(depth)
            ]
        )

    return nn.Sequential(
        *[
            GapProtectedBlock(
                dim,
                constraint,
                _build_core(dim, hidden_dim, core_type, zero_init=core_type == "mlp"),
                zero_preserving=True,
                complement_residual=split_complement_residual,
            )
            for _ in range(depth)
        ]
    )


def run_experiment(
    experiment: str,
    *,
    variants: Iterable[str] = VARIANTS,
    epochs: int = 60,
    lr: float = 2e-3,
    depth: int = 2,
    hidden_dim: int | None = None,
    core_type: str = "mlp",
    penalty_weight: float = 10.0,
    seed: int = 0,
    output_dir: str | Path = "runs",
) -> list[dict[str, float | int | str | None]]:
    """Run a named sanity experiment and write metrics to CSV/JSON."""

    config = ExperimentConfig(
        experiment=experiment,
        epochs=epochs,
        lr=lr,
        depth=depth,
        hidden_dim=hidden_dim,
        core_type=core_type,
        penalty_weight=penalty_weight,
        seed=seed,
        output_dir=Path(output_dir),
    )
    if experiment == "structure_diagnostics":
        rows = _run_structure_diagnostics(config)
        run_dir = config.output_dir / experiment
        write_csv(run_dir / "metrics.csv", rows)
        write_json(
            run_dir / "metrics.json",
            {"experiment": experiment, "config": _config_dict(config), "results": rows},
        )
        return rows
    if experiment == "learned_structure_diagnostics":
        rows = _run_learned_structure_diagnostics(config)
        run_dir = config.output_dir / experiment
        write_csv(run_dir / "metrics.csv", rows)
        write_json(
            run_dir / "metrics.json",
            {"experiment": experiment, "config": _config_dict(config), "results": rows},
        )
        return rows

    bundle = _make_dataset(config)
    rows: list[dict[str, float | int | str | None]] = []
    for index, variant in enumerate(variants):
        torch.manual_seed(seed + 101 * index)
        model = build_model(
            variant,
            bundle.constraint.dim,
            bundle.constraint,
            depth=depth,
            hidden_dim=hidden_dim,
            core_type=core_type,
            split_complement_residual="complement" in experiment,
        ).to(dtype=config.dtype)
        row = _train_and_evaluate(variant, model, bundle, config)
        rows.append(row)

    run_dir = config.output_dir / experiment
    write_csv(run_dir / "metrics.csv", rows)
    write_json(
        run_dir / "metrics.json",
        {
            "experiment": experiment,
            "config": {
                "epochs": epochs,
                "lr": lr,
                "depth": depth,
                "core_type": core_type,
                "penalty_weight": penalty_weight,
                "seed": seed,
            },
            "dataset": bundle.metadata,
            "results": rows,
        },
    )
    return rows


def print_results(rows: list[dict[str, float | int | str | None]]) -> None:
    """Print a compact metric table for examples and CLI runs."""

    preferred_headers = (
        "variant",
        "test_loss",
        "constraint_violation_mean",
        "constraint_violation_max",
        "protected_component_loss",
        "complement_component_loss",
        "protected_leakage",
        "complement_leakage",
        "commutator_error",
        "commutator_error_incompatible",
        "routing_commutator_error",
        "routing_commutator_error_random",
        "rollout_violation_final",
        "rollout_protected_relative_drift",
        "complement_sigma_proxy",
        "fit_loss",
        "training_loss",
    )
    headers = tuple(
        header for header in preferred_headers if any(header in row for row in rows)
    )
    print(" | ".join(headers))
    print(" | ".join("-" * len(header) for header in headers))
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        print(" | ".join(values))


def _make_dataset(config: ExperimentConfig) -> TensorDatasetBundle:
    if config.experiment == "edge_flow_sanity":
        return make_edge_flow_dataset(seed=config.seed, dtype=config.dtype)
    if config.experiment == "edge_flow_complement":
        return make_edge_flow_complement_dataset(seed=config.seed, dtype=config.dtype)
    if config.experiment == "divergence_free_sanity":
        return make_grid_divergence_dataset(seed=config.seed, dtype=config.dtype)
    if config.experiment == "divergence_free_complement":
        return make_grid_divergence_complement_dataset(seed=config.seed, dtype=config.dtype)
    raise ValueError(
        "Unknown experiment. Expected an edge-flow, divergence-free, or structure diagnostic run."
    )


def _train_and_evaluate(
    variant: str,
    model: nn.Module,
    bundle: TensorDatasetBundle,
    config: ExperimentConfig,
) -> dict[str, float | int | str | None]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    final_total = torch.zeros((), dtype=config.dtype)
    final_pred = torch.zeros((), dtype=config.dtype)

    for _ in range(config.epochs):
        optimizer.zero_grad()
        pred = model(bundle.train_x)
        pred_loss = F.mse_loss(pred, bundle.train_y)
        loss = pred_loss
        if variant == "soft_penalty":
            loss = loss + config.penalty_weight * torch.mean(
                bundle.constraint.apply(pred) ** 2
            )
        loss.backward()
        optimizer.step()
        final_total = loss.detach()
        final_pred = pred_loss.detach()

    with torch.no_grad():
        test_pred = model(bundle.test_x)
        test_loss = F.mse_loss(test_pred, bundle.test_y)
        stats = constraint_violation_stats(bundle.constraint, test_pred)
        target_stats = constraint_violation_stats(bundle.constraint, bundle.test_y)
        protected_component_loss = F.mse_loss(
            bundle.constraint.project_kernel(test_pred),
            bundle.constraint.project_kernel(bundle.test_y),
        )
        complement_component_loss = F.mse_loss(
            bundle.constraint.project_complement(test_pred),
            bundle.constraint.project_complement(bundle.test_y),
        )
        train_pred = model(bundle.train_x)
        train_pred_loss = F.mse_loss(train_pred, bundle.train_y)
        gap = gap_proxy(bundle.constraint)
        hierarchy = _hierarchy_metrics(bundle)
        routing = _routing_metrics(bundle.constraint, seed=config.seed)
        rollout = rollout_diagnostics(
            model,
            bundle.constraint,
            bundle.test_x[: min(16, bundle.test_x.shape[0])],
            steps=6,
        )
        sigma_perp = complement_jacobian_spectral_proxy(
            model,
            bundle.constraint,
            bundle.test_x[: min(8, bundle.test_x.shape[0])],
            steps=4,
            seed=config.seed,
        )
        row: dict[str, float | int | str | None] = {
            "variant": variant,
            "core_type": config.core_type,
            "train_loss": float(final_total.cpu()),
            "train_pred_loss": float(final_pred.cpu()),
            "train_pred_loss_after_step": float(train_pred_loss.cpu()),
            "test_loss": float(test_loss.cpu()),
            "protected_component_loss": float(protected_component_loss.cpu()),
            "complement_component_loss": float(complement_component_loss.cpu()),
            "constraint_violation_mean": float(
                stats["constraint_violation_mean"].cpu()
            ),
            "constraint_violation_max": float(stats["constraint_violation_max"].cpu()),
            "constraint_violation_p95": float(stats["constraint_violation_p95"].cpu()),
            "target_constraint_violation_mean": float(
                target_stats["constraint_violation_mean"].cpu()
            ),
            "protected_leakage": float(protected_leakage(model, bundle.constraint).cpu()),
            "complement_leakage": float(complement_leakage(model, bundle.constraint).cpu()),
            "gap_lambda_min_plus": float(gap["lambda_min_plus"].cpu()),
            "gap_ratio": float(gap["gap_ratio"].cpu()),
            "gap_nullity": int(gap["nullity"]),
            "complement_sigma_proxy": float(sigma_perp.cpu()),
            "rollout_violation_mean": float(
                rollout["rollout_violation_mean"].cpu()
            ),
            "rollout_violation_max": float(rollout["rollout_violation_max"].cpu()),
            "rollout_violation_final": float(
                rollout["rollout_violation_final"].cpu()
            ),
            "rollout_protected_relative_drift": float(
                rollout["rollout_protected_relative_drift"].cpu()
            ),
            "dim": bundle.constraint.dim,
            "codim": bundle.constraint.codim,
        }
        row.update(hierarchy)
        row.update(routing)
    return row


def _build_core(
    dim: int,
    hidden_dim: int,
    core_type: str,
    *,
    zero_init: bool = False,
) -> nn.Module:
    if core_type == "mlp":
        return MLPCore(dim, hidden_dim=hidden_dim, zero_init=zero_init)
    if core_type == "attention":
        return TinySelfAttentionCore(dim, num_tokens=_choose_num_tokens(dim))
    if core_type == "transformer":
        return TokenTransformerCore(dim, num_tokens=_choose_num_tokens(dim))
    if core_type == "moe":
        return TinyMoECore(dim, hidden_dim=hidden_dim, num_experts=4)
    raise ValueError("core_type must be 'mlp', 'attention', 'transformer', or 'moe'.")


def _choose_num_tokens(dim: int) -> int:
    for candidate in (8, 4, 2):
        if dim % candidate == 0:
            return candidate
    return 1


def _hierarchy_metrics(bundle: TensorDatasetBundle) -> dict[str, float]:
    metadata = bundle.metadata
    if metadata.get("dataset") == "edge_flow_cycle" and int(metadata["num_nodes"]) % 2 == 0:
        num_nodes = int(metadata["num_nodes"])
        compatible = cycle_graph_hierarchy(
            num_nodes, compatible=True, dtype=bundle.constraint.A.dtype
        )
        incompatible = cycle_graph_hierarchy(
            num_nodes, compatible=False, dtype=bundle.constraint.A.dtype
        )
    else:
        A_c, R, R_A, A_f = identity_hierarchy(bundle.constraint.A)
        R_bad = R.clone()
        R_bad[0, min(1, R_bad.shape[1] - 1)] += 0.25
        compatible = (A_c, R, R_A, A_f)
        incompatible = (A_c, R_bad, R_A, A_f)

    A_c, R, R_A, A_f = compatible
    A_c_bad, R_bad, R_A_bad, A_f_bad = incompatible
    return {
        "commutator_error": float(commutator_error(A_c, R, R_A, A_f).cpu()),
        "commutator_error_incompatible": float(
            commutator_error(A_c_bad, R_bad, R_A_bad, A_f_bad).cpu()
        ),
    }


def _routing_metrics(
    constraint: LinearConstraint, *, seed: int = 0, num_experts: int = 3
) -> dict[str, float]:
    compatible_router = componentwise_router(
        constraint.dim,
        num_experts,
        dtype=constraint.A.dtype,
        device=constraint.A.device,
    )
    incompatible_router = random_router(
        constraint.dim,
        num_experts,
        seed=seed,
        dtype=constraint.A.dtype,
        device=constraint.A.device,
    )
    return {
        "routing_commutator_error": float(
            routing_commutator_error(
                compatible_router, constraint, num_experts=num_experts
            ).cpu()
        ),
        "routing_commutator_error_random": float(
            routing_commutator_error(
                incompatible_router, constraint, num_experts=num_experts
            ).cpu()
        ),
    }


def _run_structure_diagnostics(
    config: ExperimentConfig,
) -> list[dict[str, float | int | str | None]]:
    bundle = make_edge_flow_dataset(seed=config.seed, dtype=config.dtype)
    hierarchy = _hierarchy_metrics(bundle)
    routing = _routing_metrics(bundle.constraint, seed=config.seed)
    return [
        {
            "variant": "compatible_cycle_hierarchy",
            "commutator_error": hierarchy["commutator_error"],
            "commutator_error_incompatible": None,
            "routing_commutator_error": None,
            "routing_commutator_error_random": None,
        },
        {
            "variant": "incompatible_cycle_hierarchy",
            "commutator_error": hierarchy["commutator_error_incompatible"],
            "commutator_error_incompatible": hierarchy["commutator_error_incompatible"],
            "routing_commutator_error": None,
            "routing_commutator_error_random": None,
        },
        {
            "variant": "componentwise_router",
            "commutator_error": None,
            "commutator_error_incompatible": None,
            "routing_commutator_error": routing["routing_commutator_error"],
            "routing_commutator_error_random": None,
        },
        {
            "variant": "random_router",
            "commutator_error": None,
            "commutator_error_incompatible": None,
            "routing_commutator_error": routing["routing_commutator_error_random"],
            "routing_commutator_error_random": routing["routing_commutator_error_random"],
        },
    ]


def _run_learned_structure_diagnostics(
    config: ExperimentConfig,
) -> list[dict[str, float | int | str | None]]:
    bundle = make_edge_flow_dataset(seed=config.seed, dtype=config.dtype)
    A_c, target_R, R_A, A_f = cycle_graph_hierarchy(
        int(bundle.metadata["num_nodes"]), compatible=False, dtype=config.dtype
    )

    compatible_restriction = CompatibleRestriction(A_c, R_A, A_f).to(dtype=config.dtype)
    penalized_restriction = UnconstrainedRestriction(
        torch.zeros_like(target_R)
    ).to(dtype=config.dtype)
    unconstrained_restriction = UnconstrainedRestriction(
        torch.zeros_like(target_R)
    ).to(dtype=config.dtype)
    restriction_rows = _fit_restrictions(
        compatible_restriction,
        penalized_restriction,
        unconstrained_restriction,
        target_R,
        A_c,
        R_A,
        A_f,
        epochs=max(80, config.epochs),
        lr=config.lr,
        penalty_weight=config.penalty_weight,
    )

    constraint = bundle.constraint
    target_router = random_router(
        constraint.dim,
        3,
        seed=config.seed + 97,
        dtype=config.dtype,
        device=constraint.A.device,
    )
    compatible_router = CompatibleRouter(constraint, 3).to(dtype=config.dtype)
    penalized_router = UnconstrainedRouter(
        constraint.dim, 3, dtype=config.dtype
    ).to(dtype=config.dtype)
    unconstrained_router = UnconstrainedRouter(
        constraint.dim, 3, dtype=config.dtype
    ).to(dtype=config.dtype)
    router_rows = _fit_routers(
        compatible_router,
        penalized_router,
        unconstrained_router,
        target_router,
        constraint,
        epochs=max(80, config.epochs),
        lr=config.lr,
        penalty_weight=config.penalty_weight,
    )
    return restriction_rows + router_rows


def _fit_restrictions(
    compatible: CompatibleRestriction,
    penalized: UnconstrainedRestriction,
    unconstrained: UnconstrainedRestriction,
    target_R: torch.Tensor,
    A_c: torch.Tensor,
    R_A: torch.Tensor,
    A_f: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    penalty_weight: float,
) -> list[dict[str, float | int | str | None]]:
    rows: list[dict[str, float | int | str | None]] = []
    for label, module in (
        ("learned_hard_compatible_restriction", compatible),
        ("learned_soft_penalty_restriction", penalized),
        ("learned_unconstrained_restriction", unconstrained),
    ):
        optimizer = torch.optim.Adam(module.parameters(), lr=lr)
        for _ in range(epochs):
            optimizer.zero_grad()
            matrix = module.matrix()
            fit_loss = F.mse_loss(matrix, target_R)
            comm = commutator_error(A_c, matrix, R_A, A_f)
            if "soft_penalty" in label:
                loss = fit_loss + penalty_weight * comm**2
            else:
                loss = fit_loss
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            matrix = module.matrix()
            comm = commutator_error(A_c, matrix, R_A, A_f)
            fit_loss = F.mse_loss(matrix, target_R)
            rows.append(
                {
                    "variant": label,
                    "fit_loss": float(fit_loss.cpu()),
                    "training_loss": float(
                        (
                            fit_loss
                            + (penalty_weight * comm**2 if "soft_penalty" in label else 0)
                        ).cpu()
                    ),
                    "commutator_error": float(comm.cpu()),
                    "routing_commutator_error": None,
                }
            )
    return rows


def _fit_routers(
    compatible: CompatibleRouter,
    penalized: UnconstrainedRouter,
    unconstrained: UnconstrainedRouter,
    target_router: torch.Tensor,
    constraint: LinearConstraint,
    *,
    epochs: int,
    lr: float,
    penalty_weight: float,
) -> list[dict[str, float | int | str | None]]:
    rows: list[dict[str, float | int | str | None]] = []
    for label, module in (
        ("learned_hard_compatible_router", compatible),
        ("learned_soft_penalty_router", penalized),
        ("learned_unconstrained_router", unconstrained),
    ):
        optimizer = torch.optim.Adam(module.parameters(), lr=lr)
        for _ in range(epochs):
            optimizer.zero_grad()
            matrix = module.matrix()
            fit_loss = F.mse_loss(matrix, target_router)
            comm = routing_commutator_error(matrix, constraint, num_experts=3)
            if "soft_penalty" in label:
                loss = fit_loss + penalty_weight * comm**2
            else:
                loss = fit_loss
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            matrix = module.matrix()
            fit_loss = F.mse_loss(matrix, target_router)
            comm = routing_commutator_error(matrix, constraint, num_experts=3)
            rows.append(
                {
                    "variant": label,
                    "fit_loss": float(fit_loss.cpu()),
                    "training_loss": float(
                        (
                            fit_loss
                            + (penalty_weight * comm**2 if "soft_penalty" in label else 0)
                        ).cpu()
                    ),
                    "commutator_error": None,
                    "routing_commutator_error": float(comm.cpu()),
                }
            )
    return rows


def _config_dict(config: ExperimentConfig) -> dict[str, float | int | str]:
    return {
        "experiment": config.experiment,
        "epochs": config.epochs,
        "lr": config.lr,
        "depth": config.depth,
        "core_type": config.core_type,
        "penalty_weight": config.penalty_weight,
        "seed": config.seed,
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``python -m gap_protected_transformers.experiments``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=(
            "edge_flow_sanity",
            "divergence_free_sanity",
            "edge_flow_complement",
            "divergence_free_complement",
            "structure_diagnostics",
            "learned_structure_diagnostics",
        ),
        default="edge_flow_sanity",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument(
        "--core",
        choices=("mlp", "attention", "transformer", "moe"),
        default="mlp",
    )
    parser.add_argument("--penalty-weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="Comma-separated subset of variants.",
    )
    args = parser.parse_args(argv)
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    rows = run_experiment(
        args.experiment,
        variants=variants,
        epochs=args.epochs,
        lr=args.lr,
        depth=args.depth,
        core_type=args.core,
        penalty_weight=args.penalty_weight,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print_results(rows)
    print(f"Wrote metrics under {Path(args.output_dir) / args.experiment}")


if __name__ == "__main__":
    main()
