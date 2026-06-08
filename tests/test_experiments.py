from __future__ import annotations

import torch

from gap_protected_transformers.datasets import make_edge_flow_complement_dataset
from gap_protected_transformers.experiments import build_model, run_experiment


def test_complement_dynamics_dataset_has_nontrivial_complement_target() -> None:
    bundle = make_edge_flow_complement_dataset(seed=22)

    protected_delta = bundle.constraint.project_kernel(bundle.train_y - bundle.train_x)
    target_complement = bundle.constraint.project_complement(bundle.train_y)

    assert torch.linalg.norm(protected_delta) < 1e-8
    assert torch.linalg.norm(target_complement) > 1e-3


def test_attention_core_variant_preserves_shape() -> None:
    bundle = make_edge_flow_complement_dataset(seed=23)
    model = build_model(
        "split_gap_protected",
        bundle.constraint.dim,
        bundle.constraint,
        core_type="attention",
    ).to(dtype=torch.float64)

    y = model(bundle.test_x[:3])

    assert y.shape == bundle.test_x[:3].shape


def test_transformer_core_variant_preserves_shape() -> None:
    bundle = make_edge_flow_complement_dataset(seed=24)
    model = build_model(
        "split_gap_protected",
        bundle.constraint.dim,
        bundle.constraint,
        core_type="transformer",
        split_complement_residual=True,
    ).to(dtype=torch.float64)

    y = model(bundle.test_x[:3])

    assert y.shape == bundle.test_x[:3].shape


def test_structure_diagnostics_run_reports_nontrivial_commutators(tmp_path) -> None:
    rows = run_experiment(
        "structure_diagnostics",
        output_dir=tmp_path,
        variants=("vanilla",),
    )

    values = {row["variant"]: row for row in rows}
    assert values["compatible_cycle_hierarchy"]["commutator_error"] < 1e-12
    assert values["incompatible_cycle_hierarchy"]["commutator_error"] > 1e-3
    assert values["componentwise_router"]["routing_commutator_error"] < 1e-12
    assert values["random_router"]["routing_commutator_error"] > 1e-3


def test_learned_structure_diagnostics_trade_fit_for_compatibility(tmp_path) -> None:
    rows = run_experiment(
        "learned_structure_diagnostics",
        output_dir=tmp_path,
        epochs=10,
        lr=5e-3,
    )

    values = {row["variant"]: row for row in rows}
    assert values["learned_hard_compatible_restriction"]["commutator_error"] < 1e-10
    assert values["learned_soft_penalty_restriction"]["commutator_error"] > 1e-3
    assert values["learned_unconstrained_restriction"]["commutator_error"] > 1e-3
    assert values["learned_hard_compatible_router"]["routing_commutator_error"] < 1e-10
    assert values["learned_soft_penalty_router"]["routing_commutator_error"] > 1e-3
    assert values["learned_unconstrained_router"]["routing_commutator_error"] > 1e-3
