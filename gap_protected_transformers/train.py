"""CLI for supervised training on external array datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import load_array_pair_dataset, load_constraint, split_dataset
from .models import ModelConfig, build_model
from .training import TrainingConfig, train_supervised


def main(argv: list[str] | None = None) -> None:
    """Run supervised training from ``.npz`` arrays and a constraint matrix."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help=".npz file containing inputs/targets.")
    parser.add_argument("--constraint", required=True, help=".npy or .npz constraint matrix.")
    parser.add_argument("--input-key", default="x")
    parser.add_argument("--target-key", default="y")
    parser.add_argument("--constraint-key", default="A")
    parser.add_argument(
        "--implicit-constraint",
        action="store_true",
        help="Use iterative implicit projection instead of dense pseudoinverse projection.",
    )
    parser.add_argument(
        "--variant",
        choices=(
            "vanilla",
            "soft_penalty",
            "output_projection",
            "layerwise_projection",
            "split_gap_protected",
        ),
        default="split_gap_protected",
    )
    parser.add_argument(
        "--core",
        choices=("mlp", "attention", "transformer", "patch_transformer", "moe"),
        default="transformer",
    )
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--ffn-multiplier", type=int, default=4)
    parser.add_argument(
        "--field-shape",
        default=None,
        help="Optional C,H,W shape for patch_transformer; inferred from data if omitted.",
    )
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--patch-embed-dim", type=int, default=128)
    parser.add_argument("--patch-layers", type=int, default=2)
    parser.add_argument("--no-complement-residual", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--penalty-weight", type=float, default=10.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--output-dir", default="runs/supervised")
    parser.add_argument("--float64", action="store_true")
    args = parser.parse_args(argv)

    dtype = torch.float64 if args.float64 else torch.float32
    dataset = load_array_pair_dataset(
        args.data,
        input_key=args.input_key,
        target_key=args.target_key,
        dtype=dtype,
    )
    constraint = load_constraint(
        args.constraint,
        key=args.constraint_key,
        dtype=dtype,
        implicit=args.implicit_constraint,
    )
    constraint = constraint.to(torch.device(args.device))
    if dataset.metadata.flat_dim != constraint.dim:
        raise ValueError(
            f"Dataset flat dim {dataset.metadata.flat_dim} does not match "
            f"constraint dim {constraint.dim}."
        )

    train_subset, val_subset = split_dataset(
        dataset, val_fraction=args.val_fraction, seed=args.seed
    )
    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    field_shape = _parse_field_shape(args.field_shape)
    if field_shape is None and args.core == "patch_transformer":
        if len(dataset.metadata.input_shape) != 3:
            raise ValueError(
                "patch_transformer requires --field-shape C,H,W for non-3D samples."
            )
        field_shape = tuple(int(item) for item in dataset.metadata.input_shape)

    model_config = ModelConfig(
        dim=constraint.dim,
        variant=args.variant,
        core_type=args.core,
        depth=args.depth,
        hidden_dim=args.hidden_dim,
        num_tokens=args.num_tokens,
        num_heads=args.num_heads,
        ffn_multiplier=args.ffn_multiplier,
        field_shape=field_shape,
        patch_size=args.patch_size,
        patch_embed_dim=args.patch_embed_dim,
        patch_layers=args.patch_layers,
        complement_residual=not args.no_complement_residual,
    )
    model = build_model(model_config, constraint).to(dtype=dtype)
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        penalty_weight=args.penalty_weight if args.variant == "soft_penalty" else 0.0,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
        rollout_steps=args.rollout_steps,
    )
    summary = train_supervised(
        model,
        constraint,
        train_loader,
        val_loader,
        config=training_config,
        model_config=model_config.to_dict()
        | {
            "data": str(Path(args.data)),
            "constraint": str(Path(args.constraint)),
            "input_shape": dataset.metadata.input_shape,
            "target_shape": dataset.metadata.target_shape,
        },
    )
    print(f"best_epoch={summary['best_epoch']} best_val_loss={summary['best_val_loss']:.6g}")
    print(f"wrote {Path(args.output_dir)}")


def _parse_field_shape(value: str | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    parts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(parts) != 3:
        raise ValueError("--field-shape must have form C,H,W.")
    return parts


if __name__ == "__main__":
    main()
