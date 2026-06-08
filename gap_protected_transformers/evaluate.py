"""CLI for evaluating saved protected-operator checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import load_array_pair_dataset, load_constraint
from .logging_utils import write_json
from .training import evaluate_supervised, load_trained_model


def main(argv: list[str] | None = None) -> None:
    """Evaluate a saved checkpoint on an external array-pair dataset."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--data", required=True, help=".npz file containing inputs/targets.")
    parser.add_argument("--constraint", required=True, help=".npy or .npz constraint matrix.")
    parser.add_argument("--input-key", default="x")
    parser.add_argument("--target-key", default="y")
    parser.add_argument("--constraint-key", default="A")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--output", default=None, help="Optional metrics JSON output path.")
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
    )
    model, checkpoint = load_trained_model(
        args.checkpoint,
        constraint,
        map_location=args.device,
    )
    model = model.to(torch.device(args.device)).to(dtype=dtype)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    metrics = evaluate_supervised(
        model,
        constraint,
        loader,
        device=torch.device(args.device),
        rollout_steps=args.rollout_steps,
    )
    payload = {
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "data": str(Path(args.data)),
        "constraint": str(Path(args.constraint)),
        "metrics": metrics,
    }
    if args.output is not None:
        write_json(args.output, payload)
    print(
        "loss={loss:.6g} violation_mean={constraint_violation_mean:.6g} "
        "protected_drift={rollout_protected_relative_drift:.6g}".format(**metrics)
    )


if __name__ == "__main__":
    main()
