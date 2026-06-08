"""Validate external datasets and constraints before benchmark training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from .data import load_array_pair_dataset, load_constraint
from .diagnostics import constraint_violation_stats
from .logging_utils import write_json


def validate_inputs(
    *,
    data_path: str | Path,
    constraint_path: str | Path,
    input_key: str = "x",
    target_key: str = "y",
    constraint_key: str = "A",
    implicit_constraint: bool = False,
    dtype: torch.dtype = torch.float32,
    sample_count: int = 8,
) -> dict[str, Any]:
    """Validate dataset/constraint shape compatibility and basic diagnostics."""

    dataset = load_array_pair_dataset(
        data_path,
        input_key=input_key,
        target_key=target_key,
        dtype=dtype,
    )
    constraint = load_constraint(
        constraint_path,
        key=constraint_key,
        dtype=dtype,
        implicit=implicit_constraint,
    )
    report: dict[str, Any] = {
        "data": str(Path(data_path)),
        "constraint": str(Path(constraint_path)),
        "num_samples": len(dataset),
        "input_shape": dataset.metadata.input_shape,
        "target_shape": dataset.metadata.target_shape,
        "flat_dim": dataset.metadata.flat_dim,
        "constraint_dim": constraint.dim,
        "constraint_codim": constraint.codim,
        "dtype": str(dtype),
        "ok": True,
        "errors": [],
    }
    if dataset.metadata.flat_dim != constraint.dim:
        report["ok"] = False
        report["errors"].append(
            f"flat_dim {dataset.metadata.flat_dim} != constraint dim {constraint.dim}"
        )
        return report

    xs = []
    ys = []
    for index in range(min(sample_count, len(dataset))):
        x, y = dataset[index]
        xs.append(x)
        ys.append(y)
    batch_x = torch.stack(xs)
    batch_y = torch.stack(ys)
    if batch_x.shape != batch_y.shape:
        report["ok"] = False
        report["errors"].append(
            f"sample batch shape {tuple(batch_x.shape)} != target {tuple(batch_y.shape)}"
        )
        return report

    x_stats = constraint_violation_stats(constraint, batch_x)
    y_stats = constraint_violation_stats(constraint, batch_y)
    x_relative_stats = constraint_violation_stats(constraint, batch_x, relative=True)
    y_relative_stats = constraint_violation_stats(constraint, batch_y, relative=True)
    report.update(
        {
            "input_constraint_violation_mean": float(
                x_stats["constraint_violation_mean"].cpu()
            ),
            "target_constraint_violation_mean": float(
                y_stats["constraint_violation_mean"].cpu()
            ),
            "target_constraint_violation_max": float(
                y_stats["constraint_violation_max"].cpu()
            ),
            "input_constraint_violation_relative_mean": float(
                x_relative_stats["constraint_violation_mean"].cpu()
            ),
            "target_constraint_violation_relative_mean": float(
                y_relative_stats["constraint_violation_mean"].cpu()
            ),
            "target_constraint_violation_relative_max": float(
                y_relative_stats["constraint_violation_max"].cpu()
            ),
        }
    )
    return report


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for external data/constraint validation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--constraint", required=True)
    parser.add_argument("--input-key", default="x")
    parser.add_argument("--target-key", default="y")
    parser.add_argument("--constraint-key", default="A")
    parser.add_argument("--implicit-constraint", action="store_true")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--output", default=None)
    parser.add_argument("--float64", action="store_true")
    args = parser.parse_args(argv)

    dtype = torch.float64 if args.float64 else torch.float32
    report = validate_inputs(
        data_path=args.data,
        constraint_path=args.constraint,
        input_key=args.input_key,
        target_key=args.target_key,
        constraint_key=args.constraint_key,
        implicit_constraint=args.implicit_constraint,
        dtype=dtype,
        sample_count=args.sample_count,
    )
    if args.output is not None:
        write_json(args.output, report)
    status = "OK" if report["ok"] else "FAILED"
    print(f"validation={status} samples={report['num_samples']} dim={report['flat_dim']}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"error: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
