"""Generate visual reports for benchmark runs and field datasets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import load_array_pair_dataset, load_constraint
from .training import load_trained_model


def plot_benchmark_summary(
    summary_csv: str | Path,
    output_dir: str | Path,
    *,
    title: str | None = None,
) -> list[Path]:
    """Plot benchmark metrics from a summary CSV file."""

    plt = _pyplot()
    rows = _read_summary(summary_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = [row["variant"] for row in rows]

    metric_groups = [
        (
            "benchmark_metrics.png",
            [
                ("loss", "Prediction MSE", False),
                ("constraint_violation_mean", "Constraint Violation", True),
                ("rollout_protected_relative_drift", "Protected Drift", True),
            ],
        ),
        (
            "component_losses.png",
            [
                ("protected_component_loss", "Protected Loss", True),
                ("complement_component_loss", "Complement Loss", True),
                ("rollout_violation_final", "Final Rollout Violation", True),
            ],
        ),
    ]
    paths: list[Path] = []
    for filename, metrics in metric_groups:
        fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4), constrained_layout=True)
        if len(metrics) == 1:
            axes = [axes]
        for axis, (metric, label, log_scale) in zip(axes, metrics):
            values = [_float(row.get(metric, "nan")) for row in rows]
            axis.bar(variants, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"][: len(variants)])
            axis.set_title(label)
            axis.tick_params(axis="x", rotation=35)
            if log_scale:
                axis.set_yscale("symlog", linthresh=1e-8)
            axis.grid(axis="y", alpha=0.25)
        if title:
            fig.suptitle(title)
        path = output_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_field_sample(
    data_path: str | Path,
    output_path: str | Path,
    *,
    sample_index: int = 0,
    input_key: str = "x",
    target_key: str = "y",
) -> Path:
    """Plot input and target velocity field channels from an ``.npz`` dataset."""

    plt = _pyplot()
    with np.load(data_path) as data:
        x = data[input_key][sample_index]
        y = data[target_key][sample_index]
    if x.ndim != 3 or x.shape[0] < 2:
        raise ValueError("field sample plotting expects arrays with shape (C,H,W), C>=2.")
    panels = [
        ("input vx", x[0]),
        ("input vy", x[1]),
        ("input speed", np.sqrt(x[0] ** 2 + x[1] ** 2)),
        ("target vx", y[0]),
        ("target vy", y[1]),
        ("target speed", np.sqrt(y[0] ** 2 + y[1] ** 2)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for axis, (label, arr) in zip(axes.flat, panels):
        image = axis.imshow(arr, origin="lower", cmap="viridis")
        axis.set_title(label)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, shrink=0.8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_prediction_sample(
    *,
    checkpoint: str | Path,
    data_path: str | Path,
    constraint_path: str | Path,
    output_path: str | Path,
    field_shape: tuple[int, int, int],
    sample_index: int = 0,
    implicit_constraint: bool = False,
    input_key: str = "x",
    target_key: str = "y",
) -> Path:
    """Plot target, prediction, and absolute error for one checkpoint sample."""

    plt = _pyplot()
    dataset = load_array_pair_dataset(data_path, input_key=input_key, target_key=target_key)
    constraint = load_constraint(constraint_path, implicit=implicit_constraint)
    model, _ = load_trained_model(checkpoint, constraint)
    x, y = dataset[sample_index]
    with torch.no_grad():
        pred = model(x.unsqueeze(0)).squeeze(0)
    target = y.reshape(field_shape).cpu().numpy()
    prediction = pred.reshape(field_shape).cpu().numpy()
    error = np.abs(prediction - target)
    panels = [
        ("target vx", target[0]),
        ("target vy", target[1]),
        ("target speed", np.sqrt(target[0] ** 2 + target[1] ** 2)),
        ("pred vx", prediction[0]),
        ("pred vy", prediction[1]),
        ("pred speed", np.sqrt(prediction[0] ** 2 + prediction[1] ** 2)),
        ("abs err vx", error[0]),
        ("abs err vy", error[1]),
        ("abs err speed", np.abs(np.sqrt(prediction[0] ** 2 + prediction[1] ** 2) - np.sqrt(target[0] ** 2 + target[1] ** 2))),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(10, 9), constrained_layout=True)
    for axis, (label, arr) in zip(axes.flat, panels):
        image = axis.imshow(arr, origin="lower", cmap="magma" if "err" in label else "viridis")
        axis.set_title(label)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, shrink=0.75)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_divergence_residual_sample(
    *,
    checkpoint: str | Path,
    data_path: str | Path,
    constraint_path: str | Path,
    output_path: str | Path,
    field_shape: tuple[int, int, int],
    sample_index: int = 0,
    implicit_constraint: bool = False,
    input_key: str = "x",
    target_key: str = "y",
) -> Path:
    """Plot target and prediction divergence residuals for a field sample."""

    plt = _pyplot()
    channels, height, width = field_shape
    if channels != 2:
        raise ValueError("divergence residual plotting expects two velocity channels.")
    dataset = load_array_pair_dataset(data_path, input_key=input_key, target_key=target_key)
    constraint = load_constraint(constraint_path, implicit=implicit_constraint)
    if constraint.codim != height * width:
        raise ValueError(
            "divergence residual plotting expects constraint codim H*W, "
            f"got {constraint.codim} for {height}x{width}."
        )
    model, _ = load_trained_model(checkpoint, constraint)
    x, y = dataset[sample_index]
    with torch.no_grad():
        pred = model(x.unsqueeze(0)).squeeze(0)
    target_residual = constraint.apply(y.unsqueeze(0)).reshape(height, width).cpu().numpy()
    pred_residual = constraint.apply(pred.unsqueeze(0)).reshape(height, width).cpu().numpy()
    panels = [
        ("target divergence", target_residual),
        ("pred divergence", pred_residual),
        ("abs residual diff", np.abs(pred_residual - target_residual)),
    ]
    vmax = max(float(np.max(np.abs(target_residual))), float(np.max(np.abs(pred_residual))), 1e-12)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, (label, arr) in zip(axes, panels):
        cmap = "magma" if label.startswith("abs") else "coolwarm"
        image = axis.imshow(
            arr,
            origin="lower",
            cmap=cmap,
            vmin=0.0 if label.startswith("abs") else -vmax,
            vmax=vmax,
        )
        axis.set_title(label)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, shrink=0.8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for benchmark visual reports."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--constraint", default=None)
    parser.add_argument("--implicit-constraint", action="store_true")
    parser.add_argument("--field-shape", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--divergence-plot", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    produced: list[Path] = []
    if args.summary_csv is not None:
        produced.extend(plot_benchmark_summary(args.summary_csv, output_dir, title=args.title))
    if args.data is not None:
        produced.append(
            plot_field_sample(
                args.data,
                output_dir / "field_sample.png",
                sample_index=args.sample_index,
            )
        )
    if args.checkpoint is not None:
        if args.constraint is None or args.data is None or args.field_shape is None:
            raise ValueError("--checkpoint requires --data, --constraint, and --field-shape.")
        field_shape = _parse_shape(args.field_shape)
        produced.append(
            plot_prediction_sample(
                checkpoint=args.checkpoint,
                data_path=args.data,
                constraint_path=args.constraint,
                output_path=output_dir / "prediction_sample.png",
                field_shape=field_shape,
                sample_index=args.sample_index,
                implicit_constraint=args.implicit_constraint,
            )
        )
        if args.divergence_plot:
            produced.append(
                plot_divergence_residual_sample(
                    checkpoint=args.checkpoint,
                    data_path=args.data,
                    constraint_path=args.constraint,
                    output_path=output_dir / "divergence_residual_sample.png",
                    field_shape=field_shape,
                    sample_index=args.sample_index,
                    implicit_constraint=args.implicit_constraint,
                )
            )
    for path in produced:
        print(f"wrote {path}")


def _read_summary(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _parse_shape(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(parts) != 3:
        raise ValueError("--field-shape must have form C,H,W.")
    return parts


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Visualization requires matplotlib. Install with `pip install -e .[visual]`."
        ) from exc
    return plt


if __name__ == "__main__":
    main()
