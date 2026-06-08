"""Run configured benchmark ablations on external datasets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from .data import load_array_pair_dataset, load_constraint, split_dataset
from .diagnostics import complement_jacobian_spectral_proxy
from .evaluate import main as evaluate_main
from .logging_utils import write_csv, write_json
from .train import main as train_main
from .training import load_trained_model
from .validate import validate_inputs


DEFAULT_VARIANTS = (
    "vanilla",
    "soft_penalty",
    "output_projection",
    "layerwise_projection",
    "split_gap_protected",
)


def run_benchmark(config_path: str | Path) -> dict[str, Any]:
    """Run a JSON-configured benchmark ablation."""

    path = Path(config_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    data = config["data"]
    constraint = config["constraint"]
    output_dir = Path(config.get("output_dir", "runs/benchmark"))
    output_dir.mkdir(parents=True, exist_ok=True)
    input_key = config.get("input_key", "x")
    target_key = config.get("target_key", "y")
    constraint_key = config.get("constraint_key", "A")
    implicit_constraint = bool(config.get("implicit_constraint", False))
    dtype = torch.float64 if config.get("float64", False) else torch.float32

    validation = validate_inputs(
        data_path=data,
        constraint_path=constraint,
        input_key=input_key,
        target_key=target_key,
        constraint_key=constraint_key,
        implicit_constraint=implicit_constraint,
        dtype=dtype,
    )
    write_json(output_dir / "validation.json", validation)
    if not validation["ok"]:
        raise ValueError(f"Validation failed: {validation['errors']}")

    model = config.get("model", {})
    training = config.get("training", {})
    spectral_config = config.get("spectral_proxy", {})
    variants = tuple(config.get("variants", DEFAULT_VARIANTS))
    rows: list[dict[str, Any]] = []
    for variant in variants:
        run_dir = output_dir / variant
        _reset_peak_memory(training.get("device", "cpu"))
        train_args = [
            "--data",
            data,
            "--constraint",
            constraint,
            "--input-key",
            input_key,
            "--target-key",
            target_key,
            "--constraint-key",
            constraint_key,
            "--variant",
            variant,
            "--core",
            str(model.get("core", "transformer")),
            "--depth",
            str(model.get("depth", 4)),
            "--num-tokens",
            str(model.get("num_tokens", 8)),
            "--num-heads",
            str(model.get("num_heads", 1)),
            "--ffn-multiplier",
            str(model.get("ffn_multiplier", 4)),
            "--epochs",
            str(training.get("epochs", 50)),
            "--batch-size",
            str(training.get("batch_size", 32)),
            "--lr",
            str(training.get("lr", 1e-3)),
            "--weight-decay",
            str(training.get("weight_decay", 0.0)),
            "--penalty-weight",
            str(training.get("penalty_weight", 10.0)),
            "--val-fraction",
            str(training.get("val_fraction", 0.1)),
            "--seed",
            str(training.get("seed", 0)),
            "--device",
            str(training.get("device", "cpu")),
            "--rollout-steps",
            str(training.get("rollout_steps", 4)),
            "--output-dir",
            str(run_dir),
        ]
        if implicit_constraint:
            train_args.append("--implicit-constraint")
        if config.get("float64", False):
            train_args.append("--float64")
        if model.get("core") == "patch_transformer":
            field_shape = model.get("field_shape")
            if field_shape is not None:
                train_args.extend(["--field-shape", ",".join(map(str, field_shape))])
            train_args.extend(
                [
                    "--patch-size",
                    str(model.get("patch_size", 4)),
                    "--patch-embed-dim",
                    str(model.get("patch_embed_dim", 128)),
                    "--patch-layers",
                    str(model.get("patch_layers", 2)),
                ]
            )
        train_start = time.perf_counter()
        train_main(train_args)
        _synchronize(training.get("device", "cpu"))
        train_wall_seconds = time.perf_counter() - train_start
        train_peak_memory_mb = _peak_memory_mb(training.get("device", "cpu"))

        metrics_path = run_dir / "eval_metrics.json"
        _reset_peak_memory(training.get("device", "cpu"))
        eval_args = [
            "--checkpoint",
            str(run_dir / "best.pt"),
            "--data",
            data,
            "--constraint",
            constraint,
            "--input-key",
            input_key,
            "--target-key",
            target_key,
            "--constraint-key",
            constraint_key,
            "--batch-size",
            str(training.get("batch_size", 32)),
            "--device",
            str(training.get("device", "cpu")),
            "--rollout-steps",
            str(training.get("rollout_steps", 4)),
            "--output",
            str(metrics_path),
        ]
        if implicit_constraint:
            eval_args.append("--implicit-constraint")
        if config.get("float64", False):
            eval_args.append("--float64")
        eval_start = time.perf_counter()
        evaluate_main(eval_args)
        _synchronize(training.get("device", "cpu"))
        eval_wall_seconds = time.perf_counter() - eval_start
        eval_peak_memory_mb = _peak_memory_mb(training.get("device", "cpu"))
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        complement_sigma_proxy = _benchmark_spectral_proxy(
            checkpoint=run_dir / "best.pt",
            data=data,
            constraint_path=constraint,
            input_key=input_key,
            target_key=target_key,
            constraint_key=constraint_key,
            implicit_constraint=implicit_constraint,
            dtype=dtype,
            device=str(training.get("device", "cpu")),
            val_fraction=float(training.get("val_fraction", 0.1)),
            seed=int(training.get("seed", 0)),
            enabled=bool(spectral_config.get("enabled", False)),
            sample_count=int(spectral_config.get("sample_count", 1)),
            steps=int(spectral_config.get("steps", 2)),
            eps=float(spectral_config.get("eps", 1e-3)),
        )
        peak_values = [
            item for item in (train_peak_memory_mb, eval_peak_memory_mb) if item is not None
        ]
        rows.append(
            {
                "variant": variant,
                **payload["metrics"],
                "complement_sigma_proxy": complement_sigma_proxy,
                "train_wall_seconds": train_wall_seconds,
                "eval_wall_seconds": eval_wall_seconds,
                "peak_memory_mb": max(peak_values) if peak_values else None,
            }
        )

    write_csv(output_dir / "benchmark_summary.csv", rows)
    summary = {"config": config, "validation": validation, "results": rows}
    write_json(output_dir / "benchmark_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for configured benchmark ablations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON benchmark config.")
    args = parser.parse_args(argv)
    summary = run_benchmark(args.config)
    print(f"completed variants={len(summary['results'])}")


def _benchmark_spectral_proxy(
    *,
    checkpoint: str | Path,
    data: str | Path,
    constraint_path: str | Path,
    input_key: str,
    target_key: str,
    constraint_key: str,
    implicit_constraint: bool,
    dtype: torch.dtype,
    device: str,
    val_fraction: float,
    seed: int,
    enabled: bool,
    sample_count: int,
    steps: int,
    eps: float,
) -> float | None:
    if not enabled:
        return None
    torch_device = torch.device(device)
    dataset = load_array_pair_dataset(
        data,
        input_key=input_key,
        target_key=target_key,
        dtype=dtype,
    )
    _, val_subset = split_dataset(dataset, val_fraction=val_fraction, seed=seed)
    xs = []
    for index in range(min(sample_count, len(val_subset))):
        x, _ = val_subset[index]
        xs.append(x)
    if not xs:
        return None
    x = torch.stack(xs).to(torch_device)
    constraint = load_constraint(
        constraint_path,
        key=constraint_key,
        dtype=dtype,
        implicit=implicit_constraint,
    ).to(torch_device)
    model_loaded, _ = load_trained_model(checkpoint, constraint, map_location=torch_device)
    model_loaded = model_loaded.to(torch_device).to(dtype=dtype)
    sigma = complement_jacobian_spectral_proxy(
        lambda z: model_loaded(z),
        constraint,
        x,
        eps=eps,
        steps=steps,
    )
    return float(sigma.detach().cpu())


def _synchronize(device: str | torch.device) -> None:
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(torch_device)


def _reset_peak_memory(device: str | torch.device) -> None:
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch_device)


def _peak_memory_mb(device: str | torch.device) -> float | None:
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated(torch_device) / 1_000_000.0)


if __name__ == "__main__":
    main()
