"""Download and prepare Rayleigh-Benard convection velocity benchmarks."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

from .grid_operators import (
    sparse_mean_zero_constraint,
    sparse_periodic_grid_divergence_2d,
)
from .make_constraint import write_sparse_coo_npz


RB_DATA_12E3_URL = (
    "https://huggingface.co/datasets/ashiq24/Rayleigh_Benard_Convection/"
    "resolve/main/data/data_12e3.npz"
)


def prepare_rayleigh_benard_velocity_pairs(
    *,
    source: str | Path,
    output: str | Path,
    constraint_output: str | Path | None = None,
    mean_constraint_output: str | Path | None = None,
    resolution: int = 16,
    max_pairs: int = 160,
    normalize: bool = True,
) -> dict:
    """Prepare next-step velocity pairs from a Rayleigh-Benard ``.npz`` file."""

    source = Path(source)
    output = Path(output)
    with np.load(source) as data:
        vx = data["vx"]
        vy = data["vy"]
        time = data["time"] if "time" in data else None

    if vx.shape != vy.shape or vx.ndim != 3:
        raise ValueError("Expected vx and vy arrays with shape (T, H, W).")
    if vx.shape[1] != vx.shape[2]:
        raise ValueError("Expected square velocity fields.")
    if vx.shape[1] % resolution != 0:
        raise ValueError("source resolution must be divisible by output resolution.")
    factor = vx.shape[1] // resolution
    frame_count = min(max_pairs + 1, vx.shape[0])
    vx_small = _block_average(vx[:frame_count], factor)
    vy_small = _block_average(vy[:frame_count], factor)
    fields = np.stack([vx_small, vy_small], axis=1).astype("float32")
    mean = fields.mean(axis=(0, 2, 3), keepdims=True)
    std = fields.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    if normalize:
        fields = (fields - mean) / std
    x = fields[:-1]
    y = fields[1:]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        x=x.astype("float32"),
        y=y.astype("float32"),
        time=time[:frame_count].astype("float32") if time is not None else np.arange(frame_count, dtype="float32"),
        source=str(source),
        resolution=np.asarray([resolution, resolution], dtype=np.int64),
        velocity_mean=mean.reshape(-1).astype("float32"),
        velocity_std=std.reshape(-1).astype("float32"),
    )
    if constraint_output is not None:
        matrix = sparse_periodic_grid_divergence_2d(resolution, resolution)
        write_sparse_coo_npz(constraint_output, matrix)
    if mean_constraint_output is not None:
        matrix = sparse_mean_zero_constraint(2 * resolution * resolution)
        write_sparse_coo_npz(mean_constraint_output, matrix)
    metadata = {
        "source": str(source),
        "output": str(output),
        "constraint_output": str(constraint_output) if constraint_output else None,
        "mean_constraint_output": (
            str(mean_constraint_output) if mean_constraint_output else None
        ),
        "pairs": int(x.shape[0]),
        "field_shape": tuple(int(item) for item in x.shape[1:]),
        "normalized": normalize,
        "source_frames": int(vx.shape[0]),
        "source_resolution": int(vx.shape[1]),
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def download_if_needed(url: str, path: str | Path) -> Path:
    """Download a file unless it already exists."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        urllib.request.urlretrieve(url, target)
    return target


def main(argv: list[str] | None = None) -> None:
    """CLI for downloading and preparing the Rayleigh-Benard benchmark slice."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None)
    parser.add_argument("--download-url", default=RB_DATA_12E3_URL)
    parser.add_argument(
        "--download-path",
        default="external_data/rayleigh_benard/data_12e3.npz",
    )
    parser.add_argument(
        "--output",
        default="external_data/rayleigh_benard/rb_velocity_16x16_pairs.npz",
    )
    parser.add_argument(
        "--constraint-output",
        default="external_data/rayleigh_benard/rb_velocity_16x16_divergence.npz",
    )
    parser.add_argument("--mean-constraint-output", default=None)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=160)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args(argv)

    source = Path(args.source) if args.source else download_if_needed(
        args.download_url, args.download_path
    )
    metadata = prepare_rayleigh_benard_velocity_pairs(
        source=source,
        output=args.output,
        constraint_output=args.constraint_output,
        mean_constraint_output=args.mean_constraint_output,
        resolution=args.resolution,
        max_pairs=args.max_pairs,
        normalize=not args.no_normalize,
    )
    print(
        f"prepared {metadata['pairs']} pairs shape={metadata['field_shape']} "
        f"output={metadata['output']}"
    )


def _block_average(arr: np.ndarray, factor: int) -> np.ndarray:
    frames, height, width = arr.shape
    return arr.reshape(frames, height // factor, factor, width // factor, factor).mean(
        axis=(2, 4)
    )


if __name__ == "__main__":
    main()
