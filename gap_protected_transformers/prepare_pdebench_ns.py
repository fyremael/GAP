"""Prepare a matched incompressible Navier-Stokes benchmark slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .grid_operators import (
    sparse_periodic_grid_divergence_2d_central,
    velocity_from_streamfunction,
)
from .make_constraint import write_sparse_coo_npz


PDEBENCH_REPO_URL = "https://github.com/pdebench/PDEBench"
PDEBENCH_NS_GENERATOR = (
    "https://github.com/pdebench/PDEBench/blob/main/"
    "pdebench/data_gen/gen_ns_incomp.py"
)


def prepare_pdebench_ns_velocity_pairs(
    *,
    source: str | Path | None = None,
    output: str | Path,
    constraint_output: str | Path,
    resolution: int = 64,
    pairs: int = 1024,
    dx: float = 1.0,
    dy: float = 1.0,
    normalize: bool = True,
    seed: int = 0,
    input_key: str = "x",
    target_key: str = "y",
) -> dict[str, Any]:
    """Prepare channel-first divergence-free velocity pairs.

    If ``source`` is supplied, the loader accepts PDEBench-style HDF5 files with
    a ``velocity`` dataset shaped ``(batch, time, x, y, 2)`` or NPZ files with
    ``velocity``/``streamfunction``-like fields. Without a source, this creates a
    deterministic streamfunction-generated slice and records the PDEBench
    generator as provenance rather than claiming a downloaded corpus.
    """

    if resolution < 8:
        raise ValueError("resolution must be at least 8.")
    if pairs < 1:
        raise ValueError("pairs must be positive.")
    if source is None:
        fields, provenance = _generate_streamfunction_velocity(
            frames=pairs + 1,
            resolution=resolution,
            dx=dx,
            dy=dy,
            seed=seed,
        )
    else:
        fields, provenance = _load_velocity_source(
            Path(source),
            resolution=resolution,
            dx=dx,
            dy=dy,
        )
        if fields.shape[0] < pairs + 1:
            raise ValueError(
                f"source contains {fields.shape[0]} frames, need at least {pairs + 1}."
            )
        fields = fields[: pairs + 1]

    if fields.shape != (pairs + 1, 2, resolution, resolution):
        raise ValueError(
            "prepared velocity fields must have shape "
            f"{(pairs + 1, 2, resolution, resolution)}, got {fields.shape}."
        )

    velocity_mean = fields.mean(axis=(0, 2, 3), keepdims=True)
    velocity_scale = np.asarray(fields.std() + 1e-6, dtype=np.float32)
    if normalize:
        fields = (fields - velocity_mean) / float(velocity_scale)

    x = fields[:-1].astype("float32")
    y = fields[1:].astype("float32")
    output = Path(output)
    constraint_output = Path(constraint_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        **{
            input_key: x,
            target_key: y,
            "time": np.arange(pairs + 1, dtype=np.float32),
            "resolution": np.asarray([resolution, resolution], dtype=np.int64),
            "velocity_mean": velocity_mean.reshape(2).astype("float32"),
            "velocity_scale": velocity_scale.reshape(()).astype("float32"),
            "source": str(source) if source is not None else "",
        },
    )

    divergence = sparse_periodic_grid_divergence_2d_central(
        resolution,
        resolution,
        dx=dx,
        dy=dy,
        dtype=torch.float64,
    )
    write_sparse_coo_npz(constraint_output, divergence.to(dtype=torch.float32))
    divergence_stats = _divergence_stats(divergence, torch.as_tensor(y, dtype=torch.float64))
    metadata: dict[str, Any] = {
        "source": str(source) if source is not None else None,
        "source_url": PDEBENCH_REPO_URL,
        "pdebench_generator": PDEBENCH_NS_GENERATOR,
        "provenance": provenance,
        "output": str(output),
        "constraint_output": str(constraint_output),
        "pairs": int(pairs),
        "field_shape": [2, resolution, resolution],
        "normalized": bool(normalize),
        "normalization": {
            "velocity_mean": velocity_mean.reshape(2).astype(float).tolist(),
            "velocity_scale": float(velocity_scale),
            "shared_scalar_velocity_scale": True,
        },
        "dx": float(dx),
        "dy": float(dy),
        "divergence_stats": divergence_stats,
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def _generate_streamfunction_velocity(
    *,
    frames: int,
    resolution: int,
    dx: float,
    dy: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    coords = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    mode_count = 36
    kx = rng.integers(1, 10, size=mode_count)
    ky = rng.integers(1, 10, size=mode_count)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=mode_count)
    omega = rng.normal(0.0, 0.12, size=mode_count)
    amplitude = rng.normal(0.0, 1.0, size=mode_count) / (kx * kx + ky * ky)
    viscosity = 0.0015
    dt = 0.05
    psi_frames = []
    for frame in range(frames):
        t = frame * dt
        psi = np.zeros((resolution, resolution), dtype=np.float64)
        for amp, mx, my, ph, om in zip(amplitude, kx, ky, phase, omega):
            decay = np.exp(-viscosity * float(mx * mx + my * my) * t)
            psi += amp * decay * np.sin(mx * xx + my * yy + ph + om * t)
        psi_frames.append(psi)
    psi_tensor = torch.as_tensor(np.stack(psi_frames), dtype=torch.float64)
    velocity = velocity_from_streamfunction(psi_tensor, dx=dx, dy=dy).numpy()
    provenance = {
        "method": "deterministic_streamfunction_surrogate",
        "seed": int(seed),
        "frames": int(frames),
        "mode_count": int(mode_count),
        "dt": float(dt),
        "viscosity": float(viscosity),
        "note": (
            "Self-contained divergence-free slice generated from periodic "
            "streamfunctions; use --source to prepare official PDEBench HDF5 output."
        ),
    }
    return velocity.astype("float32"), provenance


def _load_velocity_source(
    source: Path,
    *,
    resolution: int,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if source.suffix.lower() in {".h5", ".hdf5"}:
        try:
            import h5py  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Reading PDEBench HDF5 files requires h5py. Install "
                "`pip install -e .[benchmark]`."
            ) from exc
        with h5py.File(source, "r") as handle:
            if "velocity" not in handle:
                raise KeyError("PDEBench HDF5 source must contain a 'velocity' dataset.")
            velocity = np.asarray(handle["velocity"], dtype=np.float32)
        fields = _coerce_velocity_frames(velocity, resolution)
        return fields, {"method": "pdebench_hdf5_velocity", "source": str(source)}

    if source.suffix.lower() != ".npz":
        raise ValueError("source must be .npz, .h5, or .hdf5.")
    with np.load(source) as data:
        if "velocity" in data:
            fields = _coerce_velocity_frames(np.asarray(data["velocity"]), resolution)
            return fields, {"method": "npz_velocity", "source": str(source)}
        for key in ("streamfunction", "psi"):
            if key in data:
                psi = torch.as_tensor(data[key], dtype=torch.float64)
                velocity = velocity_from_streamfunction(psi, dx=dx, dy=dy).numpy()
                return _coerce_velocity_frames(velocity, resolution), {
                    "method": f"npz_{key}_reconstruction",
                    "source": str(source),
                }
        if {"u", "v"}.issubset(data.files):
            velocity = np.stack([data["u"], data["v"]], axis=-3)
            return _coerce_velocity_frames(velocity, resolution), {
                "method": "npz_uv_channels",
                "source": str(source),
            }
    raise KeyError("source must contain velocity, streamfunction/psi, or u/v arrays.")


def _coerce_velocity_frames(array: np.ndarray, resolution: int) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 5 and arr.shape[-1] == 2:
        arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3], 2)
    if arr.ndim == 4 and arr.shape[-1] == 2:
        arr = np.moveaxis(arr, -1, 1)
    elif arr.ndim == 4 and arr.shape[1] == 2:
        pass
    else:
        raise ValueError(
            "velocity arrays must have shape (T,H,W,2), (T,2,H,W), "
            "or PDEBench (B,T,H,W,2)."
        )
    if arr.shape[2:] != (resolution, resolution):
        raise ValueError(
            f"velocity resolution {arr.shape[2:]} does not match {resolution}x{resolution}."
        )
    return arr.astype("float32")


def _divergence_stats(A: torch.Tensor, fields: torch.Tensor) -> dict[str, float]:
    flat = fields.reshape(fields.shape[0], -1)
    residual = torch.sparse.mm(A.coalesce(), flat.T).T
    abs_norm = torch.linalg.norm(residual, dim=-1)
    rel_norm = abs_norm / (torch.linalg.norm(flat, dim=-1) + 1e-12)
    return {
        "target_constraint_violation_mean": float(abs_norm.mean().cpu()),
        "target_constraint_violation_max": float(abs_norm.max().cpu()),
        "target_constraint_violation_relative_mean": float(rel_norm.mean().cpu()),
        "target_constraint_violation_relative_max": float(rel_norm.max().cpu()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None)
    parser.add_argument(
        "--output",
        default="external_data/pdebench_ns/ns_incomp_64x64_pairs.npz",
    )
    parser.add_argument(
        "--constraint-output",
        default="external_data/pdebench_ns/ns_incomp_64x64_divergence.npz",
    )
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--pairs", type=int, default=1024)
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--dy", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args(argv)

    metadata = prepare_pdebench_ns_velocity_pairs(
        source=args.source,
        output=args.output,
        constraint_output=args.constraint_output,
        resolution=args.resolution,
        pairs=args.pairs,
        dx=args.dx,
        dy=args.dy,
        normalize=not args.no_normalize,
        seed=args.seed,
    )
    stats = metadata["divergence_stats"]
    print(
        f"prepared {metadata['pairs']} pairs shape={tuple(metadata['field_shape'])} "
        f"relative_divergence_mean={stats['target_constraint_violation_relative_mean']:.3g} "
        f"output={metadata['output']}"
    )


if __name__ == "__main__":
    main()
