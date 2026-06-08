"""Generate sparse benchmark constraint files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .grid_operators import (
    sparse_periodic_grid_divergence_2d_central,
    sparse_mean_zero_constraint,
    sparse_periodic_grid_divergence_2d,
)


def write_sparse_coo_npz(path: str | Path, matrix: torch.Tensor) -> None:
    """Write a sparse COO matrix using row/col/value/shape arrays."""

    coalesced = matrix.coalesce().cpu()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        target,
        row=coalesced.indices()[0].numpy(),
        col=coalesced.indices()[1].numpy(),
        value=coalesced.values().numpy(),
        shape=np.asarray(coalesced.shape, dtype=np.int64),
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for sparse constraint generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=("grid-divergence-2d", "grid-divergence-2d-central", "mean-zero"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--dy", type=float, default=1.0)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--float64", action="store_true")
    args = parser.parse_args(argv)

    dtype = torch.float64 if args.float64 else torch.float32
    if args.kind in {"grid-divergence-2d", "grid-divergence-2d-central"}:
        if args.nx is None or args.ny is None:
            raise ValueError("--nx and --ny are required for grid divergence.")
        if args.kind == "grid-divergence-2d-central":
            matrix = sparse_periodic_grid_divergence_2d_central(
                args.nx,
                args.ny,
                dx=args.dx,
                dy=args.dy,
                dtype=dtype,
            )
        else:
            matrix = sparse_periodic_grid_divergence_2d(args.nx, args.ny, dtype=dtype)
    else:
        if args.dim is None:
            raise ValueError("--dim is required for mean-zero.")
        matrix = sparse_mean_zero_constraint(args.dim, dtype=dtype)
    write_sparse_coo_npz(args.output, matrix)
    print(f"wrote {args.output} shape={tuple(matrix.shape)} nnz={matrix._nnz()}")


if __name__ == "__main__":
    main()
