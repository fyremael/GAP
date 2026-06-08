"""Sparse grid operators for benchmark-scale field constraints."""

from __future__ import annotations

import torch


def sparse_periodic_grid_divergence_2d(
    nx: int,
    ny: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return sparse COO divergence for periodic 2D cell-centered velocity.

    The flattened state stores ``u`` followed by ``v`` with shape
    ``(2 * nx * ny,)``. Each residual row computes
    ``u[i,j] - u[i-1,j] + v[i,j] - v[i,j-1]``.
    """

    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be at least 2.")
    cells = nx * ny
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def cell(i: int, j: int) -> int:
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            row = cell(i, j)
            entries = (
                (cell(i, j), 1.0),
                (cell((i - 1) % nx, j), -1.0),
                (cells + cell(i, j), 1.0),
                (cells + cell(i, (j - 1) % ny), -1.0),
            )
            for col, value in entries:
                rows.append(row)
                cols.append(col)
                vals.append(value)

    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values = torch.tensor(vals, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(cells, 2 * cells),
        dtype=dtype,
        device=device,
    ).coalesce()


def sparse_periodic_grid_divergence_2d_central(
    nx: int,
    ny: int,
    *,
    dx: float = 1.0,
    dy: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return centered periodic divergence for channel-first 2D velocity fields.

    The flattened state stores ``u`` followed by ``v`` with shape
    ``(2 * nx * ny,)``. Each residual row computes
    ``(u[i+1,j] - u[i-1,j]) / (2 dx) + (v[i,j+1] - v[i,j-1]) / (2 dy)``.
    """

    if nx < 3 or ny < 3:
        raise ValueError("centered divergence needs nx and ny to be at least 3.")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive.")
    cells = nx * ny
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def cell(i: int, j: int) -> int:
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            row = cell(i, j)
            entries = (
                (cell((i + 1) % nx, j), 0.5 / dx),
                (cell((i - 1) % nx, j), -0.5 / dx),
                (cells + cell(i, (j + 1) % ny), 0.5 / dy),
                (cells + cell(i, (j - 1) % ny), -0.5 / dy),
            )
            for col, value in entries:
                rows.append(row)
                cols.append(col)
                vals.append(value)

    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values = torch.tensor(vals, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(cells, 2 * cells),
        dtype=dtype,
        device=device,
    ).coalesce()


def velocity_from_streamfunction(
    streamfunction: torch.Tensor,
    *,
    dx: float = 1.0,
    dy: float = 1.0,
) -> torch.Tensor:
    """Return channel-first velocity ``(u, v)`` from a periodic streamfunction."""

    if streamfunction.ndim < 2:
        raise ValueError("streamfunction must have at least two spatial dimensions.")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive.")
    dpsi_dy = (
        torch.roll(streamfunction, shifts=-1, dims=-1)
        - torch.roll(streamfunction, shifts=1, dims=-1)
    ) / (2.0 * dy)
    dpsi_dx = (
        torch.roll(streamfunction, shifts=-1, dims=-2)
        - torch.roll(streamfunction, shifts=1, dims=-2)
    ) / (2.0 * dx)
    return torch.stack([dpsi_dy, -dpsi_dx], dim=-3)


def sparse_mean_zero_constraint(
    dim: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return sparse one-row constraint enforcing zero global mean."""

    indices = torch.stack(
        [
            torch.zeros(dim, dtype=torch.long, device=device),
            torch.arange(dim, dtype=torch.long, device=device),
        ]
    )
    values = torch.full((dim,), 1.0 / dim, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(1, dim),
        dtype=dtype,
        device=device,
    ).coalesce()
