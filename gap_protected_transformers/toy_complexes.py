"""Small dense graph and grid complexes used by sanity experiments."""

from __future__ import annotations

import torch


def cycle_graph_incidence(
    num_nodes: int, *, dtype: torch.dtype = torch.float64, device: torch.device | str | None = None
) -> torch.Tensor:
    """Return an oriented incidence matrix for a cycle graph."""

    if num_nodes < 3:
        raise ValueError("A cycle graph needs at least three nodes.")
    B = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=device)
    for edge in range(num_nodes):
        tail = edge
        head = (edge + 1) % num_nodes
        B[tail, edge] = -1.0
        B[head, edge] = 1.0
    return B


def path_graph_incidence(
    num_nodes: int, *, dtype: torch.dtype = torch.float64, device: torch.device | str | None = None
) -> torch.Tensor:
    """Return an oriented incidence matrix for a path graph."""

    if num_nodes < 2:
        raise ValueError("A path graph needs at least two nodes.")
    B = torch.zeros((num_nodes, num_nodes - 1), dtype=dtype, device=device)
    for edge in range(num_nodes - 1):
        B[edge, edge] = -1.0
        B[edge + 1, edge] = 1.0
    return B


def grid_divergence_operator(
    nx: int,
    ny: int,
    *,
    periodic: bool = True,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a finite-difference divergence matrix for a 2D velocity grid.

    The state vector stores all horizontal components followed by all vertical
    components. With periodic boundaries, each cell row computes
    ``u[i, j] - u[i-1, j] + v[i, j] - v[i, j-1]``.
    """

    if nx < 2 or ny < 2:
        raise ValueError("Grid divergence needs nx >= 2 and ny >= 2.")
    cells = nx * ny
    A = torch.zeros((cells, 2 * cells), dtype=dtype, device=device)

    def cell(i: int, j: int) -> int:
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            row = cell(i, j)
            A[row, cell(i, j)] += 1.0
            A[row, cells + cell(i, j)] += 1.0
            if periodic or i > 0:
                A[row, cell((i - 1) % nx, j)] -= 1.0
            if periodic or j > 0:
                A[row, cells + cell(i, (j - 1) % ny)] -= 1.0
    return A


def identity_hierarchy(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a trivial hierarchy satisfying ``A_c R = R_A A_f`` exactly."""

    state_dim = A.shape[1]
    residual_dim = A.shape[0]
    return (
        A.clone(),
        torch.eye(state_dim, dtype=A.dtype, device=A.device),
        torch.eye(residual_dim, dtype=A.dtype, device=A.device),
        A.clone(),
    )


def cycle_graph_hierarchy(
    num_fine_nodes: int,
    *,
    compatible: bool = True,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fine/coarse cycle restrictions for hierarchy diagnostics.

    The compatible map keeps the fine edges crossing aggregate-node boundaries.
    The incompatible map averages each pair of fine edges, which mixes internal
    and boundary edges and therefore fails ``A_c R = R_A A_f``.
    """

    if num_fine_nodes % 2 != 0:
        raise ValueError("num_fine_nodes must be even for pair coarsening.")
    if num_fine_nodes < 4:
        raise ValueError("Need at least four fine nodes for coarsening.")

    num_coarse_nodes = num_fine_nodes // 2
    A_f = cycle_graph_incidence(num_fine_nodes, dtype=dtype, device=device)
    A_c = cycle_graph_incidence(num_coarse_nodes, dtype=dtype, device=device)
    R_A = torch.zeros(
        (num_coarse_nodes, num_fine_nodes), dtype=dtype, device=device
    )
    R = torch.zeros(
        (num_coarse_nodes, num_fine_nodes), dtype=dtype, device=device
    )
    for coarse in range(num_coarse_nodes):
        R_A[coarse, 2 * coarse] = 1.0
        R_A[coarse, 2 * coarse + 1] = 1.0
        if compatible:
            R[coarse, (2 * coarse + 1) % num_fine_nodes] = 1.0
        else:
            R[coarse, 2 * coarse] = 0.5
            R[coarse, (2 * coarse + 1) % num_fine_nodes] = 0.5
    return A_c, R, R_A, A_f


def componentwise_router(
    dim: int,
    num_experts: int,
    *,
    weights: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a router that copies every component to experts with scalar weights."""

    if weights is None:
        weights = torch.linspace(1.0, 2.0, num_experts, dtype=dtype, device=device)
        weights = weights / weights.sum()
    weights = weights.to(dtype=dtype, device=device)
    blocks = [weight * torch.eye(dim, dtype=dtype, device=device) for weight in weights]
    return torch.cat(blocks, dim=0)


def random_router(
    dim: int,
    num_experts: int,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return an unconstrained linear router for commutator diagnostics."""

    generator = torch.Generator(device=device) if device is not None else torch.Generator()
    generator.manual_seed(seed)
    return torch.randn(
        num_experts * dim,
        dim,
        generator=generator,
        dtype=dtype,
        device=device,
    ) / dim**0.5
