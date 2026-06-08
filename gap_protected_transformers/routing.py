"""Learned router maps for protected-mode compatibility diagnostics."""

from __future__ import annotations

import torch
from torch import nn

from .diagnostics import routing_commutator_error
from .operators import LinearConstraint


class CompatibleRouter(nn.Module):
    """Learn expert dispatch matrices that do not mix protected/complement modes.

    Each expert block is parameterized as ``P_K W P_K + P_perp W P_perp``.
    This is sufficient for ``R P_K = P_K^expert R`` and leaves both subspaces
    internally learnable.
    """

    def __init__(self, constraint: LinearConstraint, num_experts: int) -> None:
        super().__init__()
        self.constraint = constraint
        self.num_experts = num_experts
        self.raw = nn.Parameter(
            torch.zeros(
                num_experts,
                constraint.dim,
                constraint.dim,
                dtype=constraint.A.dtype,
                device=constraint.A.device,
            )
        )
        self.register_buffer("P_kernel", constraint.kernel_projector())
        self.register_buffer("P_complement", constraint.complement_projector())

    def matrix(self) -> torch.Tensor:
        """Return the stacked router matrix with shape ``(E * dim, dim)``."""

        blocks = []
        for raw in self.raw:
            block = self.P_kernel @ raw @ self.P_kernel
            block = block + self.P_complement @ raw @ self.P_complement
            blocks.append(block)
        return torch.cat(blocks, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route row-vector states into concatenated expert streams."""

        return x @ self.matrix().T

    def commutator_error(self) -> torch.Tensor:
        """Return normalized router/projection commutator error."""

        return routing_commutator_error(
            self.matrix(), self.constraint, num_experts=self.num_experts
        )


class UnconstrainedRouter(nn.Module):
    """Learn a stacked router matrix with no protected-mode constraint."""

    def __init__(
        self, dim: int, num_experts: int, *, dtype: torch.dtype = torch.float64
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.raw = nn.Parameter(torch.zeros(num_experts * dim, dim, dtype=dtype))

    def matrix(self) -> torch.Tensor:
        """Return the stacked unconstrained router matrix."""

        return self.raw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route row-vector states into concatenated expert streams."""

        return x @ self.raw.T
