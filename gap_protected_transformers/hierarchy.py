"""Learned hierarchy maps with optional compatibility by construction."""

from __future__ import annotations

import torch
from torch import nn

from .diagnostics import commutator_error


class CompatibleRestriction(nn.Module):
    """Learn a restriction map constrained by ``A_c R = R_A A_f``.

    The map has shape ``R: V_f -> V_c`` represented as a matrix with shape
    ``(dim_coarse, dim_fine)`` and applied to row-vector states by ``x @ R.T``.
    Each column is parameterized as a particular solution plus a vector in
    ``ker A_c``, which preserves the commutator exactly up to numerical error.
    """

    def __init__(
        self,
        A_c: torch.Tensor,
        R_A: torch.Tensor,
        A_f: torch.Tensor,
        *,
        rtol: float = 1e-6,
    ) -> None:
        super().__init__()
        if A_c.ndim != 2 or R_A.ndim != 2 or A_f.ndim != 2:
            raise ValueError("A_c, R_A, and A_f must be 2D tensors.")
        target = R_A @ A_f
        if A_c.shape[0] != target.shape[0]:
            raise ValueError("A_c and R_A @ A_f must have matching row counts.")

        A_c_pinv = torch.linalg.pinv(A_c, rtol=rtol)
        base = A_c_pinv @ target
        projector = torch.eye(
            A_c.shape[1], dtype=A_c.dtype, device=A_c.device
        ) - A_c_pinv @ A_c
        projector = 0.5 * (projector + projector.T)

        self.raw = nn.Parameter(torch.zeros_like(base))
        self.register_buffer("A_c", A_c)
        self.register_buffer("R_A", R_A)
        self.register_buffer("A_f", A_f)
        self.register_buffer("base", base)
        self.register_buffer("kernel_projector", projector)

    def matrix(self) -> torch.Tensor:
        """Return the compatible restriction matrix ``R``."""

        return self.base + self.kernel_projector @ self.raw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the learned restriction to row-vector fine states."""

        return x @ self.matrix().T

    def commutator_error(self) -> torch.Tensor:
        """Return normalized ``||A_c R - R_A A_f||_F``."""

        return commutator_error(self.A_c, self.matrix(), self.R_A, self.A_f)


class UnconstrainedRestriction(nn.Module):
    """Learn a restriction matrix with no hierarchy compatibility constraint."""

    def __init__(self, initial_matrix: torch.Tensor) -> None:
        super().__init__()
        if initial_matrix.ndim != 2:
            raise ValueError("initial_matrix must be a 2D tensor.")
        self.raw = nn.Parameter(initial_matrix.clone())

    def matrix(self) -> torch.Tensor:
        """Return the unconstrained restriction matrix."""

        return self.raw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the learned restriction to row-vector fine states."""

        return x @ self.raw.T
