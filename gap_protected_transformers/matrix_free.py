"""Matrix-free linear constraints and projectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .solvers import CGResult, conjugate_gradient


@dataclass
class MatrixFreeLinearConstraint:
    """Linear constraint represented by apply/adjoint callables.

    This class is intended for benchmark-scale operators where materializing
    ``A`` or ``P_K`` is undesirable. Projection uses the normal-equation solve
    ``(A A^* + damping I) lambda = A x`` and returns ``x - A^* lambda``.
    """

    dim: int
    codim: int
    apply_fn: Callable[[torch.Tensor], torch.Tensor]
    adjoint_fn: Callable[[torch.Tensor], torch.Tensor]
    damping: float = 1e-6
    max_iter: int = 256
    tol: float = 1e-8

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the constraint map ``A`` to row-vector states."""

        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected last dimension {self.dim}, got {x.shape[-1]}.")
        y = self.apply_fn(x)
        if y.shape[-1] != self.codim:
            raise ValueError(
                f"apply_fn returned last dimension {y.shape[-1]}, expected {self.codim}."
            )
        return y

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the Euclidean adjoint ``A^*`` to row-vector residuals."""

        if y.shape[-1] != self.codim:
            raise ValueError(f"Expected last dimension {self.codim}, got {y.shape[-1]}.")
        x = self.adjoint_fn(y)
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"adjoint_fn returned last dimension {x.shape[-1]}, expected {self.dim}."
            )
        return x

    def to(self, *args, **kwargs) -> "MatrixFreeLinearConstraint":
        """Return self; callable-backed constraints manage device inside callables."""

        return self

    def project_kernel(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto ``ker A`` via a matrix-free residual-space solve."""

        lagrange = self.solve_lagrange(self.apply(x)).x
        return x - self.adjoint(lagrange)

    def project_complement(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto the row-space complement via ``A^* lambda``."""

        lagrange = self.solve_lagrange(self.apply(x)).x
        return self.adjoint(lagrange)

    def solve_lagrange(self, residual: torch.Tensor) -> CGResult:
        """Solve the damped residual-space normal equation for ``lambda``."""

        def gram(y: torch.Tensor) -> torch.Tensor:
            out = self.apply(self.adjoint(y))
            if self.damping:
                out = out + self.damping * y
            return out

        return conjugate_gradient(
            gram,
            residual,
            max_iter=self.max_iter,
            tol=self.tol,
        )

    def violation(
        self, x: torch.Tensor, *, relative: bool = False, eps: float = 1e-12
    ) -> torch.Tensor:
        """Return ``||A x||`` for each sample, optionally normalized."""

        residual = torch.linalg.norm(self.apply(x), dim=-1)
        if not relative:
            return residual
        return residual / (torch.linalg.norm(x, dim=-1) + eps)
