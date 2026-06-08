"""Matrix-free linear solvers used by implicit projectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class CGResult:
    """Result from batched conjugate-gradient iteration."""

    x: torch.Tensor
    iterations: int
    residual_norm: torch.Tensor
    converged: bool


def conjugate_gradient(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    *,
    x0: torch.Tensor | None = None,
    max_iter: int = 256,
    tol: float = 1e-8,
    eps: float | None = None,
) -> CGResult:
    """Solve ``M x = b`` for batched row-vector right-hand sides.

    ``matvec`` must preserve the shape of ``b`` and represent a symmetric
    positive semidefinite operator on the last dimension. Damped normal
    equations should be used when the residual-space operator is singular.
    """

    if eps is None:
        eps = torch.finfo(b.dtype).eps
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    p = r.clone()
    rs_old = torch.sum(r * r, dim=-1, keepdim=True)
    residual = torch.sqrt(rs_old.max())
    if residual <= tol:
        return CGResult(x=x, iterations=0, residual_norm=residual, converged=True)

    converged = False
    iterations = 0
    for iteration in range(1, max_iter + 1):
        Ap = matvec(p)
        denom = torch.sum(p * Ap, dim=-1, keepdim=True).clamp_min(eps)
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.sum(r * r, dim=-1, keepdim=True)
        residual = torch.sqrt(rs_new.max())
        iterations = iteration
        if residual <= tol:
            converged = True
            break
        beta = rs_new / rs_old.clamp_min(eps)
        p = r + beta * p
        rs_old = rs_new
    return CGResult(
        x=x,
        iterations=iterations,
        residual_norm=residual,
        converged=converged,
    )
