"""Projection helpers for protected/complement decompositions."""

from __future__ import annotations

import torch

from .operators import LinearConstraint


def decompose(
    x: torch.Tensor, constraint: LinearConstraint
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a state into protected and complement components."""

    x_kernel = constraint.project_kernel(x)
    return x_kernel, x - x_kernel


def reconstruction_error(
    x: torch.Tensor, x_kernel: torch.Tensor, x_complement: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Return relative reconstruction error for ``x = x_K + x_perp``."""

    numerator = torch.linalg.norm(x - (x_kernel + x_complement))
    return numerator / (torch.linalg.norm(x) + eps)
