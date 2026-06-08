from __future__ import annotations

import torch

from gap_protected_transformers import LinearConstraint, decompose, reconstruction_error


def test_kernel_projection_satisfies_constraint_and_reconstructs() -> None:
    torch.manual_seed(0)
    A = torch.tensor(
        [[1.0, -1.0, 0.0, 0.0], [0.0, 1.0, -1.0, 1.0]],
        dtype=torch.float64,
    )
    constraint = LinearConstraint(A)
    x = torch.randn(16, 4, dtype=torch.float64)

    x_kernel, x_complement = decompose(x, constraint)

    assert torch.max(torch.abs(constraint.apply(x_kernel))) < 1e-9
    assert reconstruction_error(x, x_kernel, x_complement) < 1e-12
    assert torch.allclose(x_complement, constraint.project_complement(x), atol=1e-9)


def test_projectors_are_idempotent() -> None:
    A = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float64)
    constraint = LinearConstraint(A)
    P_kernel = constraint.kernel_projector()
    P_complement = constraint.complement_projector()

    assert torch.allclose(P_kernel @ P_kernel, P_kernel, atol=1e-9)
    assert torch.allclose(P_complement @ P_complement, P_complement, atol=1e-9)
    assert torch.allclose(P_kernel + P_complement, torch.eye(3, dtype=torch.float64))
