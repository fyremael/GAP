from __future__ import annotations

import torch

from gap_protected_transformers import ImplicitLinearConstraint, LinearConstraint


def test_sparse_implicit_projection_matches_dense_projector() -> None:
    torch.manual_seed(10)
    A_dense = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]],
        dtype=torch.float64,
    )
    dense = LinearConstraint(A_dense)
    implicit = ImplicitLinearConstraint(
        A_dense.to_sparse_coo(), max_iter=64, tol=1e-12
    )
    x = torch.randn(12, 4, dtype=torch.float64)

    y_dense = dense.project_kernel(x)
    y_implicit = implicit.project_kernel(x)

    assert torch.max(implicit.violation(y_implicit)) < 1e-8
    assert torch.allclose(y_implicit, y_dense, atol=1e-8)


def test_implicit_projectors_reconstruct_state() -> None:
    A = torch.tensor([[1.0, -2.0, 0.0]], dtype=torch.float64).to_sparse_coo()
    constraint = ImplicitLinearConstraint(A, max_iter=64, tol=1e-12)
    x = torch.tensor([[2.0, -1.0, 3.0]], dtype=torch.float64)

    y_kernel = constraint.project_kernel(x)
    y_complement = constraint.project_complement(x)

    assert torch.allclose(y_kernel + y_complement, x, atol=1e-8)
