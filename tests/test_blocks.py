from __future__ import annotations

import torch

from gap_protected_transformers import GapProtectedBlock, LinearConstraint, MLPCore
from gap_protected_transformers.blocks import ProjectedResidualBlock
from gap_protected_transformers.toy_complexes import cycle_graph_incidence


def test_projected_residual_block_outputs_kernel_state() -> None:
    torch.manual_seed(1)
    A = cycle_graph_incidence(6)
    constraint = LinearConstraint(A)
    block = ProjectedResidualBlock(
        constraint.dim,
        constraint,
        MLPCore(constraint.dim, hidden_dim=16),
    ).to(dtype=torch.float64)
    x = torch.randn(10, constraint.dim, dtype=torch.float64)

    y = block(x)

    assert torch.max(constraint.violation(y)) < 1e-8


def test_gap_protected_block_does_not_leak_from_pure_kernel_input() -> None:
    torch.manual_seed(2)
    A = cycle_graph_incidence(6)
    constraint = LinearConstraint(A)
    block = GapProtectedBlock(
        constraint.dim,
        constraint,
        MLPCore(constraint.dim, hidden_dim=16),
        zero_preserving=True,
    ).to(dtype=torch.float64)
    x = torch.randn(10, constraint.dim, dtype=torch.float64)
    x_kernel = constraint.project_kernel(x)

    y = block(x_kernel)

    assert torch.max(constraint.violation(y)) < 1e-8
    assert torch.allclose(y, x_kernel, atol=1e-8)


def test_gap_protected_block_can_emit_protected_readout_with_zero_core() -> None:
    torch.manual_seed(3)
    A = cycle_graph_incidence(6)
    constraint = LinearConstraint(A)
    block = GapProtectedBlock(
        constraint.dim,
        constraint,
        MLPCore(constraint.dim, hidden_dim=16, zero_init=True),
        zero_preserving=True,
    ).to(dtype=torch.float64)
    x = torch.randn(10, constraint.dim, dtype=torch.float64)
    target = constraint.project_kernel(x)

    y = block(x)

    assert torch.max(constraint.violation(y)) < 1e-8
    assert torch.allclose(y, target, atol=1e-8)


def test_gap_protected_block_can_transport_complement_residually() -> None:
    torch.manual_seed(4)
    A = cycle_graph_incidence(6)
    constraint = LinearConstraint(A)
    block = GapProtectedBlock(
        constraint.dim,
        constraint,
        MLPCore(constraint.dim, hidden_dim=16, zero_init=True),
        zero_preserving=True,
        complement_residual=True,
    ).to(dtype=torch.float64)
    x = torch.randn(10, constraint.dim, dtype=torch.float64)

    y = block(x)

    assert torch.allclose(y, x, atol=1e-8)
