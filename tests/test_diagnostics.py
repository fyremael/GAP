from __future__ import annotations

import torch

from gap_protected_transformers import (
    LinearConstraint,
    complement_jacobian_spectral_proxy,
    complement_leakage,
    constraint_violation_stats,
    gap_proxy,
    protected_leakage,
    rollout_diagnostics,
)
from gap_protected_transformers.blocks import GapProtectedBlock, MLPCore
from gap_protected_transformers.toy_complexes import cycle_graph_incidence


def test_gap_proxy_finds_cycle_graph_nullity() -> None:
    constraint = LinearConstraint(cycle_graph_incidence(7))

    gap = gap_proxy(constraint)

    assert gap["nullity"] == 1
    assert float(gap["lambda_min_plus"]) > 0.0
    assert float(gap["gap_ratio"]) > 0.0


def test_constraint_violation_stats_report_expected_keys() -> None:
    torch.manual_seed(4)
    constraint = LinearConstraint(cycle_graph_incidence(5))
    x = torch.randn(12, constraint.dim, dtype=torch.float64)
    y = constraint.project_kernel(x)

    stats = constraint_violation_stats(constraint, y)

    assert set(stats) == {
        "constraint_violation_mean",
        "constraint_violation_max",
        "constraint_violation_p95",
        "constraint_violation_final",
    }
    assert float(stats["constraint_violation_max"]) < 1e-8


def test_split_block_has_zero_protected_to_complement_probe() -> None:
    constraint = LinearConstraint(cycle_graph_incidence(5))
    block = GapProtectedBlock(
        constraint.dim,
        constraint,
        MLPCore(constraint.dim, hidden_dim=16),
        zero_preserving=True,
    ).to(dtype=torch.float64)

    assert protected_leakage(block, constraint) < 1e-8
    assert complement_leakage(torch.eye(constraint.dim, dtype=torch.float64), constraint) < 1e-8


def test_rollout_diagnostics_detect_projected_stability() -> None:
    constraint = LinearConstraint(cycle_graph_incidence(5))

    def project_once(x: torch.Tensor) -> torch.Tensor:
        return constraint.project_kernel(x)

    x = torch.randn(8, constraint.dim, dtype=torch.float64)
    metrics = rollout_diagnostics(project_once, constraint, x, steps=4)

    assert metrics["rollout_violation_max"] < 1e-8


def test_complement_jacobian_spectral_proxy_matches_scaled_complement_map() -> None:
    constraint = LinearConstraint(cycle_graph_incidence(5))
    scale = 0.25

    def scaled_complement(x: torch.Tensor) -> torch.Tensor:
        return constraint.project_kernel(x) + scale * constraint.project_complement(x)

    x = torch.randn(4, constraint.dim, dtype=torch.float64)
    sigma = complement_jacobian_spectral_proxy(
        scaled_complement, constraint, x, steps=4, seed=7
    )

    assert torch.allclose(sigma, torch.tensor(scale, dtype=torch.float64), atol=1e-4)
