from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_phase1_pdebench_matched_divergence_artifacts_are_consistent() -> None:
    data_path = Path("external_data/pdebench_ns/ns_incomp_64x64_pairs.npz")
    validation_path = Path("runs/pdebench_ns_64x64_divergence/validation.json")
    summary_path = Path("runs/pdebench_ns_64x64_divergence/benchmark_summary.json")
    if not data_path.exists() or not summary_path.exists():
        pytest.skip("Phase 1 PDEBench-compatible benchmark artifacts are not present")

    with np.load(data_path) as data:
        assert data["x"].shape == (1024, 2, 64, 64)
        assert data["y"].shape == (1024, 2, 64, 64)
        assert tuple(data["resolution"]) == (64, 64)

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["ok"]
    assert validation["flat_dim"] == 8192
    assert validation["constraint_codim"] == 4096
    assert validation["target_constraint_violation_relative_mean"] <= 1e-4
    assert validation["target_constraint_violation_relative_max"] <= 1e-2

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = {row["variant"]: row for row in summary["results"]}
    assert set(results) == {
        "vanilla",
        "soft_penalty",
        "output_projection",
        "layerwise_projection",
        "split_gap_protected",
    }
    for row in results.values():
        assert row["train_wall_seconds"] > 0.0
        assert row["eval_wall_seconds"] > 0.0
        assert row["peak_memory_mb"] is not None
        assert row["complement_sigma_proxy"] is not None

    assert results["split_gap_protected"]["constraint_violation_relative_mean"] < 1e-6
    assert (
        results["split_gap_protected"]["rollout_protected_relative_drift"]
        < results["vanilla"]["rollout_protected_relative_drift"]
    )

    visual_dir = Path("runs/pdebench_ns_64x64_divergence/visuals")
    for name in (
        "benchmark_metrics.png",
        "component_losses.png",
        "field_sample.png",
        "prediction_sample.png",
        "divergence_residual_sample.png",
    ):
        assert (visual_dir / name).stat().st_size > 0
