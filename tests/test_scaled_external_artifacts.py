from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_scaled_rayleigh_benard_32x32_artifacts_are_consistent() -> None:
    data_path = Path("external_data/rayleigh_benard/rb_velocity_32x32_pairs.npz")
    summary_path = Path("runs/rayleigh_benard_32x32_mean/benchmark_summary.json")
    validation_path = Path("runs/rayleigh_benard_32x32_mean/validation.json")
    divergence_validation_path = Path(
        "runs/rayleigh_benard_32x32_divergence_validation.json"
    )
    if not data_path.exists() or not summary_path.exists():
        pytest.skip("scaled external Rayleigh-Benard artifacts are not present")

    with np.load(data_path) as data:
        assert data["x"].shape == (320, 2, 32, 32)
        assert data["y"].shape == (320, 2, 32, 32)
        assert tuple(data["resolution"]) == (32, 32)

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["ok"]
    assert validation["flat_dim"] == 2048
    assert validation["constraint_codim"] == 1

    divergence_validation = json.loads(
        divergence_validation_path.read_text(encoding="utf-8")
    )
    assert divergence_validation["ok"]
    assert divergence_validation["constraint_codim"] == 1024
    assert divergence_validation["target_constraint_violation_mean"] > 1.0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = {row["variant"]: row for row in summary["results"]}
    assert set(results) == {
        "vanilla",
        "soft_penalty",
        "output_projection",
        "layerwise_projection",
        "split_gap_protected",
    }
    assert results["split_gap_protected"]["loss"] < results["vanilla"]["loss"]
    assert (
        results["split_gap_protected"]["rollout_protected_relative_drift"]
        < results["vanilla"]["rollout_protected_relative_drift"]
    )

    visual_dir = Path("runs/rayleigh_benard_32x32_mean/visuals")
    for name in (
        "benchmark_metrics.png",
        "component_losses.png",
        "field_sample.png",
        "prediction_sample.png",
    ):
        assert (visual_dir / name).stat().st_size > 0
