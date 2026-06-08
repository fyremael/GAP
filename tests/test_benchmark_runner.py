from __future__ import annotations

import json

import numpy as np

from gap_protected_transformers.benchmark import run_benchmark
from gap_protected_transformers.make_constraint import main as make_constraint_main
from gap_protected_transformers.prepare_rayleigh_benard import (
    prepare_rayleigh_benard_velocity_pairs,
)
from gap_protected_transformers.validate import validate_inputs


def test_validate_and_make_constraint_cli(tmp_path) -> None:
    x = np.ones((10, 2, 4, 4), dtype="float32")
    y = np.zeros((10, 2, 4, 4), dtype="float32")
    data_path = tmp_path / "fields.npz"
    constraint_path = tmp_path / "constraint_sparse.npz"
    np.savez(data_path, x=x, y=y)

    make_constraint_main(
        [
            "--kind",
            "grid-divergence-2d",
            "--nx",
            "4",
            "--ny",
            "4",
            "--output",
            str(constraint_path),
        ]
    )
    report = validate_inputs(
        data_path=data_path,
        constraint_path=constraint_path,
        implicit_constraint=True,
    )

    assert report["ok"]
    assert report["flat_dim"] == 32
    assert report["constraint_codim"] == 16


def test_configured_benchmark_runner_smoke(tmp_path) -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(16, 8)).astype("float32")
    y = (0.5 * x).astype("float32")
    A = np.array([[1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype="float32")
    data_path = tmp_path / "pairs.npz"
    constraint_path = tmp_path / "constraint.npy"
    output_dir = tmp_path / "benchmark"
    config_path = tmp_path / "benchmark.json"
    np.savez(data_path, x=x, y=y)
    np.save(constraint_path, A)
    config_path.write_text(
        json.dumps(
            {
                "data": str(data_path),
                "constraint": str(constraint_path),
                "output_dir": str(output_dir),
                "variants": ["vanilla", "split_gap_protected"],
                "model": {"core": "mlp", "depth": 1},
                "training": {"epochs": 1, "batch_size": 8, "lr": 1e-3},
                "spectral_proxy": {"enabled": True, "sample_count": 1, "steps": 1},
            }
        ),
        encoding="utf-8",
    )

    summary = run_benchmark(config_path)

    assert len(summary["results"]) == 2
    for row in summary["results"]:
        assert "train_wall_seconds" in row
        assert "eval_wall_seconds" in row
        assert "peak_memory_mb" in row
        assert "complement_sigma_proxy" in row
    assert (output_dir / "benchmark_summary.csv").exists()
    assert (output_dir / "vanilla" / "best.pt").exists()
    assert (output_dir / "split_gap_protected" / "eval_metrics.json").exists()


def test_prepare_rayleigh_benard_velocity_pairs(tmp_path) -> None:
    rng = np.random.default_rng(9)
    vx = rng.normal(size=(6, 8, 8))
    vy = rng.normal(size=(6, 8, 8))
    temp = rng.normal(size=(6, 8, 8))
    time = np.linspace(0.0, 1.0, 6)
    source = tmp_path / "rb.npz"
    output = tmp_path / "pairs.npz"
    constraint = tmp_path / "div.npz"
    np.savez(source, vx=vx, vy=vy, temp=temp, time=time)

    metadata = prepare_rayleigh_benard_velocity_pairs(
        source=source,
        output=output,
        constraint_output=constraint,
        mean_constraint_output=tmp_path / "mean.npz",
        resolution=4,
        max_pairs=4,
    )

    report = validate_inputs(
        data_path=output,
        constraint_path=constraint,
        implicit_constraint=True,
    )
    assert metadata["pairs"] == 4
    assert metadata["mean_constraint_output"] is not None
    assert report["ok"]
    assert report["flat_dim"] == 32
