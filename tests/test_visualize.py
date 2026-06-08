from __future__ import annotations

import numpy as np
import pytest

from gap_protected_transformers.visualize import (
    main as visualize_main,
    plot_benchmark_summary,
    plot_field_sample,
)


pytest.importorskip("matplotlib")


def test_visualize_summary_and_field_sample(tmp_path) -> None:
    summary_csv = tmp_path / "summary.csv"
    summary_csv.write_text(
        "\n".join(
            [
                "variant,loss,constraint_violation_mean,rollout_protected_relative_drift,"
                "protected_component_loss,complement_component_loss,rollout_violation_final",
                "vanilla,0.2,0.5,0.8,0.1,0.1,0.7",
                "split_gap_protected,0.05,0.01,0.001,0.04,0.01,0.02",
            ]
        ),
        encoding="utf-8",
    )
    data_path = tmp_path / "pairs.npz"
    x = np.zeros((2, 2, 4, 4), dtype=np.float32)
    y = np.ones((2, 2, 4, 4), dtype=np.float32)
    x[:, 0] = np.arange(16, dtype=np.float32).reshape(4, 4)
    y[:, 1] = np.arange(16, dtype=np.float32).reshape(4, 4)
    np.savez(data_path, x=x, y=y)

    summary_paths = plot_benchmark_summary(summary_csv, tmp_path / "plots")
    field_path = plot_field_sample(data_path, tmp_path / "plots" / "field.png")

    assert {path.name for path in summary_paths} == {
        "benchmark_metrics.png",
        "component_losses.png",
    }
    assert all(path.stat().st_size > 0 for path in summary_paths)
    assert field_path.stat().st_size > 0


def test_visualize_cli_writes_requested_outputs(tmp_path) -> None:
    summary_csv = tmp_path / "summary.csv"
    summary_csv.write_text(
        "\n".join(
            [
                "variant,loss,constraint_violation_mean,rollout_protected_relative_drift,"
                "protected_component_loss,complement_component_loss,rollout_violation_final",
                "vanilla,0.3,0.2,0.4,0.2,0.1,0.5",
            ]
        ),
        encoding="utf-8",
    )
    data_path = tmp_path / "pairs.npz"
    np.savez(
        data_path,
        x=np.zeros((1, 2, 4, 4), dtype=np.float32),
        y=np.ones((1, 2, 4, 4), dtype=np.float32),
    )
    output_dir = tmp_path / "visuals"

    visualize_main(
        [
            "--summary-csv",
            str(summary_csv),
            "--data",
            str(data_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert (output_dir / "benchmark_metrics.png").exists()
    assert (output_dir / "component_losses.png").exists()
    assert (output_dir / "field_sample.png").exists()
