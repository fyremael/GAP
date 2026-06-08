from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from gap_protected_transformers import (
    ModelConfig,
    TrainingConfig,
    build_model,
    load_trained_model,
    load_array_pair_dataset,
    load_constraint,
    split_dataset,
    train_supervised,
)
from gap_protected_transformers.evaluate import main as evaluate_main
from gap_protected_transformers.train import main as train_main


def test_disk_dataset_model_and_trainer_smoke(tmp_path) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(24, 8)).astype("float32")
    y = (0.5 * x).astype("float32")
    A = np.array(
        [
            [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype="float32",
    )
    data_path = tmp_path / "pairs.npz"
    constraint_path = tmp_path / "constraint.npy"
    np.savez(data_path, x=x, y=y)
    np.save(constraint_path, A)

    dataset = load_array_pair_dataset(data_path)
    constraint = load_constraint(constraint_path)
    train_subset, val_subset = split_dataset(dataset, val_fraction=0.25, seed=1)
    train_loader = DataLoader(train_subset, batch_size=8)
    val_loader = DataLoader(val_subset, batch_size=8)
    model_config = ModelConfig(
        dim=constraint.dim,
        variant="split_gap_protected",
        core_type="transformer",
        depth=1,
        num_tokens=4,
        num_heads=1,
    )
    model = build_model(model_config, constraint)
    summary = train_supervised(
        model,
        constraint,
        train_loader,
        val_loader,
        config=TrainingConfig(
            epochs=2,
            batch_size=8,
            lr=1e-3,
            output_dir=str(tmp_path / "run"),
            rollout_steps=2,
        ),
        model_config=model_config.to_dict(),
    )

    assert summary["best_epoch"] in {1, 2}
    assert (tmp_path / "run" / "best.pt").exists()
    assert (tmp_path / "run" / "last.pt").exists()
    assert (tmp_path / "run" / "history.csv").exists()
    assert (tmp_path / "run" / "metrics.jsonl").exists()
    assert "constraint_violation_mean" in summary["final_metrics"]

    reloaded_model, checkpoint = load_trained_model(
        tmp_path / "run" / "best.pt", constraint
    )
    assert checkpoint["epoch"] in {1, 2}
    assert reloaded_model(dataset[0][0].unsqueeze(0)).shape == (1, constraint.dim)


def test_model_factory_rejects_bad_tokenization() -> None:
    A = torch.zeros(1, 10)
    constraint = load_constraint_from_tensor(A)
    config = ModelConfig(dim=10, core_type="transformer", num_tokens=4)

    try:
        build_model(config, constraint)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("Expected invalid tokenization to raise ValueError.")


def load_constraint_from_tensor(A: torch.Tensor):
    from gap_protected_transformers import LinearConstraint

    return LinearConstraint(A)


def test_train_and_evaluate_cli_smoke(tmp_path) -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(20, 8)).astype("float32")
    y = (0.25 * x).astype("float32")
    A = np.array([[1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype="float32")
    data_path = tmp_path / "pairs.npz"
    constraint_path = tmp_path / "constraint.npy"
    run_dir = tmp_path / "cli_run"
    metrics_path = tmp_path / "eval_metrics.json"
    np.savez(data_path, x=x, y=y)
    np.save(constraint_path, A)

    train_main(
        [
            "--data",
            str(data_path),
            "--constraint",
            str(constraint_path),
            "--variant",
            "split_gap_protected",
            "--core",
            "transformer",
            "--depth",
            "1",
            "--num-tokens",
            "4",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--output-dir",
            str(run_dir),
        ]
    )
    evaluate_main(
        [
            "--checkpoint",
            str(run_dir / "best.pt"),
            "--data",
            str(data_path),
            "--constraint",
            str(constraint_path),
            "--output",
            str(metrics_path),
        ]
    )

    assert (run_dir / "best.pt").exists()
    assert metrics_path.exists()
