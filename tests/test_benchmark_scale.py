from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from gap_protected_transformers import (
    MatrixFreeLinearConstraint,
    ModelConfig,
    TrainingConfig,
    build_model,
    load_array_pair_dataset,
    load_constraint,
    train_supervised,
)
from gap_protected_transformers.grid_operators import (
    sparse_periodic_grid_divergence_2d,
    sparse_periodic_grid_divergence_2d_central,
)
from gap_protected_transformers.operators import LinearConstraint
from gap_protected_transformers.solvers import conjugate_gradient


def test_conjugate_gradient_solves_batched_system() -> None:
    M = torch.diag(torch.tensor([2.0, 4.0, 8.0], dtype=torch.float64))
    b = torch.tensor([[2.0, 8.0, 16.0]], dtype=torch.float64)

    result = conjugate_gradient(lambda x: x @ M.T, b, tol=1e-12)

    assert result.converged
    assert torch.allclose(result.x, torch.tensor([[1.0, 2.0, 2.0]], dtype=torch.float64))


def test_matrix_free_projection_matches_dense_constraint() -> None:
    A = torch.tensor([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]], dtype=torch.float64)
    dense = LinearConstraint(A)
    matrix_free = MatrixFreeLinearConstraint(
        dim=3,
        codim=2,
        apply_fn=lambda x: x @ A.T,
        adjoint_fn=lambda y: y @ A,
        damping=0.0,
        tol=1e-12,
    )
    x = torch.randn(5, 3, dtype=torch.float64)

    assert torch.allclose(matrix_free.project_kernel(x), dense.project_kernel(x), atol=1e-8)


def test_sparse_grid_divergence_and_loader(tmp_path) -> None:
    A = sparse_periodic_grid_divergence_2d(4, 4)
    path = tmp_path / "constraint_sparse.npz"
    coalesced = A.coalesce()
    np.savez(
        path,
        row=coalesced.indices()[0].numpy(),
        col=coalesced.indices()[1].numpy(),
        value=coalesced.values().numpy(),
        shape=np.array(coalesced.shape),
    )

    constraint = load_constraint(path)
    x = torch.randn(3, 32)
    projected = constraint.project_kernel(x)

    assert A.layout == torch.sparse_coo
    assert constraint.dim == 32
    assert torch.max(constraint.violation(projected)) < 1e-4


def test_make_constraint_central_divergence_cli(tmp_path) -> None:
    from gap_protected_transformers.make_constraint import main as make_constraint_main

    path = tmp_path / "central_div.npz"
    make_constraint_main(
        [
            "--kind",
            "grid-divergence-2d-central",
            "--nx",
            "4",
            "--ny",
            "4",
            "--output",
            str(path),
        ]
    )
    constraint = load_constraint(path, implicit=True)
    expected = sparse_periodic_grid_divergence_2d_central(4, 4)

    assert constraint.dim == 32
    assert constraint.codim == 16
    assert expected._nnz() == 64


def test_patch_transformer_training_path(tmp_path) -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(12, 2, 4, 4)).astype("float32")
    y = (0.1 * x).astype("float32")
    data_path = tmp_path / "fields.npz"
    constraint_path = tmp_path / "mean.npy"
    np.savez(data_path, x=x, y=y)
    np.save(constraint_path, np.ones((1, 32), dtype="float32") / 32.0)

    dataset = load_array_pair_dataset(data_path)
    constraint = load_constraint(constraint_path)
    loader = DataLoader(dataset, batch_size=4)
    config = ModelConfig(
        dim=constraint.dim,
        variant="split_gap_protected",
        core_type="patch_transformer",
        depth=1,
        field_shape=(2, 4, 4),
        patch_size=2,
        patch_embed_dim=16,
        num_heads=4,
        patch_layers=1,
    )
    model = build_model(config, constraint)
    summary = train_supervised(
        model,
        constraint,
        loader,
        loader,
        config=TrainingConfig(
            epochs=1,
            batch_size=4,
            output_dir=str(tmp_path / "patch_run"),
            rollout_steps=1,
        ),
        model_config=config.to_dict(),
    )

    assert "final_metrics" in summary
    assert (tmp_path / "patch_run" / "best.pt").exists()


def test_train_cli_rollout_steps_are_serialized(tmp_path) -> None:
    from gap_protected_transformers.train import main as train_main

    rng = np.random.default_rng(13)
    x = rng.normal(size=(10, 2, 4, 4)).astype("float32")
    y = (0.2 * x).astype("float32")
    data_path = tmp_path / "fields.npz"
    constraint_path = tmp_path / "mean.npz"
    output_dir = tmp_path / "run"
    np.savez(data_path, x=x, y=y)
    A = torch.ones((1, 32), dtype=torch.float32) / 32.0
    coalesced = A.to_sparse().coalesce()
    np.savez(
        constraint_path,
        row=coalesced.indices()[0].numpy(),
        col=coalesced.indices()[1].numpy(),
        value=coalesced.values().numpy(),
        shape=np.array(coalesced.shape),
    )

    train_main(
        [
            "--data",
            str(data_path),
            "--constraint",
            str(constraint_path),
            "--implicit-constraint",
            "--variant",
            "split_gap_protected",
            "--core",
            "patch_transformer",
            "--field-shape",
            "2,4,4",
            "--patch-size",
            "2",
            "--patch-embed-dim",
            "16",
            "--patch-layers",
            "1",
            "--num-heads",
            "4",
            "--depth",
            "1",
            "--epochs",
            "1",
            "--batch-size",
            "5",
            "--rollout-steps",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )

    import json

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["training_config"]["rollout_steps"] == 2
