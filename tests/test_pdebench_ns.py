from __future__ import annotations

import numpy as np
import torch

from gap_protected_transformers.grid_operators import (
    sparse_periodic_grid_divergence_2d_central,
    velocity_from_streamfunction,
)
from gap_protected_transformers.prepare_pdebench_ns import (
    prepare_pdebench_ns_velocity_pairs,
)
from gap_protected_transformers.validate import validate_inputs


def test_centered_divergence_annihilates_streamfunction_velocity() -> None:
    generator = torch.Generator().manual_seed(123)
    psi = torch.randn(3, 16, 16, generator=generator, dtype=torch.float64)
    velocity = velocity_from_streamfunction(psi)
    A = sparse_periodic_grid_divergence_2d_central(16, 16, dtype=torch.float64)
    residual = torch.sparse.mm(A, velocity.reshape(3, -1).T).T

    assert A.shape == (256, 512)
    assert A._nnz() == 1024
    assert torch.max(torch.abs(residual)) < 1e-10


def test_prepare_pdebench_ns_streamfunction_slice_and_validation(tmp_path) -> None:
    data_path = tmp_path / "ns_pairs.npz"
    constraint_path = tmp_path / "ns_divergence.npz"

    metadata = prepare_pdebench_ns_velocity_pairs(
        output=data_path,
        constraint_output=constraint_path,
        resolution=8,
        pairs=6,
        seed=7,
    )
    report = validate_inputs(
        data_path=data_path,
        constraint_path=constraint_path,
        implicit_constraint=True,
        sample_count=6,
    )

    with np.load(data_path) as data:
        assert data["x"].shape == (6, 2, 8, 8)
        assert data["y"].shape == (6, 2, 8, 8)
        assert data["velocity_scale"].shape == ()

    assert metadata["pairs"] == 6
    assert metadata["field_shape"] == [2, 8, 8]
    assert metadata["normalization"]["shared_scalar_velocity_scale"]
    assert metadata["divergence_stats"]["target_constraint_violation_relative_mean"] < 1e-5
    assert report["ok"]
    assert report["flat_dim"] == 128
    assert report["constraint_codim"] == 64
    assert report["target_constraint_violation_relative_mean"] < 1e-5
