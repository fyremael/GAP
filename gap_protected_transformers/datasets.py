"""Synthetic datasets for small protected-mode sanity checks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .operators import LinearConstraint
from .toy_complexes import cycle_graph_incidence, grid_divergence_operator


@dataclass
class TensorDatasetBundle:
    """Train/test tensors plus the constraint used to generate them."""

    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    constraint: LinearConstraint
    metadata: dict[str, int | float | str]


def make_projection_denoising_dataset(
    constraint: LinearConstraint,
    *,
    num_samples: int = 160,
    train_fraction: float = 0.75,
    complement_scale: float = 0.35,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> TensorDatasetBundle:
    """Generate ``x = x_K + noise_perp`` with target ``y = x_K``."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    dim = constraint.dim
    raw_protected = torch.randn(num_samples, dim, generator=generator, dtype=dtype)
    raw_noise = torch.randn(num_samples, dim, generator=generator, dtype=dtype)
    constraint = constraint.to(dtype=dtype)
    protected = constraint.project_kernel(raw_protected)
    complement = constraint.project_complement(raw_noise)
    x = protected + complement_scale * complement
    y = protected
    split = int(num_samples * train_fraction)
    return TensorDatasetBundle(
        train_x=x[:split],
        train_y=y[:split],
        test_x=x[split:],
        test_y=y[split:],
        constraint=constraint,
        metadata={
            "num_samples": num_samples,
            "train_samples": split,
            "dim": dim,
            "codim": constraint.codim,
            "complement_scale": complement_scale,
        },
    )


def make_complement_dynamics_dataset(
    constraint: LinearConstraint,
    *,
    num_samples: int = 192,
    train_fraction: float = 0.75,
    complement_scale: float = 0.4,
    dynamics_scale: float = 0.7,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> TensorDatasetBundle:
    """Generate ``x_K + x_perp -> x_K + S x_perp`` complement dynamics."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    dim = constraint.dim
    constraint = constraint.to(dtype=dtype)
    raw_protected = torch.randn(num_samples, dim, generator=generator, dtype=dtype)
    raw_complement = torch.randn(num_samples, dim, generator=generator, dtype=dtype)
    protected = constraint.project_kernel(raw_protected)
    complement = complement_scale * constraint.project_complement(raw_complement)

    P_perp = constraint.complement_projector()
    random_map = torch.randn(dim, dim, generator=generator, dtype=dtype) / dim**0.5
    complement_map = 0.45 * P_perp + dynamics_scale * (P_perp @ random_map @ P_perp)
    target_complement = complement @ complement_map.T

    x = protected + complement
    y = protected + target_complement
    split = int(num_samples * train_fraction)
    return TensorDatasetBundle(
        train_x=x[:split],
        train_y=y[:split],
        test_x=x[split:],
        test_y=y[split:],
        constraint=constraint,
        metadata={
            "num_samples": num_samples,
            "train_samples": split,
            "dim": dim,
            "codim": constraint.codim,
            "complement_scale": complement_scale,
            "dynamics_scale": dynamics_scale,
            "task": "complement_dynamics",
        },
    )


def make_edge_flow_dataset(
    *,
    num_nodes: int = 8,
    num_samples: int = 160,
    complement_scale: float = 0.35,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> TensorDatasetBundle:
    """Create a cycle-graph edge-flow denoising task with divergence constraint."""

    incidence = cycle_graph_incidence(num_nodes, dtype=dtype)
    constraint = LinearConstraint(incidence)
    bundle = make_projection_denoising_dataset(
        constraint,
        num_samples=num_samples,
        complement_scale=complement_scale,
        seed=seed,
        dtype=dtype,
    )
    bundle.metadata.update({"dataset": "edge_flow_cycle", "num_nodes": num_nodes})
    return bundle


def make_edge_flow_complement_dataset(
    *,
    num_nodes: int = 8,
    num_samples: int = 192,
    complement_scale: float = 0.4,
    dynamics_scale: float = 0.7,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> TensorDatasetBundle:
    """Create a cycle-graph task with nontrivial complement dynamics."""

    incidence = cycle_graph_incidence(num_nodes, dtype=dtype)
    constraint = LinearConstraint(incidence)
    bundle = make_complement_dynamics_dataset(
        constraint,
        num_samples=num_samples,
        complement_scale=complement_scale,
        dynamics_scale=dynamics_scale,
        seed=seed,
        dtype=dtype,
    )
    bundle.metadata.update({"dataset": "edge_flow_cycle", "num_nodes": num_nodes})
    return bundle


def make_grid_divergence_dataset(
    *,
    nx: int = 4,
    ny: int = 4,
    num_samples: int = 160,
    complement_scale: float = 0.25,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> TensorDatasetBundle:
    """Create a periodic-grid velocity denoising task with divergence constraint."""

    divergence = grid_divergence_operator(nx, ny, dtype=dtype)
    constraint = LinearConstraint(divergence)
    bundle = make_projection_denoising_dataset(
        constraint,
        num_samples=num_samples,
        complement_scale=complement_scale,
        seed=seed,
        dtype=dtype,
    )
    bundle.metadata.update({"dataset": "grid_divergence", "nx": nx, "ny": ny})
    return bundle


def make_grid_divergence_complement_dataset(
    *,
    nx: int = 4,
    ny: int = 4,
    num_samples: int = 192,
    complement_scale: float = 0.35,
    dynamics_scale: float = 0.5,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> TensorDatasetBundle:
    """Create a grid-divergence task with nontrivial complement dynamics."""

    divergence = grid_divergence_operator(nx, ny, dtype=dtype)
    constraint = LinearConstraint(divergence)
    bundle = make_complement_dynamics_dataset(
        constraint,
        num_samples=num_samples,
        complement_scale=complement_scale,
        dynamics_scale=dynamics_scale,
        seed=seed,
        dtype=dtype,
    )
    bundle.metadata.update({"dataset": "grid_divergence", "nx": nx, "ny": ny})
    return bundle
