"""Minimal scaffold for gap-protected Transformer experiments."""

from .blocks import (
    DiagnosticOutput,
    GapProtectedBlock,
    MLPCore,
    OutputProjectionWrapper,
    PatchTransformerCore,
    ProjectedResidualBlock,
    ResidualBlock,
    TokenTransformerCore,
    TinyMoECore,
    TinySelfAttentionCore,
)
from .diagnostics import (
    commutator_error,
    complement_energy,
    complement_jacobian_spectral_proxy,
    complement_leakage,
    constraint_violation,
    constraint_violation_stats,
    gap_proxy,
    protected_energy,
    protected_leakage,
    rollout_diagnostics,
    routing_commutator_error,
)
from .operators import ImplicitLinearConstraint, LinearConstraint
from .matrix_free import MatrixFreeLinearConstraint
from .solvers import CGResult, conjugate_gradient
from .projections import decompose, reconstruction_error
from .hierarchy import CompatibleRestriction, UnconstrainedRestriction
from .routing import CompatibleRouter, UnconstrainedRouter
from .data import (
    ArrayPairDataset,
    HDF5PairDataset,
    load_array_pair_dataset,
    load_constraint,
    split_dataset,
)
from .models import ModelConfig, VectorOperatorModel, build_model
from .logging_utils import MetricLogger
from .training import (
    TrainingConfig,
    evaluate_supervised,
    load_trained_model,
    train_supervised,
)

_VISUAL_EXPORTS = {
    "plot_divergence_residual_sample",
    "plot_benchmark_summary",
    "plot_field_sample",
    "plot_prediction_sample",
}


def __getattr__(name: str):
    if name in _VISUAL_EXPORTS:
        from . import visualize

        return getattr(visualize, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ArrayPairDataset",
    "CGResult",
    "CompatibleRestriction",
    "CompatibleRouter",
    "DiagnosticOutput",
    "GapProtectedBlock",
    "HDF5PairDataset",
    "ImplicitLinearConstraint",
    "LinearConstraint",
    "MLPCore",
    "MatrixFreeLinearConstraint",
    "MetricLogger",
    "ModelConfig",
    "OutputProjectionWrapper",
    "PatchTransformerCore",
    "ProjectedResidualBlock",
    "ResidualBlock",
    "TokenTransformerCore",
    "TrainingConfig",
    "TinyMoECore",
    "TinySelfAttentionCore",
    "UnconstrainedRestriction",
    "UnconstrainedRouter",
    "VectorOperatorModel",
    "build_model",
    "conjugate_gradient",
    "commutator_error",
    "complement_energy",
    "complement_jacobian_spectral_proxy",
    "complement_leakage",
    "constraint_violation",
    "constraint_violation_stats",
    "decompose",
    "gap_proxy",
    "load_array_pair_dataset",
    "load_constraint",
    "load_trained_model",
    "plot_benchmark_summary",
    "plot_divergence_residual_sample",
    "plot_field_sample",
    "plot_prediction_sample",
    "protected_energy",
    "protected_leakage",
    "reconstruction_error",
    "rollout_diagnostics",
    "routing_commutator_error",
    "split_dataset",
    "evaluate_supervised",
    "train_supervised",
]
