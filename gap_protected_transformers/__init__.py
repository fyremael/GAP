"""Minimal scaffold for gap-protected Transformer experiments."""

from .blocks import (
    DiagnosticOutput,
    GapProtectedBlock,
    MLPCore,
    OutputProjectionWrapper,
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
from .projections import decompose, reconstruction_error
from .hierarchy import CompatibleRestriction, UnconstrainedRestriction
from .routing import CompatibleRouter, UnconstrainedRouter
from .data import ArrayPairDataset, load_array_pair_dataset, load_constraint, split_dataset
from .models import ModelConfig, VectorOperatorModel, build_model
from .logging_utils import MetricLogger
from .training import (
    TrainingConfig,
    evaluate_supervised,
    load_trained_model,
    train_supervised,
)

__all__ = [
    "ArrayPairDataset",
    "CompatibleRestriction",
    "CompatibleRouter",
    "DiagnosticOutput",
    "GapProtectedBlock",
    "ImplicitLinearConstraint",
    "LinearConstraint",
    "MLPCore",
    "MetricLogger",
    "ModelConfig",
    "OutputProjectionWrapper",
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
    "protected_energy",
    "protected_leakage",
    "reconstruction_error",
    "rollout_diagnostics",
    "routing_commutator_error",
    "split_dataset",
    "evaluate_supervised",
    "train_supervised",
]
