"""Reusable model definitions for protected-mode operator learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Literal

from torch import nn

from .blocks import (
    GapProtectedBlock,
    MLPCore,
    OutputProjectionWrapper,
    ProjectedResidualBlock,
    ResidualBlock,
    TinyMoECore,
    TinySelfAttentionCore,
    TokenTransformerCore,
)
from .operators import LinearConstraint


Variant = Literal[
    "vanilla",
    "soft_penalty",
    "output_projection",
    "layerwise_projection",
    "split_gap_protected",
]
CoreType = Literal["mlp", "attention", "transformer", "moe"]


@dataclass
class ModelConfig:
    """Architecture config for supervised protected-operator models."""

    dim: int
    variant: Variant = "split_gap_protected"
    core_type: CoreType = "transformer"
    depth: int = 4
    hidden_dim: int | None = None
    num_tokens: int = 8
    num_heads: int = 1
    ffn_multiplier: int = 4
    complement_residual: bool = True
    zero_preserving: bool = True

    def to_dict(self) -> dict[str, int | str | bool | None]:
        """Return a serializable config dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Build a config from a dictionary, ignoring non-architecture metadata."""

        names = {field.name for field in fields(cls)}
        kwargs = {key: value for key, value in data.items() if key in names}
        return cls(**kwargs)


class VectorOperatorModel(nn.Module):
    """Thin named wrapper around a vector-to-vector operator stack."""

    def __init__(self, config: ModelConfig, network: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.network = network

    def forward(self, x):
        return self.network(x)


def build_model(config: ModelConfig, constraint: LinearConstraint) -> VectorOperatorModel:
    """Build a supervised vector operator model from ``ModelConfig``."""

    _validate_config(config, constraint)
    hidden_dim = config.hidden_dim or max(64, 4 * config.dim)

    if config.variant in {"vanilla", "soft_penalty", "output_projection"}:
        network: nn.Module = nn.Sequential(
            *[
                ResidualBlock(_build_core(config, hidden_dim))
                for _ in range(config.depth)
            ]
        )
        if config.variant == "output_projection":
            network = OutputProjectionWrapper(network, constraint)
        return VectorOperatorModel(config, network)

    if config.variant == "layerwise_projection":
        return VectorOperatorModel(
            config,
            nn.Sequential(
                *[
                    ProjectedResidualBlock(
                        config.dim,
                        constraint,
                        _build_core(config, hidden_dim),
                    )
                    for _ in range(config.depth)
                ]
            ),
        )

    return VectorOperatorModel(
        config,
        nn.Sequential(
            *[
                GapProtectedBlock(
                    config.dim,
                    constraint,
                    _build_core(config, hidden_dim),
                    zero_preserving=config.zero_preserving,
                    complement_residual=config.complement_residual,
                )
                for _ in range(config.depth)
            ]
        ),
    )


def _build_core(config: ModelConfig, hidden_dim: int) -> nn.Module:
    if config.core_type == "mlp":
        return MLPCore(config.dim, hidden_dim=hidden_dim)
    if config.core_type == "attention":
        return TinySelfAttentionCore(
            config.dim,
            num_tokens=config.num_tokens,
            num_heads=config.num_heads,
        )
    if config.core_type == "transformer":
        return TokenTransformerCore(
            config.dim,
            num_tokens=config.num_tokens,
            num_heads=config.num_heads,
            ffn_multiplier=config.ffn_multiplier,
        )
    if config.core_type == "moe":
        return TinyMoECore(config.dim, hidden_dim=hidden_dim, num_experts=4)
    raise ValueError(f"Unknown core_type {config.core_type!r}.")


def _validate_config(config: ModelConfig, constraint: LinearConstraint) -> None:
    if config.dim != constraint.dim:
        raise ValueError(
            f"Model dim {config.dim} does not match constraint dim {constraint.dim}."
        )
    if config.depth < 1:
        raise ValueError("depth must be at least 1.")
    if config.core_type in {"attention", "transformer"}:
        if config.dim % config.num_tokens != 0:
            raise ValueError("dim must be divisible by num_tokens.")
        token_dim = config.dim // config.num_tokens
        if token_dim % config.num_heads != 0:
            raise ValueError("token dimension must be divisible by num_heads.")
