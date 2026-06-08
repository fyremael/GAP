"""Small neural blocks for gap-protected residual experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .operators import LinearConstraint
from .projections import decompose


@dataclass
class DiagnosticOutput:
    """Output tensor plus detached scalar diagnostics from a block forward pass."""

    y: torch.Tensor
    metrics: dict[str, torch.Tensor]


class MLPCore(nn.Module):
    """Replaceable pointwise MLP used as a tiny learned complement update."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        *,
        bias: bool = True,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(16, 2 * dim)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=bias),
        )
        if zero_init:
            final = self.net[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            if final.bias is not None:
                nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinySelfAttentionCore(nn.Module):
    """Tiny self-attention core over fixed chunks of a state vector."""

    def __init__(self, dim: int, *, num_tokens: int = 4, num_heads: int = 1) -> None:
        super().__init__()
        if dim % num_tokens != 0:
            raise ValueError("dim must be divisible by num_tokens.")
        token_dim = dim // num_tokens
        if token_dim % num_heads != 0:
            raise ValueError("token dimension must be divisible by num_heads.")
        self.dim = dim
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.norm = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, self.dim)
        tokens = flat.reshape(-1, self.num_tokens, self.token_dim)
        h = self.norm(tokens)
        y, _ = self.attn(h, h, h, need_weights=False)
        return self.out(y.reshape(-1, self.dim)).reshape(original_shape)


class TokenTransformerCore(nn.Module):
    """Small pre-norm Transformer encoder core over vector chunks."""

    def __init__(
        self,
        dim: int,
        *,
        num_tokens: int = 4,
        num_heads: int = 1,
        ffn_multiplier: int = 2,
    ) -> None:
        super().__init__()
        if dim % num_tokens != 0:
            raise ValueError("dim must be divisible by num_tokens.")
        token_dim = dim // num_tokens
        if token_dim % num_heads != 0:
            raise ValueError("token dimension must be divisible by num_heads.")
        self.dim = dim
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.attn_norm = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, ffn_multiplier * token_dim),
            nn.GELU(),
            nn.Linear(ffn_multiplier * token_dim, token_dim),
        )
        self.out_norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, self.dim)
        tokens = flat.reshape(-1, self.num_tokens, self.token_dim)
        h = self.attn_norm(tokens)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return self.out(self.out_norm(tokens.reshape(-1, self.dim))).reshape(
            original_shape
        )


class TinyMoECore(nn.Module):
    """Small dense MoE placeholder with soft expert mixing."""

    def __init__(
        self, dim: int, *, hidden_dim: int | None = None, num_experts: int = 4
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList(
            MLPCore(dim, hidden_dim=hidden_dim) for _ in range(num_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        return torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=-2)


class ResidualBlock(nn.Module):
    """Vanilla residual block ``x + core(x)`` with no constraint handling."""

    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.core(x)


class ProjectedResidualBlock(nn.Module):
    """Layerwise projected residual update ``P_K(x + core(x))``."""

    def __init__(
        self, dim: int, constraint: LinearConstraint, core: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.dim = dim
        self.constraint = constraint
        self.core = core or MLPCore(dim)

    def forward(self, x: torch.Tensor, return_metrics: bool = False):
        y_raw = x + self.core(x)
        y = self.constraint.project_kernel(y_raw)
        if not return_metrics:
            return y
        return DiagnosticOutput(
            y=y,
            metrics={
                "constraint_violation_in": self.constraint.violation(x).mean().detach(),
                "constraint_violation_raw": self.constraint.violation(y_raw).mean().detach(),
                "constraint_violation_out": self.constraint.violation(y).mean().detach(),
            },
        )


class GapProtectedBlock(nn.Module):
    """Split update that transports protected modes and learns on the complement.

    The mathematical form is ``x_K = P_K x``, ``x_perp = P_perp x``, and
    ``y = T_K x_K + P_perp core(x_perp)``. By default the core is made
    zero-preserving by subtracting ``core(0)``; this prevents a pure protected
    input from creating complement leakage through core biases.
    """

    def __init__(
        self,
        dim: int,
        constraint: LinearConstraint,
        core: nn.Module | None = None,
        *,
        protected_transport: nn.Module | Callable[[torch.Tensor], torch.Tensor] | None = None,
        zero_preserving: bool = True,
        complement_residual: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.constraint = constraint
        self.core = core or MLPCore(dim)
        self.protected_transport = protected_transport or nn.Identity()
        self.zero_preserving = zero_preserving
        self.complement_residual = complement_residual

    def forward(self, x: torch.Tensor, return_metrics: bool = False):
        x_kernel, x_complement = decompose(x, self.constraint)
        y_kernel = self.protected_transport(x_kernel)
        y_kernel = self.constraint.project_kernel(y_kernel)
        y_complement = self.core(x_complement)
        if self.zero_preserving:
            y_complement = y_complement - self.core(torch.zeros_like(x_complement))
        if self.complement_residual:
            y_complement = x_complement + y_complement
        y_complement = self.constraint.project_complement(y_complement)
        y = y_kernel + y_complement
        if not return_metrics:
            return y
        return DiagnosticOutput(
            y=y,
            metrics={
                "protected_violation": self.constraint.violation(y_kernel).mean().detach(),
                "complement_norm": torch.linalg.norm(y_complement, dim=-1).mean().detach(),
                "constraint_violation_out": self.constraint.violation(y).mean().detach(),
            },
        )


class OutputProjectionWrapper(nn.Module):
    """Wrap a model and project only its final output onto ``ker A``."""

    def __init__(self, model: nn.Module, constraint: LinearConstraint) -> None:
        super().__init__()
        self.model = model
        self.constraint = constraint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.constraint.project_kernel(self.model(x))
