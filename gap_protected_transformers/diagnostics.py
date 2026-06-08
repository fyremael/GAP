"""Leakage, commutator, and spectral diagnostics for protected modes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from .blocks import DiagnosticOutput
from .operators import LinearConstraint


def constraint_violation(
    constraint: LinearConstraint,
    x: torch.Tensor,
    *,
    relative: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return per-sample constraint residual ``||A x||``."""

    return constraint.violation(x, relative=relative, eps=eps)


def constraint_violation_stats(
    constraint: LinearConstraint, x: torch.Tensor, *, relative: bool = False
) -> dict[str, torch.Tensor]:
    """Return mean, max, 95th percentile, and final residual statistics."""

    values = constraint_violation(constraint, x, relative=relative).reshape(-1)
    return {
        "constraint_violation_mean": values.mean(),
        "constraint_violation_max": values.max(),
        "constraint_violation_p95": torch.quantile(values, 0.95),
        "constraint_violation_final": values[-1],
    }


def protected_energy(constraint: LinearConstraint, x: torch.Tensor) -> torch.Tensor:
    """Return mean squared energy in the protected kernel component."""

    return torch.mean(constraint.project_kernel(x) ** 2)


def complement_energy(constraint: LinearConstraint, x: torch.Tensor) -> torch.Tensor:
    """Return mean squared energy in the complement component."""

    return torch.mean(constraint.project_complement(x) ** 2)


def commutator_error(
    A_c: torch.Tensor,
    R: torch.Tensor,
    R_A: torch.Tensor,
    A_f: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return normalized ``||A_c R - R_A A_f||_F``."""

    left = A_c @ R
    right = R_A @ A_f
    denom = torch.linalg.norm(left) + torch.linalg.norm(right) + eps
    return torch.linalg.norm(left - right) / denom


def routing_commutator_error(
    router: torch.Tensor,
    constraint: LinearConstraint,
    *,
    num_experts: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return normalized ``||R P_K - P_K^expert R||_F`` for a linear router."""

    expected_shape = (num_experts * constraint.dim, constraint.dim)
    if router.shape != expected_shape:
        raise ValueError(f"Expected router shape {expected_shape}, got {tuple(router.shape)}.")
    P_k = constraint.kernel_projector().to(device=router.device, dtype=router.dtype)
    expert_projector = torch.block_diag(*[P_k for _ in range(num_experts)])
    left = router @ P_k
    right = expert_projector @ router
    denom = torch.linalg.norm(left) + torch.linalg.norm(right) + eps
    return torch.linalg.norm(left - right) / denom


def gap_proxy(
    constraint: LinearConstraint, *, rtol: float = 1e-8
) -> dict[str, torch.Tensor | int]:
    """Estimate the normal-operator gap of ``A^* A`` on the complement."""

    eigvals = torch.linalg.eigvalsh(constraint.normal_matrix())
    eigvals = torch.clamp(eigvals.real, min=0)
    lambda_max = eigvals.max()
    tol = torch.maximum(
        torch.as_tensor(rtol, dtype=eigvals.dtype, device=eigvals.device) * lambda_max,
        torch.as_tensor(torch.finfo(eigvals.dtype).eps, dtype=eigvals.dtype, device=eigvals.device),
    )
    positive = eigvals[eigvals > tol]
    if positive.numel() == 0:
        lambda_min_plus = torch.zeros((), dtype=eigvals.dtype, device=eigvals.device)
    else:
        lambda_min_plus = positive.min()
    gap_ratio = lambda_min_plus / (lambda_max + torch.finfo(eigvals.dtype).eps)
    return {
        "lambda_min_plus": lambda_min_plus,
        "lambda_max": lambda_max,
        "gap_ratio": gap_ratio,
        "nullity": int((eigvals <= tol).sum().item()),
    }


def protected_leakage(
    transform: torch.Tensor | nn.Module | Callable[[torch.Tensor], Any],
    constraint: LinearConstraint,
) -> torch.Tensor:
    """Probe ``||P_perp T P_K||_F`` for a linear map or callable transform."""

    if isinstance(transform, torch.Tensor):
        T = _operator_matrix(transform, constraint)
        P_k = constraint.kernel_projector().to(device=T.device, dtype=T.dtype)
        P_perp = constraint.complement_projector().to(device=T.device, dtype=T.dtype)
        return torch.linalg.norm(P_perp @ T @ P_k)
    response = _projected_response_matrix(transform, constraint, "kernel")
    P_perp = constraint.complement_projector().to(
        device=response.device, dtype=response.dtype
    )
    return torch.linalg.norm(P_perp @ response)


def complement_leakage(
    transform: torch.Tensor | nn.Module | Callable[[torch.Tensor], Any],
    constraint: LinearConstraint,
) -> torch.Tensor:
    """Probe ``||P_K T P_perp||_F`` for a linear map or callable transform."""

    if isinstance(transform, torch.Tensor):
        T = _operator_matrix(transform, constraint)
        P_k = constraint.kernel_projector().to(device=T.device, dtype=T.dtype)
        P_perp = constraint.complement_projector().to(device=T.device, dtype=T.dtype)
        return torch.linalg.norm(P_k @ T @ P_perp)
    response = _projected_response_matrix(transform, constraint, "complement")
    P_k = constraint.kernel_projector().to(device=response.device, dtype=response.dtype)
    return torch.linalg.norm(P_k @ response)


def jacobian_spectral_proxy(
    transform: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    steps: int = 8,
    eps: float = 1e-4,
    seed: int = 0,
) -> torch.Tensor:
    """Finite-difference proxy for the local spectral norm of a square map."""

    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    v = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    v = v / (torch.linalg.norm(v) + 1e-12)
    base = transform(x)
    sigma = torch.zeros((), device=x.device, dtype=x.dtype)
    for _ in range(steps):
        jv = (transform(x + eps * v) - base) / eps
        sigma = torch.linalg.norm(jv)
        v = jv / (sigma + 1e-12)
    return sigma


def complement_jacobian_spectral_proxy(
    transform: Callable[[torch.Tensor], torch.Tensor],
    constraint: LinearConstraint,
    x: torch.Tensor,
    *,
    steps: int = 8,
    eps: float = 1e-4,
    seed: int = 0,
) -> torch.Tensor:
    """Finite-difference spectral proxy for ``P_perp J P_perp``."""

    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    v = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    v = constraint.project_complement(v)
    v = v / (torch.linalg.norm(v) + 1e-12)
    base = _as_tensor_output(transform(x))
    sigma = torch.zeros((), device=x.device, dtype=x.dtype)
    for _ in range(steps):
        jv = (_as_tensor_output(transform(x + eps * v)) - base) / eps
        jv = constraint.project_complement(jv)
        sigma = torch.linalg.norm(jv)
        v = constraint.project_complement(jv) / (sigma + 1e-12)
    return sigma


def rollout_diagnostics(
    transform: Callable[[torch.Tensor], torch.Tensor],
    constraint: LinearConstraint,
    x0: torch.Tensor,
    *,
    steps: int = 8,
) -> dict[str, torch.Tensor]:
    """Roll a map forward and report constraint drift through time."""

    with torch.no_grad():
        x = x0
        violations = []
        protected_energies = []
        complement_energies = []
        initial_protected = constraint.project_kernel(x0)
        for _ in range(steps):
            x = _as_tensor_output(transform(x))
            violations.append(constraint.violation(x).mean())
            protected_energies.append(protected_energy(constraint, x))
            complement_energies.append(complement_energy(constraint, x))
        violation_series = torch.stack(violations)
        protected_series = torch.stack(protected_energies)
        complement_series = torch.stack(complement_energies)
        protected_drift = torch.linalg.norm(
            constraint.project_kernel(x) - initial_protected
        ) / (torch.linalg.norm(initial_protected) + 1e-12)
    return {
        "rollout_violation_mean": violation_series.mean(),
        "rollout_violation_max": violation_series.max(),
        "rollout_violation_final": violation_series[-1],
        "rollout_protected_energy_initial": protected_series[0],
        "rollout_protected_energy_final": protected_series[-1],
        "rollout_complement_energy_final": complement_series[-1],
        "rollout_protected_relative_drift": protected_drift,
    }


def _operator_matrix(
    transform: torch.Tensor | nn.Module | Callable[[torch.Tensor], Any],
    constraint: LinearConstraint,
) -> torch.Tensor:
    if isinstance(transform, torch.Tensor):
        if transform.shape != (constraint.dim, constraint.dim):
            raise ValueError(
                f"Expected transform matrix {(constraint.dim, constraint.dim)}, "
                f"got {tuple(transform.shape)}."
            )
        return transform

    dtype = constraint.A.dtype
    device = constraint.A.device
    if isinstance(transform, nn.Module):
        params = list(transform.parameters())
        if params:
            dtype = params[0].dtype
            device = params[0].device

    basis = torch.eye(constraint.dim, dtype=dtype, device=device)
    with torch.no_grad():
        output = transform(constraint.project_kernel(basis) + constraint.project_complement(basis))
    output = _as_tensor_output(output)
    output = output.reshape(constraint.dim, constraint.dim)
    return output.T


def _projected_response_matrix(
    transform: nn.Module | Callable[[torch.Tensor], Any],
    constraint: LinearConstraint,
    component: str,
) -> torch.Tensor:
    dtype = constraint.A.dtype
    device = constraint.A.device
    if isinstance(transform, nn.Module):
        params = list(transform.parameters())
        if params:
            dtype = params[0].dtype
            device = params[0].device

    basis = torch.eye(constraint.dim, dtype=dtype, device=device)
    if component == "kernel":
        inputs = constraint.project_kernel(basis)
    elif component == "complement":
        inputs = constraint.project_complement(basis)
    else:
        raise ValueError("component must be 'kernel' or 'complement'.")

    with torch.no_grad():
        output = transform(inputs)
    output = _as_tensor_output(output)
    return output.reshape(constraint.dim, constraint.dim).T


def _as_tensor_output(output: Any) -> torch.Tensor:
    if isinstance(output, DiagnosticOutput):
        output = output.y
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError("Transform must return a tensor or DiagnosticOutput.")
    return output
