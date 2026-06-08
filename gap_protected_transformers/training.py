"""Supervised training loop for protected vector-operator models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .diagnostics import (
    complement_energy,
    constraint_violation_stats,
    protected_energy,
    rollout_diagnostics,
)
from .logging_utils import MetricLogger, write_csv, write_json
from .models import ModelConfig, build_model
from .operators import LinearConstraint


@dataclass
class TrainingConfig:
    """Training config for supervised vector-operator fitting."""

    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    penalty_weight: float = 0.0
    grad_clip_norm: float | None = 1.0
    device: str = "cpu"
    output_dir: str = "runs/supervised"
    seed: int = 0
    rollout_steps: int = 4


def train_supervised(
    model: nn.Module,
    constraint: LinearConstraint,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    config: TrainingConfig,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a vector operator and write history/checkpoints to disk."""

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    best_val = float("inf")
    best_epoch = -1

    with MetricLogger(output_dir / "metrics.jsonl") as logger:
        for epoch in range(1, config.epochs + 1):
            train_metrics = _train_epoch(model, constraint, train_loader, optimizer, config)
            val_metrics = evaluate_supervised(
                model,
                constraint,
                val_loader,
                device=device,
                rollout_steps=config.rollout_steps,
            )
            row = {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            history.append(row)
            logger.log(row, step=epoch)
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_epoch = epoch
                _save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    config,
                    model_config,
                    epoch,
                    val_metrics,
                )

    final_metrics = evaluate_supervised(
        model,
        constraint,
        val_loader,
        device=device,
        rollout_steps=config.rollout_steps,
    )
    _save_checkpoint(
        output_dir / "last.pt",
        model,
        config,
        model_config,
        config.epochs,
        final_metrics,
    )
    write_csv(output_dir / "history.csv", history)
    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "final_metrics": final_metrics,
        "training_config": asdict(config),
        "model_config": model_config or {},
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def evaluate_supervised(
    model: nn.Module,
    constraint: LinearConstraint,
    loader: DataLoader,
    *,
    device: torch.device,
    rollout_steps: int = 4,
) -> dict[str, float]:
    """Evaluate prediction loss and protected-mode diagnostics."""

    model.eval()
    losses = []
    element_count = 0
    predictions = []
    targets = []
    inputs = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            losses.append(F.mse_loss(pred, y, reduction="sum"))
            element_count += y.numel()
            predictions.append(pred.detach().cpu())
            targets.append(y.detach().cpu())
            inputs.append(x.detach().cpu())

    pred_all = torch.cat(predictions, dim=0)
    target_all = torch.cat(targets, dim=0)
    input_all = torch.cat(inputs, dim=0)
    loss = torch.stack(losses).sum() / element_count
    stats = constraint_violation_stats(constraint, pred_all)
    protected_loss = F.mse_loss(
        constraint.project_kernel(pred_all), constraint.project_kernel(target_all)
    )
    complement_loss = F.mse_loss(
        constraint.project_complement(pred_all),
        constraint.project_complement(target_all),
    )
    rollout = rollout_diagnostics(
        lambda z: model(z.to(device)).detach().cpu(),
        constraint,
        input_all[: min(16, input_all.shape[0])],
        steps=rollout_steps,
    )
    return {
        "loss": float(loss.cpu()),
        "protected_component_loss": float(protected_loss.cpu()),
        "complement_component_loss": float(complement_loss.cpu()),
        "constraint_violation_mean": float(
            stats["constraint_violation_mean"].cpu()
        ),
        "constraint_violation_max": float(stats["constraint_violation_max"].cpu()),
        "protected_energy": float(protected_energy(constraint, pred_all).cpu()),
        "complement_energy": float(complement_energy(constraint, pred_all).cpu()),
        "rollout_violation_final": float(rollout["rollout_violation_final"].cpu()),
        "rollout_protected_relative_drift": float(
            rollout["rollout_protected_relative_drift"].cpu()
        ),
    }


def _train_epoch(
    model: nn.Module,
    constraint: LinearConstraint,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> dict[str, float]:
    model.train()
    device = torch.device(config.device)
    loss_total = 0.0
    pred_total = 0.0
    penalty_total = 0.0
    sample_count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        pred_loss = F.mse_loss(pred, y)
        penalty = torch.mean(constraint.apply(pred) ** 2)
        loss = pred_loss + config.penalty_weight * penalty
        loss.backward()
        if config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        batch_size = x.shape[0]
        loss_total += float(loss.detach().cpu()) * batch_size
        pred_total += float(pred_loss.detach().cpu()) * batch_size
        penalty_total += float(penalty.detach().cpu()) * batch_size
        sample_count += batch_size
    return {
        "loss": loss_total / sample_count,
        "prediction_loss": pred_total / sample_count,
        "constraint_penalty": penalty_total / sample_count,
    }


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainingConfig,
    model_config: dict[str, Any] | None,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "training_config": asdict(config),
            "model_config": model_config or {},
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_trained_model(
    checkpoint_path: str | Path,
    constraint: LinearConstraint,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a checkpoint and rebuild its model from serialized config."""

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    model_config = ModelConfig.from_dict(checkpoint["model_config"])
    model = build_model(model_config, constraint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
