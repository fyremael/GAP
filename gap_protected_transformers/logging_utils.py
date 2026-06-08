"""Small CSV/JSON logging helpers for experiment outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch


def to_serializable(value: Any) -> Any:
    """Convert tensors and paths into JSON/CSV friendly values."""

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    return value


def write_json(path: str | Path, data: Any) -> None:
    """Write JSON with stable indentation."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(to_serializable(data), indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of flat dictionaries to CSV."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_serializable(row.get(key, "")) for key in fieldnames})


class MetricLogger:
    """Small local metric logger with an optional W&B mirror."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self._handle = None
        self._wandb_run = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("w", encoding="utf-8")
        if use_wandb:
            try:
                import wandb  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "W&B logging requested, but wandb is not installed."
                ) from exc
            self._wandb_run = wandb.init(project=wandb_project, config=config or {})

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        """Log one metric row locally and optionally to W&B."""

        row = {"step": step, **metrics} if step is not None else dict(metrics)
        row = to_serializable(row)
        if self._handle is not None:
            self._handle.write(json.dumps(row, sort_keys=True) + "\n")
            self._handle.flush()
        if self._wandb_run is not None:
            self._wandb_run.log(row, step=step)

    def close(self) -> None:
        """Flush local output and finish the optional W&B run."""

        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    def __enter__(self) -> "MetricLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
