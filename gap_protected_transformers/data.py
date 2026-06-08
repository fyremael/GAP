"""Dataset loading utilities for supervised vector-operator training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split

from .operators import LinearConstraint


@dataclass
class ArrayMetadata:
    """Metadata describing a loaded array-pair dataset."""

    num_samples: int
    input_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    flat_dim: int
    input_key: str
    target_key: str


class ArrayPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Supervised dataset backed by input and target arrays."""

    def __init__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        input_key: str = "x",
        target_key: str = "y",
        flatten: bool = True,
    ) -> None:
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("inputs and targets must have the same sample count.")
        self.original_input_shape = tuple(inputs.shape[1:])
        self.original_target_shape = tuple(targets.shape[1:])
        if flatten:
            inputs = inputs.reshape(inputs.shape[0], -1)
            targets = targets.reshape(targets.shape[0], -1)
        if inputs.shape[-1] != targets.shape[-1]:
            raise ValueError("flattened inputs and targets must have the same dim.")
        self.inputs = inputs.contiguous()
        self.targets = targets.contiguous()
        self.metadata = ArrayMetadata(
            num_samples=int(inputs.shape[0]),
            input_shape=self.original_input_shape,
            target_shape=self.original_target_shape,
            flat_dim=int(inputs.shape[-1]),
            input_key=input_key,
            target_key=target_key,
        )

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


def load_array_pair_dataset(
    path: str | Path,
    *,
    input_key: str = "x",
    target_key: str = "y",
    dtype: torch.dtype = torch.float32,
    flatten: bool = True,
) -> ArrayPairDataset:
    """Load an ``ArrayPairDataset`` from a ``.npz`` file."""

    source = Path(path)
    if source.suffix.lower() != ".npz":
        raise ValueError("array-pair datasets must be stored as .npz files.")
    with np.load(source) as data:
        if input_key not in data:
            raise KeyError(f"Missing input_key {input_key!r} in {source}.")
        if target_key not in data:
            raise KeyError(f"Missing target_key {target_key!r} in {source}.")
        inputs = torch.as_tensor(data[input_key], dtype=dtype)
        targets = torch.as_tensor(data[target_key], dtype=dtype)
    return ArrayPairDataset(
        inputs,
        targets,
        input_key=input_key,
        target_key=target_key,
        flatten=flatten,
    )


def split_dataset(
    dataset: ArrayPairDataset,
    *,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[Subset, Subset]:
    """Split a dataset into train and validation subsets."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    val_size = max(1, int(round(len(dataset) * val_fraction)))
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise ValueError("dataset is too small for the requested split.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    train, val = random_split(dataset, [train_size, val_size], generator=generator)
    return train, val


def load_constraint(
    path: str | Path,
    *,
    key: str = "A",
    dtype: torch.dtype = torch.float32,
) -> LinearConstraint:
    """Load a dense linear constraint matrix from ``.npy`` or ``.npz``."""

    source = Path(path)
    if source.suffix.lower() == ".npy":
        array = np.load(source)
    elif source.suffix.lower() == ".npz":
        with np.load(source) as data:
            if key not in data:
                raise KeyError(f"Missing constraint key {key!r} in {source}.")
            array = data[key]
    else:
        raise ValueError("constraint must be stored as .npy or .npz.")
    return LinearConstraint(torch.as_tensor(array, dtype=dtype))


def infer_flat_dim(dataset: Dataset[tuple[torch.Tensor, torch.Tensor]]) -> int:
    """Return the flattened feature dimension from the first dataset sample."""

    x, _ = dataset[0]
    return int(x.numel())


def subset_indices(subset: Subset) -> Sequence[int]:
    """Return indices from a PyTorch ``Subset`` for reproducibility metadata."""

    return list(subset.indices)
