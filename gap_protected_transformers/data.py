"""Dataset loading utilities for supervised vector-operator training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split

from .operators import ImplicitLinearConstraint, LinearConstraint


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


class HDF5PairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy supervised dataset backed by HDF5 input and target arrays."""

    def __init__(
        self,
        path: str | Path,
        *,
        input_key: str = "x",
        target_key: str = "y",
        dtype: torch.dtype = torch.float32,
        flatten: bool = True,
    ) -> None:
        try:
            import h5py  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "HDF5 datasets require h5py. Install with `pip install h5py` "
                "or `pip install -e .[benchmark]`."
            ) from exc
        self._h5py = h5py
        self.path = Path(path)
        self.input_key = input_key
        self.target_key = target_key
        self.dtype = dtype
        self.flatten = flatten
        with h5py.File(self.path, "r") as handle:
            if input_key not in handle:
                raise KeyError(f"Missing input_key {input_key!r} in {self.path}.")
            if target_key not in handle:
                raise KeyError(f"Missing target_key {target_key!r} in {self.path}.")
            input_shape = tuple(handle[input_key].shape)
            target_shape = tuple(handle[target_key].shape)
        if input_shape[0] != target_shape[0]:
            raise ValueError("inputs and targets must have the same sample count.")
        flat_dim = int(np.prod(input_shape[1:]))
        if flat_dim != int(np.prod(target_shape[1:])):
            raise ValueError("flattened inputs and targets must have the same dim.")
        self.metadata = ArrayMetadata(
            num_samples=int(input_shape[0]),
            input_shape=tuple(input_shape[1:]),
            target_shape=tuple(target_shape[1:]),
            flat_dim=flat_dim,
            input_key=input_key,
            target_key=target_key,
        )

    def __len__(self) -> int:
        return self.metadata.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with self._h5py.File(self.path, "r") as handle:
            x = torch.as_tensor(handle[self.input_key][index], dtype=self.dtype)
            y = torch.as_tensor(handle[self.target_key][index], dtype=self.dtype)
        if self.flatten:
            x = x.reshape(-1)
            y = y.reshape(-1)
        return x, y


def load_array_pair_dataset(
    path: str | Path,
    *,
    input_key: str = "x",
    target_key: str = "y",
    dtype: torch.dtype = torch.float32,
    flatten: bool = True,
) -> Dataset[tuple[torch.Tensor, torch.Tensor]]:
    """Load a supervised pair dataset from ``.npz`` or optional HDF5."""

    source = Path(path)
    if source.suffix.lower() in {".h5", ".hdf5"}:
        return HDF5PairDataset(
            source,
            input_key=input_key,
            target_key=target_key,
            dtype=dtype,
            flatten=flatten,
        )
    if source.suffix.lower() != ".npz":
        raise ValueError("array-pair datasets must be stored as .npz, .h5, or .hdf5 files.")
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
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
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
    implicit: bool = False,
) -> LinearConstraint | ImplicitLinearConstraint:
    """Load a linear constraint matrix from dense or sparse ``.npy/.npz`` data.

    Sparse ``.npz`` constraints may use COO triplet keys:
    ``row``, ``col``, ``value``, and ``shape``. Sparse constraints are returned
    as ``ImplicitLinearConstraint`` because dense pseudoinverse projection is
    not appropriate at benchmark scale.
    """

    source = Path(path)
    is_sparse = False
    if source.suffix.lower() == ".npy":
        array = np.load(source)
        tensor = torch.as_tensor(array, dtype=dtype)
    elif source.suffix.lower() == ".npz":
        with np.load(source) as data:
            if {"row", "col", "value", "shape"}.issubset(data.files):
                rows = torch.as_tensor(data["row"], dtype=torch.long)
                cols = torch.as_tensor(data["col"], dtype=torch.long)
                values = torch.as_tensor(data["value"], dtype=dtype)
                shape = tuple(int(item) for item in data["shape"])
                tensor = torch.sparse_coo_tensor(
                    torch.stack([rows, cols]),
                    values,
                    size=shape,
                    dtype=dtype,
                ).coalesce()
                is_sparse = True
            elif key in data:
                tensor = torch.as_tensor(data[key], dtype=dtype)
            else:
                raise KeyError(f"Missing constraint key {key!r} in {source}.")
    else:
        raise ValueError("constraint must be stored as .npy or .npz.")
    if implicit or is_sparse:
        return ImplicitLinearConstraint(tensor)
    return LinearConstraint(tensor)


def infer_flat_dim(dataset: Dataset[tuple[torch.Tensor, torch.Tensor]]) -> int:
    """Return the flattened feature dimension from the first dataset sample."""

    x, _ = dataset[0]
    return int(x.numel())


def subset_indices(subset: Subset) -> Sequence[int]:
    """Return indices from a PyTorch ``Subset`` for reproducibility metadata."""

    return list(subset.indices)
