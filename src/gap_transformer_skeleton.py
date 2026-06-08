"""Compatibility shim for the original prototype skeleton.

The scaffold now lives in the importable ``gap_protected_transformers`` package.
This file is kept so older references to ``src/gap_transformer_skeleton.py`` still
produce a quick diagnostic demo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gap_protected_transformers import (
    GapProtectedBlock,
    LinearConstraint,
    MLPCore,
    ProjectedResidualBlock,
    commutator_error,
)


DenseConstraintOperator = LinearConstraint
DenseKernelProjector = LinearConstraint
SplitProtectedBlock = GapProtectedBlock
TinyResidualMLP = MLPCore


def demo() -> dict[str, float]:
    """Run a tiny projected-residual diagnostic demo."""

    torch.manual_seed(0)
    dim = 8
    codim = 3
    A = torch.randn(codim, dim, dtype=torch.float64)
    constraint = LinearConstraint(A)
    block = ProjectedResidualBlock(dim, constraint, MLPCore(dim, 32)).to(
        dtype=torch.float64
    )
    x = torch.randn(4, dim, dtype=torch.float64)
    y = block(x)
    return {
        "constraint_violation_in": float(constraint.violation(x).mean()),
        "constraint_violation_out": float(constraint.violation(y).mean()),
        "identity_commutator_error": float(
            commutator_error(
                A,
                torch.eye(dim, dtype=torch.float64),
                torch.eye(codim, dtype=torch.float64),
                A,
            )
        ),
    }


if __name__ == "__main__":
    print(demo())
