"""Run learned compatible-vs-unconstrained hierarchy and routing diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gap_protected_transformers.experiments import print_results, run_experiment


if __name__ == "__main__":
    rows = run_experiment(
        "learned_structure_diagnostics",
        epochs=120,
        lr=0.01,
        output_dir=ROOT / "runs",
    )
    print_results(rows)
