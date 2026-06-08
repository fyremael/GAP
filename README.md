# Gap-Protected Transformers

This repository is a minimal executable scaffold for testing gap-protected neural operators in small PyTorch settings. A known linear constraint `A x = 0` defines the protected subspace `ker A`; models either ignore it, penalize violations, project outputs back to it, or split the state into protected and complement channels. The goal is to test whether Transformer-style learned operators can preserve known protected modes while learning controlled complement dynamics, and whether leakage and gap diagnostics reveal failures that ordinary losses miss.

## Installation

Use Python 3.11 or newer from the repository root.

```bash
python -m pip install torch numpy pytest
```

For editable development:

```bash
python -m pip install -e ".[dev]"
```

This installs console scripts:

- `gap-train`
- `gap-evaluate`
- `gap-experiments`

## Minimal Example

```python
import torch

from gap_protected_transformers import LinearConstraint, ProjectedResidualBlock, MLPCore

A = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float64)
constraint = LinearConstraint(A)
block = ProjectedResidualBlock(3, constraint, MLPCore(3, hidden_dim=16)).to(dtype=torch.float64)

x = torch.randn(8, 3, dtype=torch.float64)
y = block(x)
print(constraint.violation(y).max())
```

`ProjectedResidualBlock` applies `P_K(x + core(x))`, so the output is in `ker A` up to numerical precision. For sparse-compatible smoke tests, `ImplicitLinearConstraint` exposes the same projection surface but computes the correction through a conjugate-gradient solve of `A A^* lambda = A x` rather than forming `A^dagger`.

The package now also includes learned compatibility modules:

- `CompatibleRestriction`: learns hierarchy maps constrained by `A_c R = R_A A_f`.
- `CompatibleRouter`: learns expert dispatch matrices constrained to avoid protected/complement cross-mixing.
- `TokenTransformerCore`: a small pre-norm attention plus FFN core over vector chunks.

## Real Data Training

The production path trains on external arrays rather than generated sanity data. Store paired inputs and targets in a `.npz` file and a dense constraint matrix in `.npy` or `.npz`:

```text
data.npz
  x: float array with shape (N, ...)
  y: float array with shape (N, ...)

constraint.npy
  A: float array with shape (constraint_dim, flattened_state_dim)
```

Then run:

```bash
python -m gap_protected_transformers.train \
  --data data.npz \
  --constraint constraint.npy \
  --variant split_gap_protected \
  --core transformer \
  --depth 4 \
  --num-tokens 8 \
  --num-heads 1 \
  --epochs 50 \
  --batch-size 32 \
  --output-dir runs/my_real_run
```

Outputs:

- `best.pt` and `last.pt` checkpoints;
- `history.csv` with train/validation loss and diagnostics by epoch;
- `metrics.jsonl` with one JSON metric row per epoch;
- `summary.json` with final metrics and serialized model/training configs.

Use `--variant soft_penalty --penalty-weight <value>` for a loss-only baseline, `--variant output_projection` for final hard projection, `--variant layerwise_projection` for projected residual layers, and `--variant split_gap_protected` for protected/complement transport.

Evaluate a saved checkpoint:

```bash
python -m gap_protected_transformers.evaluate \
  --checkpoint runs/my_real_run/best.pt \
  --data data.npz \
  --constraint constraint.npy \
  --output runs/my_real_run/eval_metrics.json
```

If installed editable, the equivalent commands are:

```bash
gap-train --data data.npz --constraint constraint.npy --output-dir runs/my_real_run
gap-evaluate --checkpoint runs/my_real_run/best.pt --data data.npz --constraint constraint.npy
```

## What Is Protected

The protected component is `x_K = P_K x`, where `P_K = I - A^dagger A` is the dense orthogonal projector onto `ker A`. In the first two toys, this means:

- edge-flow graph toy: `A` is a graph incidence matrix, so protected flows are divergence-free edge flows;
- grid divergence toy: `A` is a periodic finite-difference divergence operator, so protected velocity fields have near-zero discrete divergence.

The learned complement is `x_perp = x - x_K`. The split block carries `x_K` through a protected transport path and applies a replaceable MLP, attention, or MoE core to `x_perp`. Complement-dynamics experiments target `x_K + S x_perp`, so exact output projection is expected to discard useful complement information while the split block can keep protected modes and learn a complement map.

## Why This Differs From A PINN

A PINN-style soft penalty adds a loss term such as `||A y||^2` and asks training to reduce violations. A hard projected model prevents selected violations by construction at the projected locations. The split gap-protected block additionally separates protected transport from complement dynamics, so leakage can be audited mode by mode rather than only measured at the final output.

## Ablation Variants

- `vanilla`: residual MLP blocks with no constraint handling.
- `soft_penalty`: same model as `vanilla`, trained with an added `||A y||^2` penalty.
- `output_projection`: vanilla model followed by one final projection onto `ker A`.
- `layerwise_projection`: every residual block applies `P_K(x + core(x))`.
- `split_gap_protected`: protected/complement decomposition at each block; denoising runs use a protected readout, while complement-dynamics runs use residual complement transport plus a learned complement correction.

## Diagnostics

- `constraint_violation`: per-sample `||A x||`.
- `protected_leakage`: probe of `||P_perp T P_K||_F`, measuring protected-to-complement leakage.
- `complement_leakage`: probe of `||P_K T P_perp||_F`, measuring complement-to-protected mixing.
- `commutator_error`: normalized `||A_c R - R_A A_f||_F` for hierarchy compatibility.
- `routing_commutator_error`: normalized `||R P_K - P_K^expert R||_F` for router compatibility.
- `gap_proxy`: smallest positive eigenvalue of `A^* A`, nullity estimate, and gap ratio.
- `jacobian_spectral_proxy`: optional finite-difference estimate of local amplification.

## First Experiment Commands

Run tests:

```bash
pytest -q
```

Run the edge-flow sanity experiment:

```bash
python examples/run_edge_flow_sanity.py
```

Run the grid divergence sanity experiment:

```bash
python examples/run_divergence_free_sanity.py
```

Run the complement-dynamics sanity experiment:

```bash
python examples/run_complement_dynamics_sanity.py
```

Run through the CLI entry point:

```bash
python -m gap_protected_transformers.experiments --experiment edge_flow_sanity
```

Use the tiny attention core instead of the MLP core:

```bash
python -m gap_protected_transformers.experiments --experiment edge_flow_complement --core attention --epochs 60
```

Run nontrivial hierarchy and routing diagnostics:

```bash
python -m gap_protected_transformers.experiments --experiment structure_diagnostics
```

Run learned compatible versus unconstrained hierarchy/routing diagnostics:

```bash
python examples/run_learned_structure_diagnostics.py
```

Metrics are written to `runs/<experiment>/metrics.csv` and `runs/<experiment>/metrics.json`.

## Observed Outputs

On this workspace with Python 3.12.10, PyTorch 2.6.0, and pytest 8.3.5:

```text
pytest -q
29 passed in 4.29s
```

Edge-flow sanity, default settings:

| variant | test_loss | constraint_violation_mean | constraint_violation_max | protected_leakage |
|---|---:|---:|---:|---:|
| vanilla | 0.00576793 | 0.269707 | 0.671901 | 0.767686 |
| soft_penalty | 0.0187155 | 0.347557 | 0.625801 | 0.929008 |
| output_projection | 0.000319048 | 1.38876e-15 | 3.31955e-15 | 1.27597e-15 |
| layerwise_projection | 0.00011671 | 8.81399e-16 | 2.31289e-15 | 3.62167e-16 |
| split_gap_protected | 2.09563e-31 | 8.24852e-16 | 2.42222e-15 | 8.61793e-17 |

Grid divergence sanity, default settings:

| variant | test_loss | constraint_violation_mean | constraint_violation_max | protected_leakage |
|---|---:|---:|---:|---:|
| vanilla | 0.0176068 | 1.40608 | 2.16595 | 4.49731 |
| soft_penalty | 0.0434412 | 1.36994 | 2.01107 | 3.92038 |
| output_projection | 0.00150667 | 2.29843e-15 | 3.26807e-15 | 3.2703e-15 |
| layerwise_projection | 0.00164948 | 1.79296e-15 | 2.82923e-15 | 2.38187e-15 |
| split_gap_protected | 2.79773e-31 | 1.58416e-15 | 2.58366e-15 | 6.36009e-16 |

Edge-flow complement dynamics, MLP core, 80 epochs:

| variant | test_loss | protected_component_loss | complement_component_loss | protected_leakage |
|---|---:|---:|---:|---:|
| vanilla | 0.00685534 | 0.000355029 | 0.00650031 | 0.89465 |
| soft_penalty | 0.0698862 | 0.00608282 | 0.0638033 | 0.927296 |
| output_projection | 0.06476 | 0.000110541 | 0.0646495 | 1.30445e-15 |
| layerwise_projection | 0.06472 | 7.05261e-05 | 0.0646495 | 3.61363e-16 |
| split_gap_protected | 0.00367197 | 5.35071e-33 | 0.00367197 | 5.64488e-12 |

Structure diagnostics:

| diagnostic | value |
|---|---:|
| compatible cycle hierarchy commutator | 0 |
| incompatible cycle hierarchy commutator | 0.414214 |
| componentwise router commutator | 0 |
| random router commutator | 0.609369 |

Learned structure diagnostics:

| variant | fit_loss | commutator |
|---|---:|---:|
| hard compatible restriction | 0.046875 | 5.20417e-17 |
| soft-penalty restriction | 0.0720119 | 0.0772431 |
| unconstrained restriction | 1.28466e-06 | 0.413441 |
| hard compatible router | 0.0374654 | 1.34361e-15 |
| soft-penalty router | 0.113465 | 0.00241911 |
| unconstrained router | 0.000513643 | 0.618472 |

Complement-dynamics rollout diagnostics, MLP core, 80 epochs:

| variant | rollout_violation_final | protected_relative_drift | complement_sigma_proxy |
|---|---:|---:|---:|
| vanilla | 1.28216 | 0.275046 | 1.0951 |
| soft_penalty | 0.594558 | 0.689906 | 0.909542 |
| output_projection | 1.26569e-15 | 0.244048 | 2.53056e-12 |
| layerwise_projection | 9.75222e-16 | 0.0498794 | 2.73317e-12 |
| split_gap_protected | 1.1876 | 8.95027e-16 | 1.05821 |

These are sanity checks, not evidence of broad model superiority. The useful early signal is that soft penalties can leave measurable leakage in short toy runs, exact projection can enforce protected outputs while discarding complement targets, and split protected transport can preserve the protected component while learning a small complement map. The learned structure diagnostic shows the expected tradeoff: unconstrained maps can fit an incompatible target closely while violating compatibility; compatible parameterizations keep commutators near numerical zero; soft penalties occupy the middle but do not provide exact guarantees. The rollout table adds a second distinction: split protection can allow complement constraint residuals while keeping protected-component drift essentially zero.

## Claim Boundary

This repository does not claim to invent Hodge decompositions, divergence-free neural networks, hard-constrained neural operators, hard-constrained PINNs, neural multigrid, or Transformer preconditioners. The contribution being scaffolded is the Transformer/MoE-native synthesis: protected-mode transport, learned complement dynamics, compatible hierarchy and routing maps, and diagnostics for leakage and gap preservation.

## Current Limitations

- `ImplicitLinearConstraint` is a CG scaffold, not a tuned sparse Hodge or Poisson solver.
- The attention and MoE cores are intentionally tiny placeholders.
- `TokenTransformerCore` is a more realistic scaffold, but still far from a production architecture.
- The complement-dynamics task is synthetic linear dynamics, not a PDE rollout benchmark.
- Learned hierarchy and routing diagnostics are matrix-level toys; learned compatible transfer in a real multiresolution model remains future work.
