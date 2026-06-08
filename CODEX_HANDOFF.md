# Codex Handoff: Gap-Protected Transformer Prototype

## Repository Goal

Implement a minimal but extensible research harness for gap-protected neural operators and Transformers.

The code must support:

1. sparse linear constraint operators `A`;
2. exact or approximate projectors onto `ker A`;
3. layerwise projected Transformer blocks;
4. split-channel protected/complement blocks;
5. commutator diagnostics for hierarchy and routing;
6. ablations against vanilla and soft-constraint baselines;
7. W&B-ready logging keys.

## Proposed File Structure

```text
src/
  operators/
    constraint_operator.py
    hodge_grid.py
    graph_complex.py
    projectors.py
  models/
    vanilla_transformer.py
    projected_block.py
    split_hodge_transformer.py
    compatible_hierarchy.py
    moe_routing.py
  diagnostics/
    leakage.py
    commutators.py
    spectral.py
    rollout.py
  data/
    synthetic_fields.py
    graph_flows.py
  training/
    losses.py
    trainer.py
    configs.py
  experiments/
    run_div_free.py
    run_graph_flow.py
    run_moe_diagnostics.py
tests/
  test_projector_exactness.py
  test_commutator_metrics.py
  test_hodge_decomposition.py
  test_projected_block.py
```

## Core Interfaces

### ConstraintOperator

```python
class ConstraintOperator:
    def apply(self, x):
        """Return A x."""

    def adjoint(self, y):
        """Return A^* y."""

    def normal(self, x):
        """Return A^* A x."""
```

### Projector

```python
class KernelProjector:
    def project(self, x):
        """Return projection onto ker A."""

    def complement(self, x):
        """Return x - project(x)."""
```

### ProjectedResidualBlock

```python
class ProjectedResidualBlock(nn.Module):
    def forward(self, x):
        y = x + self.learned_block(x)
        return self.projector.project(y)
```

### SplitProtectedBlock

```python
class SplitProtectedBlock(nn.Module):
    def forward(self, x):
        x_k = self.projector.project(x)
        x_p = self.projector.complement(x)
        y_k = self.protected_transport(x_k)
        y_p = self.complement_block(x_p)
        return y_k + y_p
```

### Diagnostics

```python
def leakage(A, x):
    return norm(A.apply(x)) / (norm(x) + eps)

def commutator_error(Ac, R, RA, Af):
    return norm(Ac @ R - RA @ Af) / denom
```

## Logging Keys

```text
train/loss_pred
train/loss_leak
train/loss_comm
train/leakage_mean
train/leakage_max
train/protected_energy
train/complement_energy
train/commutator_restrict
train/commutator_route
train/gap_lambda_min_plus
train/complement_sigma
val/mse
val/leakage
val/rollout_drift
val/resolution_transfer_error
```

## Implementation Notes

- Start with dense projectors for small tests.
- Add sparse/implicit projectors after exact tests pass.
- Keep the operator `A` explicit and auditable.
- Every model forward pass should optionally return a diagnostic dictionary.
- Avoid hiding projection inside opaque layers; the entire point is inspectability.
- Use small synthetic datasets first, but design interfaces for real PDE/graph data.
