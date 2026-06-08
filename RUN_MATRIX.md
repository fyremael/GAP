# Run Matrix

## Phase 0: Sanity Checks

| Run | Dataset | Model | Purpose |
|---|---|---|---|
| P0-01 | synthetic vector fields on 2D grid | projector only | verify `Ax=0` after projection |
| P0-02 | random graph edge flows | Hodge decomposition | verify exact/approx orthogonality |
| P0-03 | toy MoE routing streams | commutator logging | verify routing leakage metrics |

## Phase 1: Divergence-Free Field Prediction

Task: predict next-step or multi-step velocity field with incompressibility constraint.

Models:

- vanilla Transformer;
- PINN-penalty Transformer;
- output-projected Transformer;
- layerwise-projected Transformer;
- split-channel Hodge Transformer.

Metrics:

- MSE;
- divergence leakage;
- rollout drift;
- spectral complement amplification;
- runtime cost.

## Phase 2: Graph Edge-Flow Prediction

Task: predict edge flows on graphs or simplicial complexes.

Protected quantities:

- incidence constraints;
- harmonic cycle components;
- circulation structure.

Models:

- GAT / Graph Transformer baseline;
- HodgeNet-style baseline;
- spectral graph transformer baseline;
- gap-protected graph transformer.

Metrics:

- edge prediction error;
- Hodge component leakage;
- harmonic drift;
- component-wise attribution;
- transfer to perturbed graph topology.

## Phase 3: Compatible Hierarchy

Task: train at one resolution and test at another.

Models:

- ordinary pooling;
- fixed compatible transfer;
- learned transfer with commutator loss;
- learned transfer with exact parameterized compatibility.

Metrics:

- resolution transfer error;
- commutator error;
- leakage after restriction/prolongation;
- gap stability across levels.

## Phase 4: MoE Routing Diagnostics

Task: synthetic sequence or graph-flow task with known protected modes and sparse expert routing.

Models:

- dense Transformer;
- standard sparse MoE;
- component-aware MoE;
- commutator-regularized MoE.

Metrics:

- routing entropy;
- expert load;
- protected-mode leakage by expert;
- route/projection commutator;
- mode-specific specialization;
- stability under expert dropout.

## Phase 5: Neural Preconditioner / Corrective Solver

Task: use the gap-protected block as a learned correction step inside an iterative solve.

Systems:

- Poisson;
- Stokes;
- graph Laplacian;
- Helmholtz-Hodge correction.

Metrics:

- iterations to tolerance;
- residual reduction per step;
- constraint violation;
- robustness under mesh change;
- comparison with AMG / Jacobi / learned baseline.
