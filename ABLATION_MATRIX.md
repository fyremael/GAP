# Ablation Matrix

The ablation study must separate four effects:

1. accuracy improvement;
2. exact constraint preservation;
3. hierarchy compatibility;
4. spectral/complement conditioning.

## Core Models

| ID | Model | Constraint Handling | Hierarchy | Expected Use |
|---|---|---|---|---|
| A0 | Vanilla Transformer | None | ordinary pooling | baseline |
| A1 | Soft-PINN Transformer | penalty loss `||Ax||^2` | ordinary pooling | tests loss-only physics |
| A2 | Output Projection | project only at final output | ordinary pooling | tests hard final correction |
| A3 | Layerwise Projection | project after every block | ordinary pooling | tests depth protection |
| A4 | Split Hodge Transformer | protected + complement channels | ordinary pooling | tests mode separation |
| A5 | Compatible Hodge Transformer | split channels | commuting restriction/prolongation | full hypothesis |
| A6 | Compatible Hodge MoE | split channels | routing compatibility | MoE diagnostic hypothesis |

## Loss Terms

Base prediction loss:

`L_pred = MSE(y_hat, y)`

Soft physics loss:

`L_phys = ||A y_hat||^2`

Commutator loss:

`L_comm = ||A_c R_theta - R_A A_f||^2`

Leakage loss for approximate protection:

`L_leak = sum_l ||A x_l||^2`

Complement spectral penalty:

`L_spec = max(0, sigma_perp - sigma_target)^2`

Total optional loss:

`L = L_pred + alpha L_phys + beta L_comm + gamma L_leak + eta L_spec`.

## Required Ablations

### Projection Frequency

- no projection;
- output-only projection;
- every N layers;
- every layer;
- split-channel projection.

### Hierarchy Compatibility

- unconstrained pooling;
- fixed compatible pooling;
- learned pooling with commutator penalty;
- learned pooling with exact parameterized compatibility.

### Router Compatibility

- standard top-k routing;
- routing with leakage logging only;
- routing with commutator loss;
- routing constrained to component-aware dispatch.

### Complement Control

- no spectral control;
- residual scaling only;
- spectral norm regularization;
- pseudospectral perturbation penalty;
- contractive complement update.

### Projector Type

- dense SVD/QR basis projector;
- sparse Poisson/Hodge solve;
- low-rank approximate projector;
- learned projector with audit against exact operator;
- implicit DEC projector.

## Success Criteria

The full model should demonstrate at least one of:

1. lower leakage at equal predictive error;
2. better long-rollout stability;
3. better resolution transfer;
4. lower commutator error across hierarchy;
5. clearer mode attribution;
6. improved solver convergence when used as a preconditioner or corrective operator.

A result is not persuasive if it improves short-horizon MSE but fails leakage or rollout tests.
