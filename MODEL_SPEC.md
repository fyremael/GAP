# Model Specification

## 1. Mathematical Objects

Let `V` be the representation space and let

`A: V -> W`

be a prescribed constraint operator. Examples:

- divergence operator for incompressible flow;
- graph incidence or boundary operator for edge flows;
- discrete exterior derivative in a chain complex;
- boundary-condition operator;
- gauge-fixing operator;
- mean-zero operator;
- routing-balance operator in MoE diagnostics.

Define

`K = ker A`,

and choose a complement `K_perp`, often `im A^*` under a chosen inner product. The representation decomposes as

`x = x_K + x_perp`.

The gap-protected design principle is:

> Carry `x_K` exactly or equivariantly. Learn on `x_perp`. Prevent hierarchy maps from mixing the two without audit.

## 2. Projected Residual Block

The simplest block is

`x_next = P_K(x + Phi_theta(x))`,

where

`P_K = I - A^dagger A`.

This guarantees `A x_next = 0`. It is useful when the entire state must remain inside the constraint manifold.

Limitation: this can discard useful complement information if the task requires both protected and learned channels.

## 3. Split-Channel Gap-Protected Block

A stronger formulation is

`x_K = P_K x`

`x_perp = P_perp x`

`x_K_next = T_K x_K`

`x_perp_next = T_perp_theta(x_perp, context)`

`x_next = x_K_next + x_perp_next`.

Here `T_K` is identity, equivariant, or analytically constrained. `T_perp_theta` is attention/MLP/MoE/neural-operator dynamics.

This is preferable when protected global structure and learnable local dynamics must coexist.

## 4. Hodge-Transformer Block

For a discrete complex, use a Hodge decomposition:

`k-form = exact + coexact + harmonic`.

A typical edge-flow case:

`x = grad phi + curl psi + h`.

Possible policies:

- preserve harmonic component `h` exactly;
- learn coexact/divergence-free dynamics with attention;
- project away forbidden exact leakage;
- use exact/curl components as side channels for interpretability.

## 5. Compatible Hierarchy

For multiscale models, define fine and coarse operators:

`A_f: V_f -> W_f`

`A_c: V_c -> W_c`.

Restriction `R: V_f -> V_c` and prolongation `P: V_c -> V_f` should satisfy, exactly or approximately:

`A_c R = R_A A_f`

and/or, for chain complexes,

`d_c P_k = P_{k-1} d_f`.

In ML terms, pooling, patchification, token merging, and expert routing must not destroy protected modes.

## 6. MoE Extension

Treat routing as a learned transfer map:

`R_theta: tokens -> expert streams`.

Measure whether routing commutes with the protected projection:

`C_route = ||R_theta P_K - P_K^expert R_theta||`.

This turns protected-mode compatibility into an MoE diagnostic. High commutator error means the router is scrambling protected structure.

## 7. Recommended First Architecture

`Input -> Hodge/Kernel Decompose -> Protected Transport + Complement Transformer -> Optional Compatible Down/Up -> Recompose -> Output Projector`

Model variants:

1. Vanilla Transformer.
2. PINN-style soft penalty.
3. Output-only hard projection.
4. Layerwise projection.
5. Split-channel Hodge Transformer.
6. Full compatible hierarchy with commutator loss.
7. MoE version with routing compatibility diagnostics.
