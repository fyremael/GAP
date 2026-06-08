# Paper Spine

## Working Title

**Gap-Protected Neural Operators: Compatible Multigrid Principles for Structure-Preserving Attention**

## Abstract Draft

Neural operators and Transformers are increasingly used as surrogate solvers for physical, geometric, and graph-structured systems, yet their learned updates often mix modes that numerical analysis has long treated separately: constraint modes, harmonic modes, high-frequency correction modes, and coarse global modes. Physics-informed losses can penalize violations after they occur, but they do not by themselves prevent protected structure from leaking through depth, hierarchy, or routing. We introduce **gap-protected neural operators**, a transformer-native framework in which prescribed protected subspaces are transported exactly or equivariantly, learned dynamics are confined to controlled complements, and downsampling, prolongation, token merging, and expert routing are audited by compatibility relations inspired by compatible multigrid and discrete Hodge theory. The framework provides a diagnostic suite for protected-mode leakage, commutator error, gap collapse, pseudospectral sensitivity, and long-rollout drift. We evaluate the approach on divergence-free fields, graph edge flows, and sparse-MoE routing diagnostics, comparing vanilla attention, soft physics penalties, hard output projection, and fully compatible protected-mode architectures. The goal is not to replace all PINNs or neural operators, but to provide a disciplined architecture and measurement framework for settings where conservation, topology, gauge, or multiscale structure must not be treated as ordinary learnable noise.

## Core Contribution Claim

We contribute a unified transformer/MoE formulation for protected-mode learning:

1. **Protected-mode transport**: a specified kernel, harmonic subspace, conservation subspace, gauge subspace, or Hodge component is carried exactly, projected exactly, or transformed equivariantly.
2. **Complement learning**: attention, MLP, neural-operator, or MoE layers act primarily on the learned complement.
3. **Compatible hierarchy**: restriction, prolongation, pooling, patching, token merging, and expert routing are constrained or audited by commutator relations.
4. **Gap diagnostics**: layerwise measurements detect leakage, gap collapse, nonnormal amplification, and complement instability.
5. **Ablation framework**: utility is demonstrated not merely by MSE, but by constraint fidelity under depth, rollout, resolution transfer, and routing.

## What We Do Not Claim

We do not claim to invent Hodge decomposition, divergence-free neural networks, hard-constrained PINNs, neural multigrid, or transformer preconditioners. The novelty claim is the transformer-native synthesis: protected modes plus complement learning plus compatible hierarchy plus spectral/leakage diagnostics across depth and routing.

## Theorem Targets

### Theorem 1: Exact Protection Under Projected Residual Update
Let `A: V -> W` be a linear constraint map and `P_K` the orthogonal projector onto `ker A`. For any learnable map `Phi`, the update

`T(x) = P_K(x + Phi(x))`

satisfies `A T(x) = 0` for all `x`. If `x in ker A`, then all outputs remain in `ker A`.

This theorem is elementary but foundational. It states what the architecture guarantees before training.

### Theorem 2: No Hierarchy Leakage Under Exact Compatibility
Let `A_f` and `A_c` denote fine and coarse constraint maps, and let `R` be a restriction map satisfying

`A_c R = R_A A_f`.

Then if `A_f x = 0`, we have `A_c R x = 0`. Thus protected fine-level states remain protected after restriction.

### Theorem 3: Layerwise Leakage Bound Under Approximate Compatibility
If

`||A_c R - R_A A_f|| <= eps`

and `||x|| <= M`, then protected leakage after restriction is bounded by `eps M`.

This gives a practical diagnostic: the commutator norm controls worst-case constraint leakage.

### Theorem 4: Complement Stability Under Spectral Control
Suppose the complement update has Jacobian `J_perp` and satisfies

`sigma_max(J_perp) <= rho < 1 + delta`.

Then complement amplification over `L` layers is bounded by `rho^L`, modulo residual scaling and nonnormality correction terms.

This theorem must be written carefully, because attention and MoE routing are nonlinear and nonnormal. It is a diagnostic theorem, not a full convergence theorem.

## Experimental Thesis

A gap-protected model should win where ordinary models fail by violating structure: long rollout, resolution transfer, graph-flow consistency, and routing stability. It may not dominate short-horizon pointwise error on easy datasets. The benchmark must therefore measure lawfulness, not only accuracy.
