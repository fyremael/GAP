# Related-Work Boundary and Novelty-Safe Framing

## Prior Areas We Build On

1. **Divergence-free neural networks and conservation by construction.**
   Prior work shows that neural networks can be parameterized through differential forms so conservation laws such as the continuity equation are satisfied intrinsically, without penalty losses.

2. **Hodge neural operators and topology-preserving operator learning.**
   Recent work uses Hodge decomposition and Hodge orthogonality to isolate topological degrees of freedom from learnable geometric dynamics.

3. **Structure-preserving operator learning.**
   Prior frameworks use finite element discretizations and related structure-preserving machinery to preserve boundary conditions and mathematical properties at the discrete level.

4. **Neural multigrid and neural preconditioning.**
   Neural solvers and preconditioners increasingly incorporate multigrid ideas, including hierarchy and spectral conditioning.

5. **Graph Hodge learning and spectral graph Transformers.**
   Existing models use Hodge Laplacians, edge-flow decompositions, and graph spectral structure as learning priors.

## What We Should Not Claim

Do not claim:

- first Hodge neural network;
- first divergence-free network;
- first hard-constrained PINN;
- first neural multigrid method;
- first Transformer preconditioner;
- first topology-preserving neural operator;
- inherited multigrid convergence guarantees without explicit hypotheses.

## Novelty-Safe Claim

A defensible claim is:

> We introduce a transformer-native compatible-smoother formulation in which protected modes are carried exactly or equivariantly, learned dynamics are confined to controlled complements, and hierarchy/routing maps are audited or constrained by commuting-relation diagnostics.

## Strongest Differentiators

1. Transformer/MoE formulation rather than only neural operator or solver formulation.
2. Explicit protected-mode leakage diagnostics at every layer.
3. Commutator diagnostics for pooling, resolution transfer, and expert routing.
4. Gap preservation and complement amplification metrics.
5. Ablation framework comparing soft penalties, hard output projection, layerwise projection, split-channel protection, and compatible hierarchy.
6. Routing-as-transfer-operator interpretation for sparse MoE systems.

## Recommended Related-Work Language

> Our work builds on hard-constrained neural conservation laws, Hodge-based neural operator learning, structure-preserving finite-element operator networks, and neural multigrid/preconditioning. Rather than treating these as separate threads, we formulate a transformer-native protected-mode architecture: the network carries known kernel or Hodge components through depth, learns only on controlled complements, and audits every hierarchy or routing transition by the commutator errors that compatible multigrid would demand to vanish.
