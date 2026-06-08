# Figure Captions

## Figure 1: Gap-Protected Transformer Overview

A gap-protected transformer separates an input representation into protected and learnable components. Protected components encode quantities such as conservation laws, topology, harmonic modes, or divergence-free constraints. These are carried through unchanged or transformed by constrained maps. Learnable components are processed by attention, MLP, or MoE dynamics. The output recombines both channels while preserving the mathematical laws encoded by the protected subspace.

## Figure 2: Multigrid Analogy

Compatible multigrid preserves structure across fine and coarse levels by requiring transfer maps to commute with differential or constraint operators. The gap-protected transformer imports this principle into neural architecture: pooling, routing, token merging, and resolution transfer must not corrupt protected modes.

## Figure 3: PINN vs Gap-Protected Model

A PINN penalizes physics violations after they occur. A hard-constrained or gap-protected model prevents selected violations by construction. The model therefore learns inside a lawful representation space rather than being repeatedly punished for leaving it.

## Figure 4: MoE Routing as Transfer

Sparse expert routing can be viewed as a learned transfer map. If routing does not commute with protected-mode projectors, it can scramble global or conserved structure. Routing commutator diagnostics measure this failure directly.
