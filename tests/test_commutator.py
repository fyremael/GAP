from __future__ import annotations

import torch

from gap_protected_transformers import commutator_error
from gap_protected_transformers.toy_complexes import cycle_graph_incidence, identity_hierarchy


def test_identity_hierarchy_commutes_exactly() -> None:
    A = cycle_graph_incidence(5)
    A_c, R, R_A, A_f = identity_hierarchy(A)

    assert commutator_error(A_c, R, R_A, A_f) < 1e-12


def test_perturbed_hierarchy_has_nonzero_commutator() -> None:
    A = cycle_graph_incidence(5)
    A_c, R, R_A, A_f = identity_hierarchy(A)
    R = R.clone()
    R[0, 1] = 0.25

    assert commutator_error(A_c, R, R_A, A_f) > 1e-3
