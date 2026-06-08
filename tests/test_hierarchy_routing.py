from __future__ import annotations

import torch

from gap_protected_transformers import LinearConstraint, commutator_error
from gap_protected_transformers.hierarchy import CompatibleRestriction, UnconstrainedRestriction
from gap_protected_transformers.routing import CompatibleRouter, UnconstrainedRouter
from gap_protected_transformers.diagnostics import routing_commutator_error
from gap_protected_transformers.toy_complexes import (
    componentwise_router,
    cycle_graph_hierarchy,
    cycle_graph_incidence,
    random_router,
)


def test_cycle_graph_pair_coarsening_commutator_distinguishes_maps() -> None:
    compatible = cycle_graph_hierarchy(8, compatible=True)
    incompatible = cycle_graph_hierarchy(8, compatible=False)

    assert commutator_error(*compatible) < 1e-12
    assert commutator_error(*incompatible) > 1e-3


def test_router_commutator_distinguishes_componentwise_and_random_maps() -> None:
    constraint = LinearConstraint(cycle_graph_incidence(6))
    compatible = componentwise_router(constraint.dim, 3)
    random = random_router(constraint.dim, 3, seed=12)

    assert routing_commutator_error(compatible, constraint, num_experts=3) < 1e-12
    assert routing_commutator_error(random, constraint, num_experts=3) > 1e-3


def test_learned_compatible_restriction_preserves_commutator() -> None:
    torch.manual_seed(13)
    A_c, _, R_A, A_f = cycle_graph_hierarchy(8, compatible=True)
    module = CompatibleRestriction(A_c, R_A, A_f)
    with torch.no_grad():
        module.raw.normal_()

    assert module.commutator_error() < 1e-12


def test_unconstrained_restriction_can_violate_commutator() -> None:
    torch.manual_seed(14)
    A_c, _, R_A, A_f = cycle_graph_hierarchy(8, compatible=True)
    module = UnconstrainedRestriction(torch.randn(4, 8, dtype=torch.float64))

    assert commutator_error(A_c, module.matrix(), R_A, A_f) > 1e-3


def test_learned_compatible_router_preserves_projection_commutator() -> None:
    torch.manual_seed(15)
    constraint = LinearConstraint(cycle_graph_incidence(6))
    router = CompatibleRouter(constraint, 3)
    with torch.no_grad():
        router.raw.normal_()

    assert router.commutator_error() < 1e-12


def test_unconstrained_router_can_violate_projection_commutator() -> None:
    torch.manual_seed(16)
    constraint = LinearConstraint(cycle_graph_incidence(6))
    router = UnconstrainedRouter(constraint.dim, 3)
    with torch.no_grad():
        router.raw.normal_()

    assert routing_commutator_error(router.matrix(), constraint, num_experts=3) > 1e-3
