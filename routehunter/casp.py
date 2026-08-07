"""
CASP module.

Thin, swappable interface over an open-source CASP (computer-aided
synthesis planning) engine. Calling code depends only on the
CASPEngine protocol below, so trying a different backend
(AiZynthFinder, ASKCOS, etc.) later means writing one new adapter
class, not touching callers.

Predicted routes are cached into RouteHunterStore (keyed by InChIKey,
independent of whether a literature Target exists for that molecule),
so Search can surface them alongside literature papers.
"""

from typing import Optional, Protocol

from .core import Route, canonicalize, CASPRouteRecord
from .store import RouteHunterStore


class CASPEngine(Protocol):
    """Anything that can propose a route to a target SMILES."""
    def predict_route(self, smiles: str) -> Optional[Route]: ...


def predict_route(
    engine: CASPEngine,
    smiles: str,
    store: Optional[RouteHunterStore] = None,
    engine_name: str = "unknown",
) -> Optional[Route]:
    """
    Predict a route for `smiles` using `engine`.

    If `store` is given, cache the result (on success) as a
    CASPRouteRecord against the molecule's InChIKey, so a later Search
    for this molecule returns it too. Pass engine_name to identify
    which backend produced the route (useful once more than one CASP
    engine is wired in).
    """
    route = engine.predict_route(smiles)
    if route is None:
        return None

    if store is not None:
        canon = canonicalize(smiles)
        store.add_casp_route(
            CASPRouteRecord(
                inchikey=canon.inchikey,
                canonical_smiles=canon.canonical_smiles,
                route=route,
                engine_name=engine_name,
                score=route.score,
            )
        )

    return route