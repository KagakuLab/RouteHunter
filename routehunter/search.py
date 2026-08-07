"""
Search module.

Exact-structure lookup: SMILES -> InChIKey -> (literature papers,
cached CASP routes) for that InChIKey. This is O(1) and correct for
identical structures.

A molecule can have literature papers, cached CASP routes, both, or
neither -- `found` reflects whether there is anything at all to show,
not just literature.

NOTE: this does not do substructure or similarity search. If/when that
is needed, add a fingerprint index alongside the inchikey index in
RouteHunterStore rather than reworking this function's contract.
"""

from dataclasses import dataclass, field

from .core import PaperRecord, Target, CASPRouteRecord, canonicalize
from .store import RouteHunterStore


@dataclass
class SearchResult:
    found: bool
    target: Target | None
    papers: list[PaperRecord] = field(default_factory=list)
    casp_routes: list[CASPRouteRecord] = field(default_factory=list)
    message: str = ""


def search(store: RouteHunterStore, smiles: str) -> SearchResult:
    canon = canonicalize(smiles)  # raises InvalidSMILESError on bad input
    target = store.get_target(canon.inchikey)
    papers = store.get_papers_for_target(canon.inchikey) if target else []
    casp_routes = store.get_casp_routes_for_target(canon.inchikey)

    found = bool(papers) or bool(casp_routes)

    if not found:
        message = "No synthesis route found for this structure."
    else:
        parts = []
        if papers:
            parts.append(f"{len(papers)} paper(s) reporting a route")
        if casp_routes:
            parts.append(f"{len(casp_routes)} cached CASP-predicted route(s)")
        message = "Found " + " and ".join(parts) + " for this target."

    return SearchResult(
        found=found,
        target=target,
        papers=papers,
        casp_routes=casp_routes,
        message=message,
    )
