from dataclasses import dataclass, field
from typing import Optional

from .core import PaperRecord, Target, CASPRouteRecord, canonicalize
from .store import RouteHunterStore
from .properties import PropertyPredictor, compute_properties
from .casp import CaspSolvedEntry


@dataclass
class SearchResult:
    found: bool
    target: Target | None
    properties: dict[str, Optional[float]] = field(default_factory=dict)
    papers: list[PaperRecord] = field(default_factory=list)
    casp_routes: list[CASPRouteRecord] = field(default_factory=list)
    casp_solved: list[CaspSolvedEntry] = field(default_factory=list)

    def report(self) -> str:
        """
        Full sectional print-ready report -- this is the one place
        that owns Search's display formatting, so callers (notebook
        cells, a future UI) just do `print(result.report())` instead
        of re-implementing the section logic and label lookups
        themselves each time.
        """
        sections = []

        if self.papers:
            lines = [f"Found {len(self.papers)} paper(s) reporting a route for this molecule:"]
            for p in self.papers:
                lines.append(f" - [paper] {p.title} ({p.journal}, {p.year}) doi:{p.doi}")
            sections.append("\n".join(lines))

        if self.casp_solved:
            lines = [f"Found {len(self.casp_solved)} tool(s) predicted routes for this molecule:"]
            for e in self.casp_solved:
                lines.append(
                    f" - [{e.tool_display}] This molecule was solved by {e.tool_display}. "
                    f"See predicted routes: {e.link}."
                )
            sections.append("\n".join(lines))

        if not self.found:
            lines = ["This molecule was not found, but you can try CASP tools:"]
            for name, value in self.properties.items():
                if value is not None:
                    lines.append(f" - The chance to be solved by {name}: {value:.0%}")
                else:
                    lines.append(f" - The chance to be solved by {name}: n/a")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)


def search(
    store: RouteHunterStore,
    smiles: str,
    property_predictors: Optional[list[PropertyPredictor]] = None,
) -> SearchResult:
    canon = canonicalize(smiles)  # raises InvalidSMILESError on bad input

    # Level 1: molecule properties, computed before -- and independent
    # of -- whether anything is found in the dataset below. Also
    # serves as the fallback if level 2 comes up empty.
    properties = compute_properties(smiles, property_predictors or [])

    # Level 2: structural lookup against the static dataset.
    target = store.get_target(canon.inchikey)
    papers = store.get_papers_for_target(canon.inchikey) if target else []
    casp_routes = store.get_casp_routes_for_target(canon.inchikey)
    casp_solved = store.get_casp_solved_entries(canon.inchikey)

    found = bool(papers) or bool(casp_routes) or bool(casp_solved)

    return SearchResult(
        found=found,
        target=target,
        properties=properties,
        papers=papers,
        casp_routes=casp_routes,
        casp_solved=casp_solved,
    )