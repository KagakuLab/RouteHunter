from dataclasses import dataclass, field
from typing import Optional

from .core import PaperRecord, Target, canonicalize
from .store import TargetStore, CASPStore, PredictStore
from .casp import CaspSolvedEntry


@dataclass
class SearchResult:
    found: bool
    target: Target | None
    properties: dict[str, Optional[float]] = field(default_factory=dict)
    papers: list[PaperRecord] = field(default_factory=list)
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


class SearchEngine:
    def __init__(
        self,
        target_store: TargetStore,
        casp_store: CASPStore,
        predict_store: Optional[PredictStore] = None,
    ):
        self.target_store = target_store
        self.casp_store = casp_store
        self.predict_store = predict_store or PredictStore()

    def search(self, smiles: str) -> SearchResult:
        canon = canonicalize(smiles)  # raises InvalidSMILESError on bad input

        # Level 1: molecule properties, computed before
        properties = self.predict_store.predict_all(smiles)

        # Level 2: structural lookup against the static dataset.
        target = self.target_store.get_target(canon.inchikey)
        papers = self.target_store.get_papers_for_target(canon.inchikey) if target else []
        casp_solved = self.casp_store.solved_entries_for_inchikey(canon.inchikey)

        found = bool(papers) or bool(casp_solved)

        return SearchResult(
            found=found,
            target=target,
            properties=properties,
            papers=papers,
            casp_solved=casp_solved,
        )
