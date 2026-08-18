from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .core import PaperRecord, Target, canonicalize
from .store import TargetStore, CASPStore, PredictStore


@dataclass
class SearchResult:
    found: bool
    target: Target | None
    properties: dict[str, Optional[float]] = field(default_factory=dict)
    paper_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    paper_message: str = ""
    casp_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    casp_message: str = ""


class SearchEngine:
    def __init__(
        self,
        target_store: TargetStore,
        casp_store: CASPStore,
        predict_store: PredictStore,
    ):
        self.target_store = target_store
        self.casp_store = casp_store
        self.predict_store = predict_store

    def search(self, smiles: str) -> SearchResult:
        canon = canonicalize(smiles)  # raises InvalidSMILESError on bad input

        # Level 1: molecule properties, computed before -- and independent
        # of -- whether anything is found in the dataset below. Also
        # serves as the fallback if level 2 comes up empty.
        properties = self.predict_store.predict(smiles)

        # Level 2: structural lookup against the static dataset.
        target = self.target_store.targets.get(canon.inchikey)
        papers = self.target_store.get_papers_for_target(canon.inchikey) if target else []
        casp_solved = self.casp_store.get_tools_for_target(canon.inchikey)

        found = bool(papers) or bool(casp_solved)

        result = SearchResult(
            found=found,
            target=target,
            properties=properties,
            paper_report=self._papers_to_dataframe(papers),
            paper_message=self._paper_message(papers),
            casp_report=self._casp_to_dataframe(casp_solved),
            casp_message=self._casp_message(casp_solved),
        )
        return result

    @staticmethod
    def _papers_to_dataframe(papers: list[PaperRecord]) -> pd.DataFrame:
        rows = []
        for p in papers:
            rows.append({
                "journal": p.journal,
                "title": p.title,
                "year": p.year,
                "doi": p.doi,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _paper_message(papers: list[PaperRecord]) -> str:
        if not papers:
            return "No papers found reporting a route for this molecule"
        return f"Found {len(papers)} paper(s) reporting a route for this molecule"

    @staticmethod
    def _casp_to_dataframe(casp_solved: list[str]) -> pd.DataFrame:
        rows = []
        for tool_name in casp_solved:
            rows.append({
                "tool": tool_name,
                "result": f"Solved by {tool_name}",
                "route": "Cached predicted routes are not available yet",
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _casp_message(casp_solved: list[str]) -> str:
        if not casp_solved:
            return "No predictions were obtained by CASP tools for this molecule"
        return f"Found {len(casp_solved)} tool(s) predicted routes for this molecule"