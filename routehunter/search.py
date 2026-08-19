from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from .core import Paper, Target, canonicalize
from .store import TargetStore, ToolStore
from .predict import PredictEngine


@dataclass
class SearchResult:
    found: bool
    target: Target | None
    properties: dict[str, Optional[float]] = field(default_factory=dict)
    paper_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    paper_message: str = ""
    tool_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    tool_message: str = ""


class SearchEngine:
    def __init__(
        self,
        target_store: TargetStore,
        tool_store: ToolStore,
        predict_engine: PredictEngine,
    ):
        self.target_store = target_store
        self.tool_store = tool_store
        self.predict_engine = predict_engine

    def search(self, smiles: str) -> SearchResult:
        canon = canonicalize(smiles)
        properties = self.predict_engine.predict_solvability(smiles).predictions
        target = self.target_store.targets.get(canon.inchikey)
        papers = self.target_store.get_papers_for_target(canon.inchikey) if target else []
        tools = self.tool_store.get_tools_for_target(canon.inchikey)

        found = bool(papers) or bool(tools)

        result = SearchResult(
            found=found,
            target=target,
            properties=properties,
            paper_report=self._papers_to_dataframe(papers),
            paper_message=self._paper_message(papers),
            tool_report=self._tools_to_dataframe(tools),
            tool_message=self._tool_message(tools),
        )
        return result

    @staticmethod
    def _papers_to_dataframe(papers: list[Paper]) -> pd.DataFrame:
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
    def _paper_message(papers: list[Paper]) -> str:
        if not papers:
            return "No papers found reporting a route for this molecule"
        return f"Found {len(papers)} paper(s) reporting a route for this molecule"

    @staticmethod
    def _tools_to_dataframe(solved_list: list[str]) -> pd.DataFrame:
        rows = []
        for tool_name in solved_list:
            rows.append({
                "tool": tool_name,
                "result": f"Solved by {tool_name}",
                "route": "Cached predicted routes are not available yet",
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _tool_message(solved_list: list[str]) -> str:
        if not solved_list:
            return "No predictions were obtained by CASP tools for this molecule"
        return f"Found {len(solved_list)} tool(s) predicted routes for this molecule"