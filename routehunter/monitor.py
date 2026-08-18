import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MonitorStore


@dataclass
class MonitorEntry:
    journal: Optional[str]
    title: str
    abstract: Optional[str]
    doi: str
    route_prob: float
    publication_date: Optional[pd.Timestamp]

    @property
    def formatted_date(self) -> str:
        """e.g. '17/03/2014'."""
        if self.publication_date is None or pd.isna(self.publication_date):
            return ""
        d = self.publication_date
        return f"{d:%d/%m/%Y}"


@dataclass
class MonitorResult:
    year_min: Optional[int]  # the filter that was applied, or None if unfiltered
    year_max: Optional[int]
    available: bool
    entries: list[MonitorEntry] = field(default_factory=list)
    message: str = ""

    def format_table(self, limit: Optional[int] = None) -> str:

        if not self.available:
            return self.message

        rows = self.entries[:limit] if limit else self.entries
        lines = []
        for e in rows:
            journal = e.journal or "(unknown journal)"
            date = e.formatted_date or "(unknown date)"
            lines.append(f"{e.route_prob:.0%}  [{journal}] {e.title}  ({date})  doi:{e.doi}")
        return "\n".join(lines)

    def to_dataframe(self, limit: Optional[int] = None, include_abstract: bool = False) -> pd.DataFrame:

        cols = ["route_prob", "journal", "title"] + (["abstract"] if include_abstract else []) + ["publication_date", "doi"]
        if not self.available:
            return pd.DataFrame(columns=cols)

        rows = self.entries[:limit] if limit else self.entries
        data = {
            "route_prob": [f"{e.route_prob:.0%}" for e in rows],
            "journal": [e.journal for e in rows],
            "title": [e.title for e in rows],
            "publication_date": [e.formatted_date for e in rows],
            "doi": [e.doi for e in rows],
        }
        if include_abstract:
            data["abstract"] = [e.abstract for e in rows]

        return pd.DataFrame(data)[cols]


class MonitorEngine:
    def __init__(self, store: "MonitorStore"):
        self.store = store

    def select(self, year_min: Optional[int] = None, year_max: Optional[int] = None) -> MonitorResult:
        entries = self.store.get_entries(year_min=year_min, year_max=year_max)
        return self._to_result(entries, year_min, year_max)

    @staticmethod
    def _to_result(
        entries: Optional[list[MonitorEntry]],
        year_min: Optional[int],
        year_max: Optional[int],
    ) -> MonitorResult:
        """Package already-filtered/sorted entries into a MonitorResult, or an unavailable result if entries is None."""
        if entries is None:
            return MonitorResult(
                year_min=year_min,
                year_max=year_max,
                available=False,
                entries=[],
                message="Monitor data is not available.",
            )

        range_desc = MonitorEngine._describe_range(year_min, year_max)

        if not entries and (year_min is not None or year_max is not None):
            return MonitorResult(
                year_min=year_min,
                year_max=year_max,
                available=True,
                entries=[],
                message=f"No papers found{range_desc}.",
            )

        return MonitorResult(
            year_min=year_min,
            year_max=year_max,
            available=True,
            entries=entries,
            message=f"{len(entries)} paper(s){range_desc}, sorted by predicted route probability.",
        )

    @staticmethod
    def _describe_range(year_min: Optional[int], year_max: Optional[int]) -> str:
        if year_min is not None and year_max is not None:
            if year_min == year_max:
                return f" for {year_min}"
            return f" for {year_min}-{year_max}"
        if year_min is not None:
            return f" from {year_min} onward"
        if year_max is not None:
            return f" up to {year_max}"
        return ""
