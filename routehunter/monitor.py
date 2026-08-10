"""
Monitor module (previously called Hunter/Browse).

Fully static, read-only: predicted route probabilities for recent
papers are computed OFFLINE, outside the app, using
train_route_classifier.py against whatever paper source you choose,
and written to rh_data/monitor/papers_<year>.csv. The app never runs a
classifier itself here -- it just reads a year's file, sorts it, and
returns it for display.

Expected CSV columns: journal, title, abstract, doi, route_prob
(route_prob already computed, 0..1 float).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class MonitorEntry:
    journal: Optional[str]
    title: str
    abstract: Optional[str]
    doi: str
    route_prob: float


@dataclass
class MonitorResult:
    year: int
    available: bool
    entries: list[MonitorEntry] = field(default_factory=list)
    message: str = ""

    def format_table(self, limit: Optional[int] = None) -> str:
        """Human-readable listing, sorted (already done at load time),
        probabilities shown as e.g. '76%'."""
        if not self.available:
            return self.message

        rows = self.entries[:limit] if limit else self.entries
        lines = []
        for e in rows:
            journal = e.journal or "(unknown journal)"
            lines.append(f"{e.route_prob:.0%}  [{journal}] {e.title}  doi:{e.doi}")
        return "\n".join(lines)

    def to_dataframe(self, limit: Optional[int] = None, include_abstract: bool = False) -> pd.DataFrame:
        """
        Same data as format_table(), as a pandas DataFrame instead of
        preformatted text -- sorts, column selection, etc. all just
        work as normal pandas operations. route_prob is rendered as a
        percentage string (e.g. '76%') to match the rest of the app's
        display convention; if you want the raw float for further
        computation, use `result.entries` directly instead.
        """
        cols = ["route_prob", "journal", "title"] + (["abstract"] if include_abstract else []) + ["doi"]
        if not self.available:
            return pd.DataFrame(columns=cols)

        rows = self.entries[:limit] if limit else self.entries
        data = {
            "route_prob": [f"{e.route_prob:.0%}" for e in rows],
            "journal": [e.journal for e in rows],
            "title": [e.title for e in rows],
            "doi": [e.doi for e in rows],
        }
        if include_abstract:
            data["abstract"] = [e.abstract for e in rows]

        return pd.DataFrame(data)[cols]


def load_year(monitor_dir: str, year: int) -> MonitorResult:
    """
    Load rh_data/<monitor_subdir>/papers_<year>.csv, sorted by
    route_prob descending. If the file doesn't exist, returns
    available=False with an explanatory message rather than raising --
    "no data for this year yet" is an expected, normal outcome, not an
    error.
    """
    path = Path(monitor_dir) / f"papers_{year}.csv"

    if not path.exists():
        return MonitorResult(
            year=year,
            available=False,
            entries=[],
            message=f"Data for {year} is not available.",
        )

    df = pd.read_csv(path)
    df = df.sort_values("route_prob", ascending=False)

    entries = [
        MonitorEntry(
            journal=row.get("journal"),
            title=row["title"],
            abstract=row.get("abstract"),
            doi=row["doi"],
            route_prob=float(row["route_prob"]),
        )
        for _, row in df.iterrows()
    ]

    return MonitorResult(
        year=year,
        available=True,
        entries=entries,
        message=f"{len(entries)} paper(s) for {year}, sorted by predicted route probability.",
    )