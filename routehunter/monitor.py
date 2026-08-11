"""
Monitor module (previously called Hunter/Browse).

Fully static, read-only: predicted route probabilities for papers are
computed OFFLINE, outside the app, using train_route_classifier.py
against whatever paper source you choose, and written to
rh_data/monitor/paper_route_prob.csv. The app never runs a classifier
here -- it just reads that file, optionally filters it by year, sorts
it, and returns it for display.

Expected CSV columns: journal, title, abstract, doi, route_prob,
publication_date. route_prob is already computed (0..1 float).
publication_date is a date string (e.g. "2014-03-17"), parsed here and
rendered for display as e.g. "March 17, 2014".
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_MONITOR_FILENAME = "paper_route_prob.csv"


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
        """e.g. 'March 17, 2014'. %-d / %e for a no-leading-zero day
        aren't portable across platforms, so build it manually instead."""
        if self.publication_date is None or pd.isna(self.publication_date):
            return ""
        d = self.publication_date
        return f"{d:%d/%m/%Y}"


@dataclass
class MonitorResult:
    year: Optional[int]  # the filter that was applied, or None if unfiltered
    available: bool
    entries: list[MonitorEntry] = field(default_factory=list)
    message: str = ""

    def format_table(self, limit: Optional[int] = None) -> str:
        """Human-readable listing, sorted (already done at load time),
        probabilities shown as e.g. '76%', dates shown as e.g. 'March 17, 2014'."""
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
        """
        Same data as format_table(), as a pandas DataFrame. route_prob
        is rendered as a percentage string (e.g. '76%') and
        publication_date as e.g. 'March 17, 2014', matching the rest
        of the app's display convention. Use `result.entries` instead
        if you want the raw float/Timestamp for further computation.
        """
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


def load(
    monitor_dir: str,
    year: Optional[int] = None,
    filename: str = DEFAULT_MONITOR_FILENAME,
) -> MonitorResult:
    """
    Load rh_data/<monitor_subdir>/<filename> (one file covering all
    years). If `year` is given, filters to rows whose publication_date
    falls in that year; if omitted, returns the whole table. Either
    way, results are sorted by route_prob descending.

    If the file itself doesn't exist, returns available=False. If the
    file exists but no rows match the given year, returns
    available=True with an empty entries list -- those are different
    situations (no data at all vs. a filter that matched nothing).
    """
    path = Path(monitor_dir) / filename

    if not path.exists():
        return MonitorResult(
            year=year,
            available=False,
            entries=[],
            message="Monitor data is not available.",
        )

    df = pd.read_csv(path, parse_dates=["publication_date"])

    if year is not None:
        df = df[df["publication_date"].dt.year == year]
        if df.empty:
            return MonitorResult(
                year=year,
                available=True,
                entries=[],
                message=f"No papers found for {year}.",
            )

    df = df.sort_values("route_prob", ascending=False)

    entries = [
        MonitorEntry(
            journal=row.get("journal"),
            title=row["title"],
            abstract=row.get("abstract"),
            doi=row["doi"],
            route_prob=float(row["route_prob"]),
            publication_date=row.get("publication_date"),
        )
        for _, row in df.iterrows()
    ]

    suffix = f" for {year}" if year is not None else ""
    return MonitorResult(
        year=year,
        available=True,
        entries=entries,
        message=f"{len(entries)} paper(s){suffix}, sorted by predicted route probability.",
    )