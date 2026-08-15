import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
        """Human-readable listing, sorted (already done at load time),
        probabilities shown as e.g. '76%', dates shown as e.g. '17/03/2014'."""
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
        publication_date as e.g. '17/03/2014', matching the rest
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
    monitor_high_path: str,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> MonitorResult:
    """
    Load the high-confidence Monitor file at an explicit path
    (resolved from config.csv's monitor_high entry -- see config.py).
    year_min/year_max filter to rows whose publication_date falls in
    that range (either or both may be omitted; omitting both returns
    the whole table). Either way, results are sorted by route_prob
    descending.

    If the file itself doesn't exist, returns available=False. If the
    file exists but no rows match the given range, returns
    available=True with an empty entries list -- those are different
    situations (no data at all vs. a filter that matched nothing).
    """
    path = Path(monitor_high_path)

    if not path.exists():
        return MonitorResult(
            year_min=year_min,
            year_max=year_max,
            available=False,
            entries=[],
            message="Monitor data is not available.",
        )

    df = pd.read_csv(path, parse_dates=["publication_date"])

    if year_min is not None:
        df = df[df["publication_date"].dt.year >= year_min]
    if year_max is not None:
        df = df[df["publication_date"].dt.year <= year_max]

    range_desc = _describe_range(year_min, year_max)

    if df.empty and (year_min is not None or year_max is not None):
        return MonitorResult(
            year_min=year_min,
            year_max=year_max,
            available=True,
            entries=[],
            message=f"No papers found{range_desc}.",
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

    return MonitorResult(
        year_min=year_min,
        year_max=year_max,
        available=True,
        entries=entries,
        message=f"{len(entries)} paper(s){range_desc}, sorted by predicted route probability.",
    )


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


def count_predicted_targets(monitor_medium_path: str) -> int:
    """
    Row count of the medium-confidence Monitor file at an explicit
    path (resolved from config.csv's monitor_medium entry) -- lower-
    certainty candidates than the high-confidence file, used only as
    a count (targets awaiting digitalization), not displayed as a
    browsable table. Returns 0 if the file doesn't exist rather than
    raising.
    """
    path = Path(monitor_medium_path)
    if not path.exists():
        return 0
    return len(pd.read_csv(path))