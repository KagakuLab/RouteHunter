"""
Introduction module.

Thin presentation layer over RouteHunterStore.stats(). Kept separate
from store.py so the *what to show* (this file) stays independent of
the *how to compute it* (store.py) -- e.g. formatting changes here
never touch the store.
"""

from .store import RouteHunterStore

USAGE_TEXT = """\
RouteHunter -- synthesis route reference lookup (static dataset)

  Search      : give a SMILES, get papers reporting a synthesis route to it.
  Hunter      : browse recently published papers, ranked by predicted
                probability of containing a multi-step synthesis route
                (candidates for you to review and add to the CSV by hand).
  CASP        : predict a route computationally for a target with no
                known literature synthesis (cached for this session only).
  Download    : export the dataset for AI/ML training.

This dataset is loaded from a CSV file; there is no in-app way to
modify it. To add or correct data, edit the CSV and reload.
"""


def introduction(store: RouteHunterStore) -> str:
    s = store.stats()
    lines = [USAGE_TEXT, "", "Current dataset:"]
    lines.append(f"  Targets                : {s['n_targets']}")
    lines.append(f"  Papers                 : {s['n_papers']}")
    lines.append(f"  Targets w/ >1 route    : {s['n_multi_route_targets']}")
    lines.append(f"  Cached CASP routes     : {s['n_cached_casp_routes']} (session only)")

    if s["papers_by_journal"]:
        lines.append("  Papers by journal:")
        for journal, count in sorted(s["papers_by_journal"].items(), key=lambda kv: -kv[1]):
            lines.append(f"    {journal:<40} {count}")

    if s["papers_by_contributor"]:
        lines.append("  Papers by contributor:")
        for contributor, count in sorted(s["papers_by_contributor"].items(), key=lambda kv: -kv[1]):
            lines.append(f"    {contributor:<40} {count}")

    return "\n".join(lines)