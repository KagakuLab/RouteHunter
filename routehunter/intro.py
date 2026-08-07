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
  
  Browse      : browse recently published papers, ranked by predicted
                probability of containing a multi-step synthesis route.
                
  Predict     : predict a route computationally for a target with no
                known literature synthesis (cached for this session only).
                
  Download    : export data for AI/ML training.

"""


def introduction(store: RouteHunterStore) -> str:
    s = store.stats()
    lines = [USAGE_TEXT, "", "Current dataset:"]
    lines.append(f"  Targets                  : {s['n_targets']}")
    lines.append(f"  Papers                   : {s['n_papers']}")
    lines.append(f"  Targets with >1 paper    : {s['n_multi_paper_targets']}")
    lines.append(f"  Cached CASP routes       : {s['n_cached_casp_routes']}")

    if s["papers_by_journal"]:
        lines.append("  Papers by journal:")
        for journal, count in sorted(s["papers_by_journal"].items(), key=lambda kv: -kv[1]):
            lines.append(f"    {journal:<40} {count}")

    if s["papers_by_source"]:
        lines.append("  Papers by source:")
        for source, count in s["papers_by_source"].items():
            lines.append(f"    {source:<40} {count}")

    return "\n".join(lines)