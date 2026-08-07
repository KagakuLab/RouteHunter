"""
Hunter module.

Pulls recent papers (title + abstract) from a source, scores each with
a classifier that predicts P(paper contains a multi-step route), and
returns them ranked. Deliberately decoupled from the classifier's
internals -- this file only depends on the small Classifier protocol
below, so the actual RouteHunter modeling pipeline (benchmarking,
consensus layer, etc.) can live in its own module/repo untouched.

Hunter output is display-only: it is never written into the dataset.
The app is static and has no in-app write path -- if a Hunter
candidate turns out to be a real route, add it to the CSV by hand and
reload.
"""

from dataclasses import dataclass
from typing import Protocol

from .core import PaperRecord, PaperSource


@dataclass
class PaperCandidate:
    """A paper pulled from a source, not yet scored."""
    doi: str
    title: str
    abstract: str
    journal: str | None = None
    year: int | None = None


class PaperFetcher(Protocol):
    """Anything that can retrieve recent papers for a set of journals."""
    def fetch_recent(self, journals: list[str], n: int) -> list[PaperCandidate]: ...


class Classifier(Protocol):
    """Anything exposing predict_proba(title, abstract) -> float in [0, 1]."""
    def predict_proba(self, title: str, abstract: str) -> float: ...


def hunt(
    fetcher: PaperFetcher,
    classifier: Classifier,
    journals: list[str],
    n: int = 50,
) -> list[PaperRecord]:
    """
    Fetch up to `n` recent papers from `journals`, score each with
    `classifier`, and return them as PaperRecord objects sorted by
    predicted probability (descending). source=HUNTER. These are
    candidates for review only -- nothing here touches the dataset.
    """
    candidates = fetcher.fetch_recent(journals, n)

    scored = []
    for c in candidates:
        score = classifier.predict_proba(c.title, c.abstract or "")
        scored.append(
            PaperRecord(
                doi=c.doi,
                title=c.title,
                abstract=c.abstract,
                journal=c.journal,
                year=c.year,
                source=PaperSource.HUNTER,
                hunter_score=score,
            )
        )

    scored.sort(key=lambda p: p.hunter_score, reverse=True)
    return scored