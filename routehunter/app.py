"""
RouteHunterApp: single facade over RouteHunter's functionality.

The app is static: `RouteHunterApp.from_csv(path)` is the entry point
for every session -- it builds a fresh, in-memory RouteHunterStore
from a CSV and nothing else changes the dataset afterward. Each public
method corresponds to one of the functionalities: introduction /
search / hunt / casp / download.

There is no contribute()/review_submission() here on purpose -- adding
or correcting data means editing the CSV and starting a new session,
not calling an in-app method.
"""

from typing import Optional
import pandas as pd

from .core import PaperRecord
from .store import RouteHunterStore
from .intro import introduction as _introduction
from .search import search as _search, SearchResult
from .browse import hunt as _hunt, PaperFetcher, Classifier
from .casp import predict_route as _predict_route, CASPEngine, Route
from .seed import load_csv_seed as _load_csv_seed, SeedLoadReport


class RouteHunterApp:
    def __init__(self, store: RouteHunterStore):
        self.store = store

    @classmethod
    def from_csv(cls, csv_path: str, column_map: Optional[dict] = None) -> "RouteHunterApp":
        """
        Build the app fresh from a CSV -- the normal way to start a
        session. Prints nothing; call app.load_report to inspect what
        happened, or check store.stats() / app.introduction().
        """
        store = RouteHunterStore()
        report = _load_csv_seed(store, csv_path, column_map)
        app = cls(store)
        app.load_report = report  # kept for inspection, e.g. app.load_report.summary()
        return app

    # 1) Introduction --------------------------------------------------

    def introduction(self) -> str:
        return _introduction(self.store)

    # 2) Search ---------------------------------------------------------

    def search(self, smiles: str) -> SearchResult:
        return _search(self.store, smiles)

    # 3) Hunter -----------------------------------------------------------
    # Display-only: results are never written into the dataset.

    def browse(
        self,
        fetcher: PaperFetcher,
        classifier: Classifier,
        journals: list[str],
        n: int = 50,
    ) -> list[PaperRecord]:
        return _hunt(fetcher, classifier, journals, n)

    # 4) CASP -----------------------------------------------------------------
    # Cached in memory for this session only (never written to the CSV).

    def casp(
        self,
        engine: CASPEngine,
        smiles: str,
        engine_name: str = "unknown",
        cache: bool = True,
    ) -> Optional[Route]:
        return _predict_route(
            engine, smiles,
            store=self.store if cache else None,
            engine_name=engine_name,
        )

    # 5) Download ---------------------------------------------------------------

    def download(self, **filters) -> pd.DataFrame:
        from .download import download as _download
        return _download(self.store, **filters)