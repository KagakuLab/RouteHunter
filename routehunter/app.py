import pickle
from typing import Optional
import pandas as pd

from .store import RouteHunterStore
from .intro import introduction as _introduction
from .search import search as _search, SearchResult
from .monitor import load as _load_monitor, MonitorResult, count_predicted_targets as _count_predicted_targets
from .predict import (
    predict_casp_solvability as _predict_casp_solvability,
    predict_route_probability as _predict_route_probability,
    PredictResult,
)
from .seed import load_csv_seed as _load_csv_seed, SeedLoadReport
from .properties import PropertyPredictorSet
from .casp import load_casp_table as _load_casp_table
from .config import load_config


def _load_route_model(path: str):
    """abstract_model.pickle needs nothing beyond plain pickle -- it's
    a TfidfVectorizer + LogisticRegression Pipeline, no custom classes
    involved, so the app package has no reason to depend on
    routehunter_build at runtime just to load it."""
    with open(path, "rb") as f:
        return pickle.load(f)


class RouteHunterApp:
    def __init__(
        self,
        store: RouteHunterStore,
        property_predictors: Optional[PropertyPredictorSet] = None,
        monitor_high_path: Optional[str] = None,
        route_model=None,
    ):
        self.store = store
        self.property_predictors = property_predictors or PropertyPredictorSet()
        self.monitor_high_path = monitor_high_path
        self.route_model = route_model

    @classmethod
    def from_data_dir(cls, rh_data_dir: str, column_map: Optional[dict] = None) -> "RouteHunterApp":
        """
        Build the app from rh_data_dir/config.csv -- the single
        manifest of every file path this app reads. There is no
        fallback to default paths: if config.csv is missing, this
        raises rather than guessing. Within config.csv, only seed_csv
        is required; every other key (casp_table, monitor_high,
        monitor_medium, AiZynthFinder, SynPlanner, abstract_model) is
        optional and skipped gracefully if absent -- that resource
        just won't be available for this session.
        """
        paths = load_config(rh_data_dir)  # raises FileNotFoundError if config.csv itself is missing

        if "seed_csv" not in paths:
            raise KeyError("config.csv must include a 'seed_csv' entry -- the app has no dataset without it.")

        store = RouteHunterStore()
        report = _load_csv_seed(store, paths["seed_csv"], column_map)

        if "casp_table" in paths:
            store.set_casp_table(_load_casp_table(paths["casp_table"]))

        if "monitor_medium" in paths:
            store.set_n_predicted_targets(_count_predicted_targets(paths["monitor_medium"]))

        predictors = PropertyPredictorSet.load_from_config(paths)

        route_model = _load_route_model(paths["abstract_model"]) if "abstract_model" in paths else None

        app = cls(
            store,
            property_predictors=predictors,
            monitor_high_path=paths.get("monitor_high"),
            route_model=route_model,
        )
        app.load_report = report  # kept for inspection, e.g. app.load_report.summary()
        return app

    @classmethod
    def from_csv(cls, csv_path: str, column_map: Optional[dict] = None) -> "RouteHunterApp":
        """CSV-only entry point, bypassing config.csv entirely -- no
        property models, no Monitor data, no route model. Useful for
        quick testing or if the rest isn't ready yet."""
        store = RouteHunterStore()
        report = _load_csv_seed(store, csv_path, column_map)
        app = cls(store)
        app.load_report = report
        return app

    # 1) Introduction --------------------------------------------------

    def introduction(self) -> str:
        return _introduction(self.store)

    # 2) Search ---------------------------------------------------------
    # Level 1 (properties) runs first, then level 2 (literature/CASP).

    def search(self, smiles: str) -> SearchResult:
        return _search(self.store, smiles, self.property_predictors.predictors)

    # 3) Monitor -------------------------------------------------------------
    # Fully static/read-only: route probabilities were computed offline.
    # This just reads, optionally filters by year, and sorts the high-
    # confidence file -- no classifier runs in the app itself.

    def monitor(self, year_min: Optional[int] = None, year_max: Optional[int] = None) -> MonitorResult:
        """No arguments returns the whole table; year_min/year_max
        (either or both) filter to papers published in that range."""
        if self.monitor_high_path is None:
            return MonitorResult(
                year_min=year_min, year_max=year_max, available=False, entries=[],
                message="No monitor file configured for this app.",
            )
        return _load_monitor(self.monitor_high_path, year_min=year_min, year_max=year_max)

    # 4) Predict --------------------------------------------------------------
    # No CASP engine is actually run here. predict_casp_solvability uses
    # the same per-tool solvability models as Search; predict_route_probability
    # uses the separate route classifier (not used by the GUI, exposed
    # here for batch use via the Python interface).

    def predict_casp_solvability(self, smiles: str) -> PredictResult:
        return _predict_casp_solvability(smiles, self.property_predictors.predictors)

    def predict_route_probability(self, text: str) -> PredictResult:
        return _predict_route_probability(text, self.route_model)

    # 5) Download ---------------------------------------------------------------

    def download(self, **filters) -> pd.DataFrame:
        from .download import download as _download
        return _download(self.store, **filters)