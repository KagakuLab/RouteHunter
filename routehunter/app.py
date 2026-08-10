"""
RouteHunterApp: single facade over RouteHunter's functionality.

The app is static: `RouteHunterApp.from_data_dir(rh_data_path)` is the
normal entry point for a session. It expects the layout:

    rh_data/core/routehunter_seed.csv        -- the dataset (required)
    rh_data/core/routehunter_casp.csv        -- static AZ/SP/... solved-by-tool table (optional)
    rh_data/model/az_model.pickle            -- AiZynthFinder solvability model (optional)
    rh_data/model/sp_model.pickle            -- SynPlanner solvability model (optional)
    rh_data/monitor/paper_route_prob.csv     -- pre-scored papers, all years (optional)

Nothing else changes the dataset afterward -- to add/correct data,
retrain a property model, or refresh a year's Monitor file, do that
offline (edit the CSV / re-run train_solvability_model.py or
train_route_classifier.py) and start a new session.

Each public method corresponds to one of the functionalities:
introduction / search / monitor / predict / casp / download. There is
no contribute()/review_submission() on purpose -- see seed.py.
"""

from typing import Optional
from pathlib import Path
import pandas as pd

from .store import RouteHunterStore
from .intro import introduction as _introduction
from .search import search as _search, SearchResult
from .monitor import load as _load_monitor, MonitorResult
from .predict import predict as _predict, PredictResult
from .seed import load_csv_seed as _load_csv_seed, SeedLoadReport
from .properties import PropertyPredictorSet
from .casp import load_casp_table as _load_casp_table

DEFAULT_CSV_SUBPATH = "core/routehunter_seed.csv"
DEFAULT_CASP_TABLE_SUBPATH = "core/routehunter_casp.csv"
DEFAULT_MODEL_SUBDIR = "model"
DEFAULT_MONITOR_SUBDIR = "monitor"


class RouteHunterApp:
    def __init__(
        self,
        store: RouteHunterStore,
        property_predictors: Optional[PropertyPredictorSet] = None,
        monitor_dir: Optional[str] = None,
    ):
        self.store = store
        self.property_predictors = property_predictors or PropertyPredictorSet()
        self.monitor_dir = monitor_dir

    @classmethod
    def from_data_dir(
        cls,
        rh_data_dir: str,
        csv_subpath: str = DEFAULT_CSV_SUBPATH,
        casp_table_subpath: str = DEFAULT_CASP_TABLE_SUBPATH,
        model_subdir: str = DEFAULT_MODEL_SUBDIR,
        monitor_subdir: str = DEFAULT_MONITOR_SUBDIR,
        column_map: Optional[dict] = None,
    ) -> "RouteHunterApp":
        """
        Build the app from the standard rh_data/ layout: loads the CSV
        from rh_data/<csv_subpath>, the static CASP-solved table from
        rh_data/<casp_table_subpath> (if present), any az_model.pickle
        / sp_model.pickle found under rh_data/<model_subdir>/, and
        remembers rh_data/<monitor_subdir>/ for on-demand lookups via
        app.monitor(year=...). Everything except the seed CSV is
        optional and skipped silently if missing.
        """
        base = Path(rh_data_dir)
        store = RouteHunterStore()
        report = _load_csv_seed(store, str(base / csv_subpath), column_map)
        store.set_casp_table(_load_casp_table(str(base / casp_table_subpath)))
        predictors = PropertyPredictorSet.load_from_dir(str(base / model_subdir))

        app = cls(store, property_predictors=predictors, monitor_dir=str(base / monitor_subdir))
        app.load_report = report  # kept for inspection, e.g. app.load_report.summary()
        return app

    @classmethod
    def from_csv(cls, csv_path: str, column_map: Optional[dict] = None) -> "RouteHunterApp":
        """CSV-only entry point, no property models, no Monitor data.
        Useful for quick testing or if the rest isn't ready yet."""
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
    # Fully static/read-only: route probabilities were computed offline
    # and written to rh_data/monitor/paper_route_prob.csv. This just
    # reads, optionally filters by year, and sorts that file -- no
    # classifier runs in the app itself.

    def monitor(self, year: Optional[int] = None) -> MonitorResult:
        """year=None (default) returns the whole table; pass a year
        to filter to papers published in that year."""
        if self.monitor_dir is None:
            return MonitorResult(
                year=year, available=False, entries=[],
                message="No monitor directory configured for this app.",
            )
        return _load_monitor(self.monitor_dir, year=year)

    # 4) Predict --------------------------------------------------------------
    # No CASP engine is actually run here -- just the same solvability
    # models used by Search, paired with a link to each tool.

    def predict(self, smiles: str) -> PredictResult:
        return _predict(smiles, self.property_predictors.predictors)

    # 5) Download ---------------------------------------------------------------

    def download(self, **filters) -> pd.DataFrame:
        from .download import download as _download
        return _download(self.store, **filters)