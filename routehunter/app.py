import pickle
from dataclasses import dataclass
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


@dataclass
class ToolConfig:
    display_name: str  # shown throughout Search/Predict output
    model_key: str      # config.csv key for this tool's solvability model pickle
    data_key: str       # config.csv key for this tool's raw (smiles, is_solved) data
    url: str            # link to the tool, shown in Predict output

TargetStaticData = "TargetStaticData"
AizynthfinderStaticData = "AizynthfinderStaticData"
SynplannerStaticData = "SynplannerStaticData"
MonitorStaticData = "MonitorStaticData"
CandidateStaticData = "CandidateStaticData"
AbstractTrainingData = "AbstractTrainingData"
AizynthfinderPredictModel = "AizynthfinderPredictModel"
SynplannerPredictModel = "SynplannerPredictModel"
PaperPredictModel = "PaperPredictModel"

TOOLS = [
    ToolConfig(
        display_name="AiZynthFinder",
        model_key=AizynthfinderPredictModel,
        data_key=AizynthfinderStaticData,
        url="https://github.com/MolecularAI/aizynthfinder",
    ),
    ToolConfig(
        display_name="SynPlanner",
        model_key=SynplannerPredictModel,
        data_key=SynplannerStaticData,
        url="https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner",
    ),
]


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
        tool_links: Optional[dict[str, str]] = None,
    ):
        self.store = store
        self.property_predictors = property_predictors or PropertyPredictorSet()
        self.monitor_high_path = monitor_high_path
        self.route_model = route_model
        self.tool_links = tool_links or {}

    @classmethod
    def from_data_dir(cls, rh_data_dir: str, column_map: Optional[dict] = None) -> "RouteHunterApp":
        """
        Build the app from rh_data_dir/config.csv -- the single
        manifest of every file path this app reads. There is no
        fallback to default paths: if config.csv is missing, this
        raises rather than guessing. Within config.csv, only
        TargetStaticData is required; every other name (see the
        globals and TOOLS above) is optional and skipped gracefully
        if absent -- that resource just won't be available for this
        session.
        """
        paths = load_config(rh_data_dir)  # raises FileNotFoundError if config.csv itself is missing

        store = RouteHunterStore()
        report = _load_csv_seed(store, paths[TargetStaticData], column_map)

        casp_tool_paths = {
            tool.display_name: paths[tool.data_key]
            for tool in TOOLS if tool.data_key in paths
        }
        if casp_tool_paths:
            store.set_casp_table(_load_casp_table(casp_tool_paths))

        if CandidateStaticData in paths:
            store.set_n_predicted_targets(_count_predicted_targets(paths[CandidateStaticData]))

        predictors = PropertyPredictorSet.load_from_config(
            paths, [(tool.model_key, tool.display_name) for tool in TOOLS],
        )
        tool_links = {tool.display_name: tool.url for tool in TOOLS}

        route_model = _load_route_model(paths[PaperPredictModel]) if PaperPredictModel in paths else None

        app = cls(
            store,
            property_predictors=predictors,
            monitor_high_path=paths.get(MonitorStaticData),
            route_model=route_model,
            tool_links=tool_links,
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
        return _predict_casp_solvability(smiles, self.property_predictors.predictors, self.tool_links)

    def predict_route_probability(self, text: str) -> PredictResult:
        return _predict_route_probability(text, self.route_model)