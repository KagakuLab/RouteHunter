import pickle
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from .store import RouteHunterStore
from .review import review as _introduction
from .search import search as _search, SearchResult
from .monitor import load as _load_monitor, MonitorResult, count_predicted_targets as _count_predicted_targets
from .predict import (
    predict_casp_solvability as _predict_casp_solvability,
    predict_route_probability as _predict_route_probability,
    PredictResult,
)
from .seed import load_csv_seed as _load_csv_seed, SeedLoadReport
from .property import PropertyPredictorSet
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
    def from_data_dir(cls, init_data_dir: str, column_map: Optional[dict] = None) -> "RouteHunterApp":
        """Build the app from rh_data_dir/config.csv."""

        # load data location config
        paths = load_config(init_data_dir)

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

    # 1) Review
    def review(self) -> str:
        return _introduction(self.store)

    # 2) Search
    def search(self, smiles: str) -> SearchResult:
        return _search(self.store, smiles, self.property_predictors.predictors)

    # 3) Predict
    def predict_casp_solvability(self, smiles: str) -> PredictResult:
        return _predict_casp_solvability(smiles, self.property_predictors.predictors, self.tool_links)

    def predict_route_probability(self, text: str) -> PredictResult:
        return _predict_route_probability(text, self.route_model)

    # 4) Monitor
    def monitor(self, year_min: Optional[int] = None, year_max: Optional[int] = None) -> MonitorResult:
        return _load_monitor(self.monitor_high_path, year_min=year_min, year_max=year_max)

