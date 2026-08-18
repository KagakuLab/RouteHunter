from typing import Optional

import pandas as pd

from .store import TargetStore, CASPStore, MonitorStore, CandidateStore, PredictStore
from .review import ReviewEngine
from .search import SearchEngine, SearchResult
from .predict import PredictEngine, PredictResult
from .utils import load_config

TargetStaticData = "TargetStaticData"
AizynthfinderStaticData = "AizynthfinderStaticData"
SynplannerStaticData = "SynplannerStaticData"
MonitorStaticData = "MonitorStaticData"
CandidateStaticData = "CandidateStaticData"
AbstractTrainingData = "AbstractTrainingData"
AizynthfinderPredictModel = "AizynthfinderPredictModel"
SynplannerPredictModel = "SynplannerPredictModel"


class RouteHunterApp:
    def __init__(
        self,
        target_store: TargetStore,
        casp_store: CASPStore,
        monitor_store: MonitorStore,
        candidate_store: CandidateStore,
        predict_store: PredictStore,
    ):
        self.target_store = target_store
        self.casp_store = casp_store
        self.monitor_store = monitor_store
        self.candidate_store = candidate_store
        self.predict_store = predict_store

        self.search_engine = SearchEngine(target_store, casp_store, predict_store)
        self.predict_engine = PredictEngine(predict_store)
        self.review_engine = ReviewEngine(target_store, candidate_store)

    @classmethod
    def from_data_dir(cls, init_data_dir: str) -> "RouteHunterApp":
        """Build the app from rh_data_dir/config.csv. Every data file
        is expected to be present in config.csv already; a missing
        key or file crashes here rather than degrading silently."""

        paths = load_config(init_data_dir)

        target_store = TargetStore(paths[TargetStaticData])

        casp_store = CASPStore(
            aizynthfinder_data_path=paths[AizynthfinderStaticData],
            synplanner_data_path=paths[SynplannerStaticData],
        )

        monitor_store = MonitorStore(paths[MonitorStaticData])
        candidate_store = CandidateStore(paths[CandidateStaticData])

        predict_store = PredictStore(
            aizynthfinder_model_path=paths[AizynthfinderPredictModel],
            synplanner_model_path=paths[SynplannerPredictModel],
        )

        app = cls(
            target_store,
            casp_store,
            monitor_store,
            candidate_store,
            predict_store,
        )
        return app

    # 1) Review
    def review(self) -> str:
        return self.review_engine.review()

    # 2) Search
    def search(self, smiles: str) -> SearchResult:
        return self.search_engine.search(smiles)

    # 3) Predict
    def predict_casp_solvability(self, smiles: str) -> PredictResult:
        return self.predict_engine.predict_casp_solvability(smiles)

    # 4) Monitor
    def monitor(self, year_min: Optional[int] = None, year_max: Optional[int] = None) -> pd.DataFrame:
        return self.monitor_store.get_papers_by_year(year_min=year_min, year_max=year_max)
