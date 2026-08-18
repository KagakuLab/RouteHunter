from typing import Optional

from .store import TargetStore, CASPStore, MonitorStore, CandidateStore, PredictStore
from .review import ReviewEngine
from .search import SearchEngine, SearchResult
from .predict import PredictEngine, PredictResult
from .monitor import MonitorEngine, MonitorResult
from .candidate import CandidateEngine
from .config import load_config


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
        self.monitor_engine = MonitorEngine(monitor_store)
        self.candidate_engine = CandidateEngine(candidate_store)
        self.review_engine = ReviewEngine(target_store, self.candidate_engine)

    @classmethod
    def from_data_dir(cls, init_data_dir: str) -> "RouteHunterApp":
        """Build the app from rh_data_dir/config.csv."""

        # load data location config
        paths = load_config(init_data_dir)

        target_store = TargetStore(paths[TargetStaticData])

        casp_store = CASPStore(
            aizynthfinder_data_path=paths.select(AizynthfinderStaticData),
            synplanner_data_path=paths.select(SynplannerStaticData),
        )

        monitor_store = MonitorStore(paths.select(MonitorStaticData))
        candidate_store = CandidateStore(paths.select(CandidateStaticData))

        predict_store = PredictStore(
            aizynthfinder_model_path=paths.select(AizynthfinderPredictModel),
            synplanner_model_path=paths.select(SynplannerPredictModel),
        )

        return cls(
            target_store,
            casp_store,
            monitor_store,
            candidate_store,
            predict_store,
        )

    # 1) Review
    def review(self) -> str:
        return self.review_engine.review()

    # 2) Search
    def search(self, smiles: str) -> SearchResult:
        return self.search_engine.search(smiles)

    # 3) Predict
    def predict(self, smiles: str) -> PredictResult:
        return self.predict_engine.predict(smiles)

    # 4) Monitor
    def monitor(self, year_min: Optional[int] = None, year_max: Optional[int] = None) -> MonitorResult:
        return self.monitor_engine.select(year_min=year_min, year_max=year_max)
