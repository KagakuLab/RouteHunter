
import pickle
from dataclasses import dataclass, field
from typing import Callable, Optional

# Config.csv keys that are solvability models -- these are also each
# model's display name, used directly in Search/Predict output.
SOLVABILITY_MODEL_KEYS = ["aizynthfinder", "synplanner"]


@dataclass
class PropertyPredictor:
    """Anything that maps a SMILES to a single named float property."""
    name: str
    predict: Callable[[str], float]


def load_pickled_proba_predictor(pickle_path: str, name: str) -> PropertyPredictor:
    """
    Wrap a pickled sklearn-style model (anything exposing
    predict_proba) as a PropertyPredictor. The pickle is expected to
    be self-contained (see routehunter_build's use of cloudpickle),
    so loading it here requires nothing beyond the file itself.
    """
    with open(pickle_path, "rb") as f:
        model = pickle.load(f)

    def predict(smiles: str) -> float:
        return float(model.predict_proba([smiles])[0, 1])

    return PropertyPredictor(name=name, predict=predict)


def compute_properties(smiles: str, predictors: list[PropertyPredictor]) -> dict[str, Optional[float]]:
    """
    Run every registered predictor on `smiles`. A predictor that
    raises (e.g. a molecule its underlying model can't handle) yields
    None for that property rather than failing the whole Search --
    one broken/missing model shouldn't block every other property or
    the literature/CASP lookup that follows.
    """
    properties: dict[str, Optional[float]] = {}
    for predictor in predictors:
        try:
            properties[predictor.name] = predictor.predict(smiles)
        except Exception:
            properties[predictor.name] = None
    return properties


@dataclass
class PropertyPredictorSet:
    predictors: list[PropertyPredictor] = field(default_factory=list)

    @classmethod
    def load_from_config(cls, config_paths: dict[str, str]) -> "PropertyPredictorSet":
        """
        config_paths is the dict returned by config.load_config().
        Only the known solvability-model keys are loaded here; any
        key missing from config.csv (model not trained yet) is
        skipped silently.
        """
        predictors = []
        for name in SOLVABILITY_MODEL_KEYS:
            print(name)
            path = config_paths.get(name)
            if path:
                predictors.append(load_pickled_proba_predictor(path, name))
        return cls(predictors=predictors)