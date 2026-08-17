"""
Properties module.

This is "level 1" of Search: before any structural (InChIKey) lookup
happens, a set of registered PropertyPredictors run on the input
SMILES and produce a dict of named properties -- currently the
predicted probability of AiZynthFinder / SynPlanner finding a route.

This module holds no knowledge of which tools exist -- that's
app.py's TOOLS registry, the single source of truth for per-tool
config keys and display names. load_from_config just takes whatever
(model_key, display_name) pairs it's given.
"""

import pickle
from dataclasses import dataclass, field
from typing import Callable, Optional


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
    def load_from_config(cls, config_paths: dict[str, str], tools: list[tuple[str, str]]) -> "PropertyPredictorSet":
        """
        config_paths is the dict returned by config.load_config().
        tools is a list of (model_key, display_name) pairs -- see
        app.py's TOOLS registry. Any model_key missing from
        config.csv (that tool's model not trained yet) is skipped
        silently.
        """
        predictors = []
        for model_key, display_name in tools:
            path = config_paths.get(model_key)
            if path:
                predictors.append(load_pickled_proba_predictor(path, display_name))
        return cls(predictors=predictors)