"""
Properties module.

This is "level 1" of Search: before any structural (InChIKey) lookup
happens, a set of registered PropertyPredictors run on the input
SMILES and produce a dict of named properties -- currently the
predicted probability of AiZynthFinder / SynPlanner finding a route,
loaded from pre-trained pickled models (see train_solvability_model.py,
a standalone script outside this package).

Deliberately a small registry rather than hardcoded AZ/SP fields: add
another PropertyPredictor (a different pickled model, a rule-based
descriptor, whatever) and register it alongside the existing two, and
every Search call picks it up automatically with no changes needed to
search.py or SearchResult.
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


@dataclass
class PropertyPredictor:
    """Anything that maps a SMILES to a single named float property."""
    name: str
    predict: Callable[[str], float]


def load_pickled_proba_predictor(pickle_path: str, name: str) -> PropertyPredictor:
    """
    Wrap a pickled sklearn-style model (anything exposing
    predict_proba, e.g. the Pipelines produced by
    train_solvability_model.py) as a PropertyPredictor. The pickle is
    expected to be self-contained -- see that script's use of
    cloudpickle -- so loading it here requires nothing beyond the
    file itself.
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
    """Convenience bundle for the standard rh_data/model/ layout:

        rh_data/model/az_model.pickle
        rh_data/model/sp_model.pickle

    Extend by adding more (path, name) pairs to `load_from_dir` below
    as more predictors are trained.
    """
    predictors: list[PropertyPredictor] = field(default_factory=list)

    @classmethod
    def load_from_dir(cls, model_dir: str) -> "PropertyPredictorSet":
        model_path = Path(model_dir)
        predictors = []
        for filename, name in [
            ("az_model.pickle", "AiZynthFinder"),
            ("sp_model.pickle", "SynPlanner"),
        ]:
            path = model_path / filename
            if path.exists():
                predictors.append(load_pickled_proba_predictor(str(path), name))
        return cls(predictors=predictors)