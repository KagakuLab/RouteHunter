
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .properties import PropertyPredictor

# Maps a property predictor's name (the tool's display name, e.g.
# "AiZynthFinder" -- see PropertyPredictorSet.load_from_config) to a
# link. Add an entry here for any future solvability model alongside
# its own PropertyPredictor registration.
TOOL_LINKS: dict[str, str] = {
    "AiZynthFinder": "https://github.com/MolecularAI/aizynthfinder",
    "SynPlanner": "https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner",
}


@dataclass
class ToolPrediction:
    tool_name: str
    probability: Optional[float]
    url: str


@dataclass
class PredictResult:
    input_value: str  # the SMILES or text that was actually predicted on
    predictions: list[ToolPrediction] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Probability rendered as a percentage string (e.g. '76%'),
        matching the rest of the app's display convention."""
        return pd.DataFrame([
            {
                "tool": p.tool_name,
                "probability": f"{p.probability:.0%}" if p.probability is not None else "n/a",
                "url": p.url,
            }
            for p in self.predictions
        ])


def predict_casp_solvability(smiles: str, property_predictors: list[PropertyPredictor]) -> PredictResult:
    """
    Run every registered solvability predictor on `smiles` and pair
    each result with its tool's link. A predictor that fails on this
    molecule yields probability=None for that tool rather than
    failing the whole call.
    """
    predictions = []
    for predictor in property_predictors:
        url = TOOL_LINKS.get(predictor.name, "")
        try:
            probability = predictor.predict(smiles)
        except Exception:
            probability = None
        predictions.append(ToolPrediction(tool_name=predictor.name, probability=probability, url=url))

    return PredictResult(input_value=smiles, predictions=predictions)


def predict_route_probability(text: str, route_model) -> PredictResult:
    """
    Run the route classifier (abstract_model.pickle) on `text` --
    already-combined title+abstract, see routehunter_build.models.
    combine_text for the expected format. route_model=None (e.g. the
    file wasn't present in config.csv) yields probability=None rather
    than raising.
    """
    if route_model is None:
        probability = None
    else:
        try:
            probability = route_model.predict_proba([text])[0, 1]
        except Exception:
            probability = None

    prediction = ToolPrediction(tool_name="Route classifier", probability=probability, url="")
    return PredictResult(input_value=text, predictions=[prediction])