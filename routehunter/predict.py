"""
Predict module.

No CASP engine is actually wired into the app yet -- there's nothing
here that runs AiZynthFinder or SynPlanner itself. Instead: given a
SMILES, run the same pre-trained solvability models used by Search's
level-1 properties (see properties.py), and pair each tool's predicted
probability with a link to that tool, so the person can go run it
themselves on whichever target looks most promising.

Reuses whatever PropertyPredictorSet the app already loaded (from
rh_data/model/*.pickle) -- no separate model loading here.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .properties import PropertyPredictor

# Maps a property predictor's name (now already the tool's display
# name, e.g. "AiZynthFinder" -- see PropertyPredictorSet.load_from_dir)
# to a link. Add an entry here for any future solvability model
# alongside its own PropertyPredictor registration.
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
    smiles: str
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


def predict(smiles: str, property_predictors: list[PropertyPredictor]) -> PredictResult:
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

    return PredictResult(smiles=smiles, predictions=predictions)