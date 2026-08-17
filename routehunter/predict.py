"""
Predict module.

No CASP engine is actually wired into the app -- there's nothing here
that runs AiZynthFinder or SynPlanner itself. Two separate
predictions live here:

    predict_casp_solvability(smiles) -- runs the same pre-trained
    solvability models used by Search's level-1 properties (see
    properties.py), pairing each tool's predicted probability with a
    link to that tool.

    predict_route_probability(text) -- runs the route classifier
    directly. Not used by the GUI; exists for batch use via the
    Python interface.

Both return the same PredictResult shape for consistency, even though
predict_route_probability only ever has one "tool" (the classifier
itself, no link). This module holds no knowledge of which tools exist
or where they link to -- that's app.py's TOOLS registry; the tool
links are passed in per call.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .properties import PropertyPredictor


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


def predict_casp_solvability(
    smiles: str,
    property_predictors: list[PropertyPredictor],
    tool_links: dict[str, str],
) -> PredictResult:
    """
    Run every registered solvability predictor on `smiles` and pair
    each result with its tool's link (tool_links: {display_name: url},
    see app.py's TOOLS registry). A predictor that fails on this
    molecule yields probability=None for that tool rather than
    failing the whole call.
    """
    predictions = []
    for predictor in property_predictors:
        url = tool_links.get(predictor.name, "")
        try:
            probability = predictor.predict(smiles)
        except Exception:
            probability = None
        predictions.append(ToolPrediction(tool_name=predictor.name, probability=probability, url=url))

    return PredictResult(input_value=smiles, predictions=predictions)


def predict_route_probability(text: str, route_model) -> PredictResult:
    """
    Run the route classifier on `text` -- already-combined
    title+abstract, see routehunter_build.models.combine_text for the
    expected format. route_model=None (e.g. the file wasn't present
    in config.csv) yields probability=None rather than raising.
    """
    if route_model is None:
        probability = None
    else:
        try:
            probability = route_model.predict_proba([text])[0, 1]
        except Exception:
            probability = None

    prediction = ToolPrediction(tool_name="Paper classifier", probability=probability, url="")
    return PredictResult(input_value=text, predictions=[prediction])