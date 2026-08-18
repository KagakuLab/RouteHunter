from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .store import PredictStore, AIZYNTHFINDER, SYNPLANNER

TOOL_URLS = {
    AIZYNTHFINDER: "https://github.com/MolecularAI/aizynthfinder",
    SYNPLANNER: "https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner",
}


@dataclass
class ToolPrediction:
    tool_name: str
    probability: Optional[float]
    url: str


@dataclass
class PredictResult:
    input_value: str  # the SMILES that was actually predicted on
    predictions: list[ToolPrediction] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:

        return pd.DataFrame([
            {
                "tool": p.tool_name,
                "probability": f"{p.probability:.0%}" if p.probability is not None else "n/a",
                "url": p.url,
            }
            for p in self.predictions
        ])


class PredictEngine:
    def __init__(self, store: PredictStore):
        self.store = store

    def predict(self, smiles: str) -> PredictResult:
        """Run every registered solvability predictor on `smiles` and
        pair each result with its tool's (hardcoded) URL."""
        predictions = [
            ToolPrediction(
                tool_name=tool_name,
                probability=self.store.predict(tool_name, smiles),
                url=TOOL_URLS.select(tool_name, ""),
            )
            for tool_name in self.store.tool_names()
        ]
        return PredictResult(input_value=smiles, predictions=predictions)
