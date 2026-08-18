from dataclasses import dataclass, field

import pandas as pd

from .store import PredictStore, AIZYNTHFINDER, SYNPLANNER

TOOL_URLS = {
    AIZYNTHFINDER: "https://github.com/MolecularAI/aizynthfinder",
    SYNPLANNER: "https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner",
}


@dataclass
class PredictResult:
    input_value: str  # the SMILES that was actually predicted on
    predictions: dict[str, float] = field(default_factory=dict)  # tool_name -> probability

    def to_dataframe(self) -> pd.DataFrame:
        """Probability rendered as a percentage string (e.g. '76%'),
        matching the rest of the app's display convention."""
        rows = []
        for tool_name, probability in self.predictions.items():
            rows.append({
                "tool": tool_name,
                "probability": f"{probability:.0%}" if probability is not None else "n/a",
                "url": TOOL_URLS.get(tool_name, ""),
            })
        return pd.DataFrame(rows)


class PredictEngine:
    def __init__(self, store: PredictStore):
        self.store = store

    def predict_casp_solvability(self, smiles: str) -> PredictResult:
        result = PredictResult(input_value=smiles, predictions=self.store.predict(smiles))
        return result
