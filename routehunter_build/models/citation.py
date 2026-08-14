from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from .featurizers import MorganFingerprintTransformer
from .training import fit_and_evaluate

DEFAULT_PARAM_GRID = {
    "reg__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
}


def load_citation_data(
    csv_path: str,
    smiles_col: str = "smiles",
    y_col: str = "avg_cit_per_year",
) -> tuple[list[str], np.ndarray]:
    """Expects y_col already standardized upstream -- no format
    checking here."""
    df = pd.read_csv(csv_path)
    smiles = df[smiles_col].tolist()
    y = df[y_col].to_numpy()
    return smiles, y


def train_citation_model(
    X: list[str],
    y: np.ndarray,
    param_grid: Optional[dict] = None,
    radius: int = 2,
    n_bits: int = 2048,
    random_state: int = 42,
    test_size: float = 0.2,
    validation_fraction: float = 0.2,
) -> tuple[Pipeline, dict]:
    """Fits Pipeline(featurizer -> Ridge). See training.fit_and_evaluate
    for the two-level split. Returns (model, metrics)."""
    pipeline = Pipeline([
        ("featurizer", MorganFingerprintTransformer(radius=radius, n_bits=n_bits)),
        ("reg", Ridge()),
    ])
    grid = param_grid if param_grid is not None else DEFAULT_PARAM_GRID

    model, metrics = fit_and_evaluate(
        pipeline, grid, X, y,
        scoring="r2",
        stratify=False,
        test_size=test_size,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    return model, metrics