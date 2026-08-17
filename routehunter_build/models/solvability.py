from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from routehunter.featurizers import MorganFingerprintTransformer
from .training import fit_and_evaluate

# MorganFingerprintTransformer is imported from routehunter (the app
# package), not defined here. A pickled Pipeline containing this class
# needs it importable wherever the pickle is loaded -- the deployed
# app always has routehunter installed (by definition), but never
# routehunter_build (build-only tooling, not a runtime dependency).
# Defining the class in routehunter_build, even without cloudpickle,
# would mean every solvability model permanently requires
# routehunter_build importable at prediction time -- exactly what
# this avoids.

DEFAULT_PARAM_GRID = {
    "clf__n_estimators": [100, 300, 600],
    "clf__max_depth": [None, 8, 16, 32],
    "clf__min_samples_leaf": [1, 3, 5],
    "clf__max_features": ["sqrt", "log2"],
}


def load_solvability_data(
    csv_path: str,
    smiles_col: str = "smiles",
    y_col: str = "is_solved",
) -> tuple[list[str], np.ndarray]:
    """Expects y_col already standardized (0/1 integers) upstream --
    no format checking here."""
    df = pd.read_csv(csv_path)
    smiles = df[smiles_col].tolist()
    y = df[y_col].to_numpy()
    return smiles, y


def train_solvability_model(
    X: list[str],
    y: np.ndarray,
    param_grid: Optional[dict] = None,
    radius: int = 2,
    n_bits: int = 2048,
    random_state: int = 42,
    validation_fraction: float = 0.2,
) -> tuple[Pipeline, dict]:
    """Fits Pipeline(featurizer -> RandomForestClassifier). See
    training.fit_and_evaluate for the two-level split. Returns
    (model, metrics)."""
    pipeline = Pipeline([
        ("featurizer", MorganFingerprintTransformer(radius=radius, n_bits=n_bits)),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=random_state)),
    ])
    grid = param_grid if param_grid is not None else DEFAULT_PARAM_GRID

    model, metrics = fit_and_evaluate(
        pipeline, grid, X, y,
        scoring="balanced_accuracy",
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    return model, metrics