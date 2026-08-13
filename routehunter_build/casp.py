
import argparse
import pickle
from pathlib import Path
from typing import Optional

import cloudpickle
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score

# ---- training ------------------------------------------------------------
#
# Splitting is the caller's responsibility (see main()) -- this
# function only fits on whatever (X_train, y_train) it's given.

DEFAULT_PARAM_GRID = {
    "clf__n_estimators": [100, 300, 600],
    "clf__max_depth": [None, 8, 16, 32],
    "clf__min_samples_leaf": [1, 3, 5],
    "clf__max_features": ["sqrt", "log2"],
}

# ---- featurizer (RDKit Morgan/ECFP, no molfeat) ------------------------
class MorganFingerprintTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer: list of SMILES in, Morgan (ECFP)
    fingerprint bit array out. Lives inside the Pipeline that gets
    pickled, so the featurizer travels with the model -- no separate
    fingerprinting code needs to exist wherever the model is loaded.
    """

    def __init__(self, radius: int = 2, n_bits: int = 2048):
        self.radius = radius
        self.n_bits = n_bits

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)
        fps = np.zeros((len(X), self.n_bits), dtype=np.int8)
        for i, smi in enumerate(X):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Could not parse SMILES: {smi!r}")
            bit_vect = generator.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(bit_vect, fps[i])
        return fps


# ---- data loading ------------------------------------------------------

def load_casp_data(csv_path: str) -> tuple[list[str], np.ndarray]:
    """Expects exactly columns SMILES, IS_SOLVED, already standardized
    (0/1 integers) upstream -- no format checking here."""
    df = pd.read_csv(csv_path)
    smiles = df["smiles"].tolist()
    y = df["is_solved"].to_numpy()
    return smiles, y

def train_casp_model(
    X_train_smiles: list[str],
    y_train: np.ndarray,
    param_grid: Optional[dict] = None,
    radius: int = 2,
    n_bits: int = 2048,
    validation_fraction: float = 0.2,
    random_state: int = 42,
) -> Pipeline:
    """
    Fits Pipeline(featurizer -> RandomForestClassifier) on
    (X_train_smiles, y_train), tuning hyperparameters via GridSearchCV
    scored on a single train/validation split (ShuffleSplit,
    n_splits=1) rather than k-fold CV. Returns the fitted pipeline
    (refit on the full X_train_smiles with the best hyperparameters).
    """
    pipeline = Pipeline([
        ("featurizer", MorganFingerprintTransformer(radius=radius, n_bits=n_bits)),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=random_state)),
    ])

    grid = param_grid if param_grid is not None else DEFAULT_PARAM_GRID
    single_split = ShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=random_state)

    search = GridSearchCV(
        pipeline,
        param_grid=grid,
        scoring="balanced_accuracy",
        cv=single_split,
        refit=True,
        n_jobs=-1,
    )
    search.fit(X_train_smiles, y_train)

    return search.best_estimator_


def save_casp_model(pipeline: Pipeline, output_path: str) -> None:
    with open(output_path, "wb") as f:
        cloudpickle.dump(pipeline, f)
