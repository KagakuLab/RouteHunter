import sys
from typing import Optional

import cloudpickle
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .training import fit_and_evaluate

# Force cloudpickle to embed this module's classes by value rather
# than by reference, regardless of whether this file ends up imported
# as "solvability", "__main__", or anything else. Without this, a
# saved model's self-containment depends on how training happened to
# be invoked -- with it, the pickle never needs this file to be
# importable wherever it's later loaded.
cloudpickle.register_pickle_by_value(sys.modules[__name__])

DEFAULT_PARAM_GRID = {
    "clf__n_estimators": [100, 300, 600],
    "clf__max_depth": [None, 8, 16, 32],
    "clf__min_samples_leaf": [1, 3, 5],
    "clf__max_features": ["sqrt", "log2"],
}


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