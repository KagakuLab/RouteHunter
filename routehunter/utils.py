import csv
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.base import BaseEstimator, TransformerMixin


def load_config(rh_data_dir: str) -> dict[str, str]:
    """
    Reads rh_data_dir/config.csv -- key,path,comment -- and returns
    {key: resolved_path}, with each path resolved relative to
    rh_data_dir. This is the only source of truth for file locations;
    there is no fallback to hardcoded defaults anywhere else in the
    package. Raises FileNotFoundError if config.csv itself, or a path
    it lists, doesn't exist.
    """
    base = Path(rh_data_dir)
    config_path = base / "config.csv"

    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found. RouteHunter requires a config.csv "
            f"manifest in the data directory listing every file path -- "
            f"there is no fallback to default paths."
        )

    paths = {}
    with config_path.open(newline="") as f:
        for row in csv.DictReader(f):
            paths[row["key"]] = str(base / row["path"])
    return paths


class MorganFingerprintTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer: list of SMILES in, Morgan (ECFP)
    fingerprint bit array out.
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
