import csv
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.base import BaseEstimator, TransformerMixin

HF_REPO_ID = "KagakuLab/RouteHunterData"
def download_app_data(to: str = ".", repo_id: str = HF_REPO_ID) -> str:
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=to,
        allow_patterns=["app_data/*"],
    )
    return str(Path(local_dir) / "app_data")


def load_config(rh_data_dir: str) -> dict[str, str]:

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