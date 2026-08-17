import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.base import BaseEstimator, TransformerMixin


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