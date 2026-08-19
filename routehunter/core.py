from dataclasses import dataclass, field
from typing import Optional

from rdkit import Chem
from rdkit.Chem import inchi as rdkit_inchi


class InvalidSMILESError(Exception):
    pass


@dataclass(frozen=True)
class Canonicalized:
    input_smiles: str
    canonical_smiles: str
    inchikey: str


@dataclass
class Target:
    inchikey: str
    canonical_smiles: str
    input_smiles: str  # first-seen input form, kept for reference/debugging
    paper_dois: list[str] = field(default_factory=list)

    @property
    def n_papers(self) -> int:
        return len(self.paper_dois)


@dataclass
class Paper:
    doi: str
    title: str
    abstract: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    contributor: Optional[str] = None
    target_inchikeys: list[str] = field(default_factory=list)


def canonicalize(smiles: str) -> Canonicalized:

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InvalidSMILESError(f"Could not parse SMILES: {smiles!r}")

    canonical_smiles = Chem.MolToSmiles(mol)
    inchikey = rdkit_inchi.MolToInchiKey(mol)
    if not inchikey:
        raise InvalidSMILESError(f"Could not derive InChIKey for: {smiles!r}")

    result = Canonicalized(
        input_smiles=smiles,
        canonical_smiles=canonical_smiles,
        inchikey=inchikey,
    )
    return result
