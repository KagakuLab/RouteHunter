from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from rdkit import Chem
from rdkit.Chem import inchi as rdkit_inchi


class RouteHunterError(Exception):
    """Base exception for RouteHunter."""


class InvalidSMILESError(RouteHunterError):
    """Raised when a SMILES string cannot be parsed by RDKit."""


@dataclass(frozen=True)
class Canonicalized:
    """Result of canonicalizing a SMILES string."""
    input_smiles: str
    canonical_smiles: str
    inchikey: str


def canonicalize(smiles: str) -> Canonicalized:
    """
    Single source of truth for "what counts as the same molecule".

    Every module (Search, seed loading, Hunter) must route through
    this function before touching the store, so structural identity
    is defined in exactly one place.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InvalidSMILESError(f"Could not parse SMILES: {smiles!r}")

    canonical_smiles = Chem.MolToSmiles(mol)
    inchikey = rdkit_inchi.MolToInchiKey(mol)
    if not inchikey:
        raise InvalidSMILESError(f"Could not derive InChIKey for: {smiles!r}")

    return Canonicalized(
        input_smiles=smiles,
        canonical_smiles=canonical_smiles,
        inchikey=inchikey,
    )


@dataclass
class Target:
    """A molecule that RouteHunter tracks synthesis routes for."""
    inchikey: str
    canonical_smiles: str
    input_smiles: str  # first-seen input form, kept for reference/debugging
    paper_dois: list[str] = field(default_factory=list)

    @property
    def n_papers(self) -> int:
        return len(self.paper_dois)


class PaperSource(str, Enum):
    SEED = "seed"  # loaded from the CSV -- the only source in the static dataset


@dataclass
class PaperRecord:
    """A paper, optionally linked to one or more Targets."""
    doi: str
    title: str
    abstract: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    contributor: Optional[str] = None
    source: PaperSource = PaperSource.SEED
    target_inchikeys: list[str] = field(default_factory=list)