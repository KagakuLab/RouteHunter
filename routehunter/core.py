"""
Core data model for RouteHunter.

Two entities, related many-to-many:
    Target       <-- paper_dois / target_inchikeys -->      PaperRecord

A Target may have multiple independent syntheses (multiple papers).
A PaperRecord may report routes to multiple Targets.

Identity for a Target is its InChIKey (canonical), never the raw
input SMILES, so structurally-identical inputs always collapse to one
record. Identity for a PaperRecord is its DOI.

The app is static: the entire dataset comes from a CSV loaded via
seed.py at the start of a session (see seed.load_csv_seed). There is
no in-app write path that changes the dataset -- to add or correct
data, edit the CSV and reload. The only thing that changes at runtime
is the CASP cache (see CASPRouteRecord below), and that lives only in
memory for the session.
"""

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
    def n_routes(self) -> int:
        return len(self.paper_dois)


class PaperSource(str, Enum):
    SEED = "seed"      # loaded from the CSV -- the only source in the static dataset
    HUNTER = "hunter"  # classifier-predicted candidate; not part of the dataset,
                        # just a ranked suggestion for you to review and, if it
                        # checks out, add to the CSV by hand


@dataclass
class PaperRecord:
    """A paper, optionally linked to one or more Targets. Every
    PaperRecord in the dataset comes from the CSV (source=SEED) and is
    treated as reliable by construction -- there is no separate
    review/confirmation step in the static app."""
    doi: str
    title: str
    abstract: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    source: PaperSource = PaperSource.SEED
    hunter_score: Optional[float] = None
    target_inchikeys: list[str] = field(default_factory=list)


@dataclass
class RouteStep:
    product_smiles: str
    reactant_smiles: list[str]
    reaction_name: Optional[str] = None


@dataclass
class Route:
    target_smiles: str
    steps: list[RouteStep]
    score: Optional[float] = None  # engine-reported confidence, if available

    @property
    def n_steps(self) -> int:
        return len(self.steps)


@dataclass
class CASPRouteRecord:
    """A CASP-predicted route, cached in memory for the session so
    Search can surface it alongside literature papers without
    recomputing on every call. This cache is NOT part of the static
    dataset and is not persisted -- it disappears when the session
    ends. Cached independent of whether a literature Target exists for
    this molecule."""
    inchikey: str
    canonical_smiles: str
    route: Route
    engine_name: str
    score: Optional[float] = None