"""
Download module.

Exports the dataset as a flat table, one row per (target, paper) pair
-- the natural unit for ML training, since it's exactly a (SMILES,
title, abstract, label) row.

Note: only CSV-loaded (SEED) data is ever linked into a Target's
paper_dois, so every row exported here comes from the static CSV.
"""

from typing import Optional
import pandas as pd

from .core import PaperSource
from .store import RouteHunterStore


def download(
    store: RouteHunterStore,
    journals: Optional[list[str]] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    source: Optional[PaperSource] = None,
) -> pd.DataFrame:
    """
    Return a flat DataFrame, one row per (target, paper) link, with
    columns: inchikey, canonical_smiles, doi, title, abstract, journal,
    year, source.
    """
    rows = []
    for target in store.all_targets():
        for doi in target.paper_dois:
            paper = store.get_paper(doi)
            if paper is None:
                continue
            if journals and paper.journal not in journals:
                continue
            if year_min and (paper.year is None or paper.year < year_min):
                continue
            if year_max and (paper.year is None or paper.year > year_max):
                continue
            if source and paper.source != source:
                continue

            rows.append({
                "inchikey": target.inchikey,
                "canonical_smiles": target.canonical_smiles,
                "doi": paper.doi,
                "title": paper.title,
                "abstract": paper.abstract,
                "journal": paper.journal,
                "year": paper.year,
                "source": paper.source.value,
            })

    return pd.DataFrame(rows)