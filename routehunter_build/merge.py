
import argparse
from functools import reduce

import pandas as pd
from rdkit import Chem


def _canonicalize_table(csv_path: str, label: str) -> pd.DataFrame:
    """
    Loads a (smiles, is_solved) CSV and returns (inchikey, SMILES,
    <label>), one row per unique molecule. Rows with unparseable
    SMILES are dropped silently.
    """
    df = pd.read_csv(csv_path)

    inchikeys = []
    canonical_smiles = []
    keep_mask = []

    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        inchikey = Chem.MolToInchiKey(mol) if mol is not None else ""
        # An empty inchikey (e.g. RDKit parses the SMILES but InChI
        # generation fails, as happens with wildcard/dummy atoms) must
        # never be treated as a valid merge key -- unrelated molecules
        # with failed InChI generation would otherwise all collapse
        # onto the same blank-string row.
        keep_mask.append(mol is not None and bool(inchikey))
        if keep_mask[-1]:
            inchikeys.append(inchikey)
            canonical_smiles.append(Chem.MolToSmiles(mol))

    out = df[keep_mask].copy()
    out["inchikey"] = inchikeys
    out["SMILES"] = canonical_smiles

    # Two different input SMILES could canonicalize to the same
    # InChIKey (duplicate structure within one tool's own file) --
    # collapse to one row per molecule, solved=True if any row for
    # that molecule was solved.
    return (
        out.groupby("inchikey", as_index=False)
        .agg({"SMILES": "first", "is_solved": "any"})
        .rename(columns={"is_solved": label})
    )


def merge_tool_tables(tool_paths: dict[str, str], how: str = "outer") -> pd.DataFrame:
    """
    tool_paths: dict mapping tool name -> path to that tool's
    (smiles, is_solved) CSV, e.g. {"AZ": "az_data.csv", "SP": "sp_data.csv"}.
    The keys become the output table's tool columns.
    """
    per_tool = [_canonicalize_table(path, label=name) for name, path in tool_paths.items()]

    def _merge_two(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        merged = pd.merge(left, right, on="inchikey", how=how, suffixes=("_l", "_r"))
        merged["SMILES"] = merged["SMILES_l"].combine_first(merged["SMILES_r"])
        return merged.drop(columns=["SMILES_l", "SMILES_r"])

    merged = reduce(_merge_two, per_tool)

    tool_names = list(tool_paths.keys())
    return merged[["inchikey", "SMILES"] + tool_names]
