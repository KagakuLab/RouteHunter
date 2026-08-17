from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem

LINK_PLACEHOLDER = "Cached predicted routes are not available yet"


@dataclass
class CaspSolvedEntry:
    tool: str
    tool_display: str
    link: str


def _smiles_solved_map(path: str) -> dict[str, bool]:
    if not Path(path).exists():
        return {}

    df = pd.read_csv(path)
    result: dict[str, bool] = {}
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            continue
        inchikey = Chem.MolToInchiKey(mol)
        if not inchikey:
            continue
        result[inchikey] = result.get(inchikey, False) or bool(row["is_solved"])
    return result


def load_casp_table(tool_paths: dict[str, str]) -> dict[str, dict[str, bool]]:
    table: dict[str, dict[str, bool]] = {}
    for tool_name, path in tool_paths.items():
        for inchikey, solved in _smiles_solved_map(path).items():
            table.setdefault(inchikey, {})[tool_name] = solved
    return table


def solved_entries_for_inchikey(table: dict[str, dict[str, bool]], inchikey: str) -> list[CaspSolvedEntry]:
    flags = table.get(inchikey, {})
    entries = []
    for tool, solved in flags.items():
        if solved:
            entries.append(CaspSolvedEntry(tool=tool, tool_display=tool, link=LINK_PLACEHOLDER))
    return entries