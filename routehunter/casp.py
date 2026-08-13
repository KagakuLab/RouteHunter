from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# Short column name (as it appears in the CSV) -> display name shown
# to the person. Add an entry here for any future tool column.
TOOL_DISPLAY_NAMES: dict[str, str] = {
    "AZ": "AiZynthFinder",
    "SP": "SynPlanner",
}

# No real per-route link is stored yet -- just a placeholder pointing
# the person at the tool itself.
LINK_PLACEHOLDER = "Cached predicted routes are not available yet"


@dataclass
class CaspSolvedEntry:
    tool: str          # short code as it appears in the CSV, e.g. "AZ"
    tool_display: str  # e.g. "AiZynthFinder"
    link: str


def load_casp_table(path: str) -> dict[str, dict[str, bool]]:
    """
    Returns {inchikey: {tool_name: solved_bool, ...}, ...}. Tool
    columns are whatever's in the CSV besides inchikey/SMILES, so
    adding a tool later needs no code change here -- just another
    column in the file. A missing/NaN flag for a given tool means
    "not tested by that tool", not "not solved" -- it's simply
    omitted from that molecule's dict rather than stored as False.
    """
    if not Path(path).exists():
        return {}

    df = pd.read_csv(path)
    tool_columns = [c for c in df.columns if c not in ("inchikey", "SMILES")]

    table: dict[str, dict[str, bool]] = {}
    for _, row in df.iterrows():
        flags = {}
        for tool in tool_columns:
            value = row[tool]
            if pd.isna(value):
                continue
            flags[tool] = bool(value)
        table[row["inchikey"]] = flags

    return table


def solved_entries_for_inchikey(table: dict[str, dict[str, bool]], inchikey: str) -> list[CaspSolvedEntry]:
    """Only tools with solved=True produce an entry -- tools that
    didn't solve it (or weren't tested) are simply absent."""
    flags = table.get(inchikey, {})
    entries = []
    for tool, solved in flags.items():
        if solved:
            entries.append(CaspSolvedEntry(
                tool=tool,
                tool_display=TOOL_DISPLAY_NAMES.get(tool, tool),
                link=LINK_PLACEHOLDER,
            ))
    return entries