"""
Seed module.

Loads the app's dataset from a CSV -- this is the ONLY way data enters
RouteHunter. The app is static: there is no Contribute/admin pipeline,
no in-app write path, no persistence layer separate from the CSV.
Each CSV file represents one version of the dataset; to add or correct
data, edit the CSV and reload.

Rows are folded into the Target/PaperRecord graph: same molecule + new
paper -> independent route; same molecule + same paper (duplicate row)
-> skipped, not re-added; one paper appearing under multiple targets ->
correctly linked to each.

Expected CSV columns (case-insensitive), with sensible defaults but
fully remappable via `column_map` if your file uses different names:

    smiles      (required)  structure of the target
    doi         (required)  paper identifier
    title       (required)
    abstract    (optional)
    journal     (optional)
    year        (optional)

Rows with unparseable SMILES or missing required fields are recorded
in the report's `errors` list (with the 1-indexed row number) rather
than aborting the whole load.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .core import Target, PaperRecord, PaperSource, canonicalize, InvalidSMILESError
from .store import RouteHunterStore

DEFAULT_COLUMN_MAP = {
    "smiles": "smiles",
    "doi": "doi",
    "title": "title",
    "abstract": "abstract",
    "journal": "journal",
    "year": "year",
}
REQUIRED_FIELDS = ("smiles", "doi", "title")


@dataclass
class SeedLoadError:
    row_number: int  # 1-indexed, matches spreadsheet row (header = row 1)
    raw_row: dict
    error: str


@dataclass
class SeedLoadReport:
    n_rows: int = 0
    n_new_targets: int = 0
    n_new_routes: int = 0          # existing target, new paper
    n_duplicate_rows: int = 0      # exact (target, paper) already present
    errors: list[SeedLoadError] = field(default_factory=list)

    @property
    def n_loaded(self) -> int:
        return self.n_new_targets + self.n_new_routes

    def summary(self) -> str:
        lines = [
            f"Rows processed        : {self.n_rows}",
            f"Loaded (new targets)  : {self.n_new_targets}",
            f"Loaded (new routes)   : {self.n_new_routes}",
            f"Skipped (duplicates)  : {self.n_duplicate_rows}",
            f"Errors                : {len(self.errors)}",
        ]
        if self.errors:
            lines.append("")
            lines.append("First few errors:")
            for e in self.errors[:10]:
                lines.append(f"  row {e.row_number}: {e.error}")
        return "\n".join(lines)


def load_csv_seed(
    store: RouteHunterStore,
    csv_path: str,
    column_map: Optional[dict] = None,
    encoding: str = "utf-8",
) -> SeedLoadReport:
    """
    Load `csv_path` into `store` as published, admin-review-free SEED
    data. Safe to call multiple times / with overlapping data -- rows
    already present (same molecule, same DOI) are skipped rather than
    duplicated.
    """
    cmap = {**DEFAULT_COLUMN_MAP, **(column_map or {})}
    report = SeedLoadReport()

    path = Path(csv_path)
    with path.open(newline="", encoding=encoding) as f:
        reader = csv.DictReader(f)
        # normalize header lookup to be case-insensitive
        fieldname_lookup = {name.lower(): name for name in (reader.fieldnames or [])}

        for row_number, row in enumerate(reader, start=2):  # header is row 1
            report.n_rows += 1

            try:
                values = _extract_row(row, cmap, fieldname_lookup)
            except KeyError as e:
                report.errors.append(SeedLoadError(row_number, row, f"missing required field: {e}"))
                continue

            try:
                canon = canonicalize(values["smiles"])
            except InvalidSMILESError as e:
                report.errors.append(SeedLoadError(row_number, row, str(e)))
                continue

            inchikey = canon.inchikey
            doi = values["doi"]

            existing_target = store.get_target(inchikey)
            is_new_target = existing_target is None

            if is_new_target:
                target = Target(
                    inchikey=inchikey,
                    canonical_smiles=canon.canonical_smiles,
                    input_smiles=values["smiles"],
                )
            else:
                target = existing_target

            if doi in target.paper_dois:
                report.n_duplicate_rows += 1
                continue

            existing_paper = store.get_paper(doi)
            if existing_paper is not None:
                # Same paper already known (e.g. reports a route to a
                # different target too) -- reuse it, just add the link.
                paper = existing_paper
            else:
                paper = PaperRecord(
                    doi=doi,
                    title=values["title"],
                    abstract=values.get("abstract") or None,
                    journal=values.get("journal") or None,
                    year=_parse_year(values.get("year")),
                    source=PaperSource.SEED,
                )

            target.paper_dois.append(doi)
            if inchikey not in paper.target_inchikeys:
                paper.target_inchikeys.append(inchikey)

            store.add_target(target)
            store.add_paper(paper)

            if is_new_target:
                report.n_new_targets += 1
            else:
                report.n_new_routes += 1

    return report


def _extract_row(row: dict, cmap: dict, fieldname_lookup: dict) -> dict:
    values = {}
    for logical_name, csv_column in cmap.items():
        actual_column = fieldname_lookup.get(csv_column.lower())
        raw = row.get(actual_column, "") if actual_column else ""
        raw = (raw or "").strip()
        if logical_name in REQUIRED_FIELDS and not raw:
            raise KeyError(logical_name)
        values[logical_name] = raw
    return values


def _parse_year(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    try:
        return int(float(raw))  # tolerate "2023.0"-style values
    except ValueError:
        return None