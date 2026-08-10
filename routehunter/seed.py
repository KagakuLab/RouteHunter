"""
Seed module.

Loads the app's dataset from a CSV -- this is the ONLY way data enters
RouteHunter. The app is static: there is no Contribute/admin pipeline,
no in-app write path, no persistence layer separate from the CSV.
Each CSV file represents one version of the dataset; to add or correct
data, edit the CSV and reload.

Rows are folded into the Target/PaperRecord graph: multiple rows for
the same molecule (different papers) become independent routes on one
Target; one paper appearing under multiple targets is linked to each;
an exact repeat of a (molecule, paper) pair is folded in without being
double-counted in the dataset (though it still counts toward the
targets total in the load report, since it was still a row in the
file).

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
    "contributor": "contributor",
}
REQUIRED_FIELDS = ("smiles", "doi", "title")


@dataclass
class SeedLoadError:
    row_number: int  # 1-indexed, matches spreadsheet row (header = row 1)
    raw_row: dict
    error: str


@dataclass
class SeedLoadReport:
    n_rows: int = 0                                  # total rows processed, including errors
    n_targets: int = 0                              # total target mentions (one per valid row)
    unique_target_inchikeys: set = field(default_factory=set)
    unique_paper_dois: set = field(default_factory=set)
    errors: list[SeedLoadError] = field(default_factory=list)

    @property
    def n_unique_targets(self) -> int:
        return len(self.unique_target_inchikeys)

    @property
    def n_unique_papers(self) -> int:
        return len(self.unique_paper_dois)

    def summary(self) -> str:
        lines = [
            f"Rows processed : {self.n_rows}",
            f"Targets        : {self.n_targets}",
            f"Unique targets : {self.n_unique_targets}",
            f"Unique papers  : {self.n_unique_papers}",
            f"Errors         : {len(self.errors)}",
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
    Load `csv_path` into `store`. Safe to call multiple times / with
    overlapping data -- rows already present (same molecule, same DOI)
    are folded into the existing Target/PaperRecord graph rather than
    duplicated, though every valid row still counts toward the
    targets total in the report.
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

            report.n_targets += 1
            report.unique_target_inchikeys.add(inchikey)
            report.unique_paper_dois.add(doi)

            target = store.get_target(inchikey)
            if target is None:
                target = Target(
                    inchikey=inchikey,
                    canonical_smiles=canon.canonical_smiles,
                    input_smiles=values["smiles"],
                )

            if doi not in target.paper_dois:
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
                        contributor=values.get("contributor") or None,
                        source=PaperSource.SEED,
                    )

                target.paper_dois.append(doi)
                if inchikey not in paper.target_inchikeys:
                    paper.target_inchikeys.append(inchikey)

                store.add_paper(paper)

            store.add_target(target)

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