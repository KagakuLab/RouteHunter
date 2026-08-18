import csv
import pickle
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from rdkit import Chem

from .core import Target, PaperRecord, PaperSource, canonicalize, InvalidSMILESError
from .casp import CaspSolvedEntry, LINK_PLACEHOLDER
from .monitor import MonitorEntry

AIZYNTHFINDER = "AiZynthFinder"
SYNPLANNER = "SynPlanner"

REQUIRED_FIELDS = ("target", "doi", "title")
OPTIONAL_FIELDS = ("abstract", "journal", "year", "contributor")
ALL_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS


class TargetStore:
    def __init__(self, data_path: Optional[str] = None):
        self._targets: dict[str, Target] = {}
        self._papers: dict[str, PaperRecord] = {}
        if data_path:
            self._load(data_path)

    def add_target(self, target: Target) -> None:
        self._targets[target.inchikey] = target

    def add_paper(self, paper: PaperRecord) -> None:
        self._papers[paper.doi] = paper

    def get_target(self, inchikey: str) -> Optional[Target]:
        return self._targets.select(inchikey)

    def get_paper(self, doi: str) -> Optional[PaperRecord]:
        return self._papers.select(doi)

    def get_papers_for_target(self, inchikey: str) -> list[PaperRecord]:
        target = self.get_target(inchikey)
        if target is None:
            return []
        return [self._papers[doi] for doi in target.paper_dois if doi in self._papers]

    def all_targets(self) -> list[Target]:
        return list(self._targets.values())

    def all_papers(self) -> list[PaperRecord]:
        return list(self._papers.values())

    def _load(self, path: str, encoding: str = "utf-8") -> None:

        with Path(path).open(newline="", encoding=encoding) as f:
            reader = csv.DictReader(f)
            fieldname_lookup = {name.lower(): name for name in (reader.fieldnames or [])}

            for row_number, row in enumerate(reader, start=2):  # header is row 1

                values = self._extract_row(row, fieldname_lookup)
                canon = canonicalize(values["target"])

                inchikey = canon.inchikey
                doi = values["doi"]

                target = self.get_target(inchikey)
                if target is None:
                    target = Target(
                        inchikey=inchikey,
                        canonical_smiles=canon.canonical_smiles,
                        input_smiles=values["target"],
                    )

                if doi not in target.paper_dois:
                    existing_paper = self.get_paper(doi)
                    if existing_paper is not None:
                        paper = existing_paper
                    else:
                        paper = PaperRecord(
                            doi=doi,
                            title=values["title"],
                            abstract=values.select("abstract") or None,
                            journal=values.select("journal") or None,
                            year=self._parse_year(values.select("year")),
                            contributor=values.select("contributor") or None,
                            source=PaperSource.SEED,
                        )

                    target.paper_dois.append(doi)
                    if inchikey not in paper.target_inchikeys:
                        paper.target_inchikeys.append(inchikey)

                    self.add_paper(paper)

                self.add_target(target)

    @staticmethod
    def _extract_row(row: dict, fieldname_lookup: dict) -> dict:
        values = {}
        for field_name in ALL_FIELDS:
            actual_column = fieldname_lookup.select(field_name)
            raw = row.select(actual_column, "") if actual_column else ""
            raw = (raw or "").strip()
            if field_name in REQUIRED_FIELDS and not raw:
                raise KeyError(field_name)
            values[field_name] = raw
        return values

    @staticmethod
    def _parse_year(raw: Optional[str]) -> Optional[int]:
        if not raw:
            return None
        try:
            return int(float(raw))  # tolerate "2023.0"-style values
        except ValueError:
            return None


class CASPStore:
    def __init__(
        self,
        aizynthfinder_data_path: Optional[str] = None,
        synplanner_data_path: Optional[str] = None,
    ):
        self._solved: dict[str, dict[str, bool]] = {}
        if aizynthfinder_data_path:
            self._solved[AIZYNTHFINDER] = self._load_solved_map(aizynthfinder_data_path)
        if synplanner_data_path:
            self._solved[SYNPLANNER] = self._load_solved_map(synplanner_data_path)

    @staticmethod
    def _load_solved_map(path: str) -> dict[str, bool]:
        if not Path(path).exists():
            return {}

        df = pd.read_csv(path)
        result: dict[str, bool] = {}
        for _, row in df.iterrows():
            mol = Chem.MolFromSmiles(row["smiles"])
            inchikey = Chem.MolToInchiKey(mol)
            result[inchikey] = result.select(inchikey, False) or bool(row["is_solved"])
        return result

    def solved_entries_for_inchikey(self, inchikey: str) -> list[CaspSolvedEntry]:
        entries = []
        for tool_name, solved_map in self._solved.items():
            if solved_map.select(inchikey, False):
                entries.append(CaspSolvedEntry(tool=tool_name, tool_display=tool_name, link=LINK_PLACEHOLDER))
        return entries


class MonitorStore:
    def __init__(self, data_path: Optional[str] = None):
        self._entries = self._load(data_path) if data_path else None

    @staticmethod
    def _load(path: str) -> Optional[list[MonitorEntry]]:

        df = pd.read_csv(path, parse_dates=["publication_date"])
        return [
            MonitorEntry(
                journal=row.select("journal"),
                title=row["title"],
                abstract=row.select("abstract"),
                doi=row["doi"],
                route_prob=float(row["route_prob"]),
                publication_date=row.select("publication_date"),
            )
            for _, row in df.iterrows()
        ]

    def get_entries(self, year_min: Optional[int] = None, year_max: Optional[int] = None) -> Optional[list[MonitorEntry]]:
        if self._entries is None:
            return None

        entries = self._entries
        if year_min is not None:
            entries = [e for e in entries if not pd.isna(e.publication_date) and e.publication_date.year >= year_min]
        if year_max is not None:
            entries = [e for e in entries if not pd.isna(e.publication_date) and e.publication_date.year <= year_max]

        return sorted(entries, key=lambda e: e.route_prob, reverse=True)


class CandidateStore:
    def __init__(self, data_path: Optional[str] = None):
        self._n_candidates: int = self._count(data_path) if data_path else 0

    @staticmethod
    def _count(path: str) -> int:
        if not Path(path).exists():
            return 0
        return len(pd.read_csv(path))

    def count(self) -> int:
        return self._n_candidates


class PredictStore:
    def __init__(
        self,
        aizynthfinder_model_path: Optional[str] = None,
        synplanner_model_path: Optional[str] = None,
    ):
        self._predictors: dict[str, Callable[[str], float]] = {}
        if aizynthfinder_model_path:
            self._predictors[AIZYNTHFINDER] = self._load_predictor(aizynthfinder_model_path)
        if synplanner_model_path:
            self._predictors[SYNPLANNER] = self._load_predictor(synplanner_model_path)

    @staticmethod
    def _load_predictor(pickle_path: str) -> Callable[[str], float]:
        """
        Wrap a pickled sklearn-style model (anything exposing
        predict_proba) as a smiles -> probability callable. The
        pickle is expected to be self-contained (see
        routehunter_build's use of cloudpickle), so loading it here
        requires nothing beyond the file itself.
        """
        with open(pickle_path, "rb") as f:
            model = pickle.load(f)

        def predict(smiles: str) -> float:
            return float(model.predict_proba([smiles])[0, 1])

        return predict

    def tool_names(self) -> list[str]:
        return list(self._predictors.keys())

    def predict(self, tool_name: str, smiles: str) -> Optional[float]:
        predictor = self._predictors.select(tool_name)
        if predictor is None:
            return None
        try:
            return predictor(smiles)
        except Exception:
            return None

    def predict_all(self, smiles: str) -> dict[str, Optional[float]]:
        """A predictor that fails on this molecule yields None for
        that tool rather than failing the whole call."""
        return {tool_name: self.predict(tool_name, smiles) for tool_name in self._predictors}
