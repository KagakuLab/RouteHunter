import pickle
from typing import Optional

import pandas as pd
from rdkit import Chem

from .core import Target, Paper, canonicalize

AIZYNTHFINDER = "AiZynthFinder"
SYNPLANNER = "SynPlanner"


class TargetStore:
    def __init__(self, data_path: str):
        self.targets: dict[str, Target] = {}
        self.papers: dict[str, Paper] = {}
        self._parse(self._read(data_path))

    def get_papers_for_target(self, inchikey: str) -> list[Paper]:
        target = self.targets.get(inchikey)
        if target is None:
            return []
        result = []
        for doi in target.paper_dois:
            if doi in self.papers:
                result.append(self.papers[doi])
        return result

    @staticmethod
    def _read(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def _parse(self, df: pd.DataFrame) -> None:

        for _, row in df.iterrows():
            canon = canonicalize(row["target"])
            inchikey = canon.inchikey
            doi = row["doi"]

            target = self.targets.get(inchikey)
            if target is None:
                target = Target(
                    inchikey=inchikey,
                    canonical_smiles=canon.canonical_smiles,
                    input_smiles=row["target"],
                )

            if doi not in target.paper_dois:
                existing_paper = self.papers.get(doi)
                if existing_paper is not None:
                    # Same paper already known (e.g. reports a route to a
                    # different target too) -- reuse it, just add the link.
                    paper = existing_paper
                else:
                    abstract = row.get("abstract")
                    journal = row.get("journal")
                    year = row.get("year")
                    contributor = row.get("contributor")
                    paper = Paper(
                        doi=doi,
                        title=row["title"],
                        abstract=None if pd.isna(abstract) else abstract,
                        journal=None if pd.isna(journal) else journal,
                        year=None if pd.isna(year) else int(year),
                        contributor=None if pd.isna(contributor) else contributor,
                    )

                target.paper_dois.append(doi)
                if inchikey not in paper.target_inchikeys:
                    paper.target_inchikeys.append(inchikey)

                self.papers[paper.doi] = paper

            self.targets[target.inchikey] = target


class ToolStore:
    def __init__(self, aizynthfinder_data_path: str, synplanner_data_path: str):
        self.solved: dict[str, dict[str, bool]] = {
            AIZYNTHFINDER: self._parse(self._read(aizynthfinder_data_path)),
            SYNPLANNER: self._parse(self._read(synplanner_data_path)),
        }

    @staticmethod
    def _read(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    @staticmethod
    def _parse(df: pd.DataFrame) -> dict[str, bool]:
        result: dict[str, bool] = {}
        for _, row in df.iterrows():
            mol = Chem.MolFromSmiles(row["smiles"])
            inchikey = Chem.MolToInchiKey(mol)
            result[inchikey] = result.get(inchikey, False) or bool(row["is_solved"])
        return result

    def get_tools_for_target(self, inchikey: str) -> list[str]:
        result = []
        for tool_name, solved_map in self.solved.items():
            if solved_map.get(inchikey, False):
                result.append(tool_name)
        return result


class MonitorStore:
    def __init__(self, data_path: str):
        self.data: pd.DataFrame = self._read(data_path)

    @staticmethod
    def _read(path: str) -> pd.DataFrame:
        return pd.read_csv(path, parse_dates=["publication_date"])

    def get_papers_by_year(self, year_min: Optional[int] = None, year_max: Optional[int] = None) -> pd.DataFrame:
        data = self.data

        if year_min is not None:
            data = data[data["publication_date"].dt.year >= year_min]
        if year_max is not None:
            data = data[data["publication_date"].dt.year <= year_max]

        data = data.sort_values("route_prob", ascending=False)
        return data


class CandidateStore:
    def __init__(self, data_path: str):
        self.n_papers: int = len(self._read(data_path))

    @staticmethod
    def _read(path: str) -> pd.DataFrame:
        return pd.read_csv(path)


class PredictStore:
    def __init__(self, aizynthfinder_model_path: str, synplanner_model_path: str):
        self.models: dict[str, object] = {
            AIZYNTHFINDER: self._load(aizynthfinder_model_path),
            SYNPLANNER: self._load(synplanner_model_path),
        }

    @staticmethod
    def _load(pickle_path: str) -> object:

        with open(pickle_path, "rb") as f:
            return pickle.load(f)