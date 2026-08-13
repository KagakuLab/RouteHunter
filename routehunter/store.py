from typing import Optional

from .core import Target, PaperRecord, CASPRouteRecord
from .casp import CaspSolvedEntry, solved_entries_for_inchikey


class RouteHunterStore:
    def __init__(self):
        self._targets: dict[str, Target] = {}       # inchikey -> Target
        self._papers: dict[str, PaperRecord] = {}    # doi -> PaperRecord
        self._casp_routes: dict[str, list[CASPRouteRecord]] = {}  # inchikey -> [CASPRouteRecord], in-memory session cache only
        self._casp_table: dict[str, dict[str, bool]] = {}  # inchikey -> {tool_name: solved_bool}, static, loaded from CSV
        self._n_predicted_targets: int = 0  # row count of the medium-confidence Monitor file, static, loaded from CSV

    # ---- writes (used only by seed.load_csv_seed) ---------------------

    def add_target(self, target: Target) -> None:
        self._targets[target.inchikey] = target

    def add_paper(self, paper: PaperRecord) -> None:
        self._papers[paper.doi] = paper

    # ---- reads ---------------------------------------------------------

    def get_target(self, inchikey: str) -> Optional[Target]:
        return self._targets.get(inchikey)

    def get_paper(self, doi: str) -> Optional[PaperRecord]:
        return self._papers.get(doi)

    def get_papers_for_target(self, inchikey: str) -> list[PaperRecord]:
        target = self.get_target(inchikey)
        if target is None:
            return []
        return [self._papers[doi] for doi in target.paper_dois if doi in self._papers]

    def all_targets(self) -> list[Target]:
        return list(self._targets.values())

    def all_papers(self) -> list[PaperRecord]:
        return list(self._papers.values())

    # ---- session-only CASP cache (runtime engine calls, casp.py) ------

    def add_casp_route(self, record: CASPRouteRecord) -> None:
        self._casp_routes.setdefault(record.inchikey, []).append(record)

    def get_casp_routes_for_target(self, inchikey: str) -> list[CASPRouteRecord]:
        return self._casp_routes.get(inchikey, [])

    # ---- static CASP solved table (offline, routehunter_casp.csv) -----

    def set_casp_table(self, table: dict[str, dict[str, bool]]) -> None:
        self._casp_table = table

    def get_casp_solved_entries(self, inchikey: str) -> list[CaspSolvedEntry]:
        return solved_entries_for_inchikey(self._casp_table, inchikey)

    # ---- predicted-targets count (offline, paper_route_prob_medium.csv) --

    def set_n_predicted_targets(self, n: int) -> None:
        self._n_predicted_targets = n

    # ---- aggregate stats (feeds the Introduction module) ------------

    def stats(self) -> dict:
        papers = self.all_papers()
        targets = self.all_targets()

        journal_counts: dict[str, int] = {}
        contributor_counts: dict[str, int] = {}
        for p in papers:
            if p.journal:
                journal_counts[p.journal] = journal_counts.get(p.journal, 0) + 1
            contributor = p.contributor or "(unspecified)"
            contributor_counts[contributor] = contributor_counts.get(contributor, 0) + 1

        n_multi_paper_targets = sum(1 for t in targets if t.n_routes > 1)
        n_cached_casp_routes = sum(len(v) for v in self._casp_routes.values())

        return {
            "n_targets": len(targets),
            "n_papers": len(papers),
            "n_multi_paper_targets": n_multi_paper_targets,
            "n_cached_casp_routes": n_cached_casp_routes,
            "n_predicted_targets": self._n_predicted_targets,
            "targets_by_journal": journal_counts,
            "targets_by_contributor": contributor_counts,
        }