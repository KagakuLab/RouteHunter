from .store import TargetStore, CandidateStore

USAGE_TEXT = """\
RouteHunter: A system for the collection and distribution of reference information on chemical synthesis routes

  Search      : give a SMILES, get papers, static CASP-solved tool
                results, and predicted solvability for that molecule.
                
  Predict     : given a SMILES, get predicted solvability probability
                per CASP tool, with a link to that tool.
                
  Monitor     : browse recently published papers, ranked by predicted
                probability of containing a multi-step synthesis route
                (pre-scored offline; candidates for you to review and
                add to the CSV by hand).
"""


class ReviewEngine:
    def __init__(self, target_store: TargetStore, candidate_store: CandidateStore):
        self.target_store = target_store
        self.candidate_store = candidate_store

    def stats(self) -> dict:
        targets = self.target_store.targets
        papers = self.target_store.papers

        journal_counts: dict[str, int] = {}
        contributor_counts: dict[str, int] = {}
        for p in papers.values():
            if p.journal:
                journal_counts[p.journal] = journal_counts.get(p.journal, 0) + 1
            contributor = p.contributor or "(unspecified)"
            contributor_counts[contributor] = contributor_counts.get(contributor, 0) + 1

        n_multi_paper_targets = sum(1 for t in targets.values() if t.n_papers > 1)
        n_cached_casp_routes = 0

        return {
            "n_targets": len(targets),
            "n_papers": len(papers),
            "n_multi_paper_targets": n_multi_paper_targets,
            "n_cached_casp_routes": n_cached_casp_routes,
            "targets_by_journal": journal_counts,
            "targets_by_contributor": contributor_counts,
            "n_predicted_candidate_papers": self.candidate_store.n_papers,
        }

    def review(self) -> str:
        s = self.stats()
        lines = [USAGE_TEXT, "", "RouteHunter data review:"]
        lines.append(f"  Targets                    : {s['n_targets']}")
        lines.append(f"  Papers                     : {s['n_papers']}")
        lines.append(f"  Targets with >1 paper      : {s['n_multi_paper_targets']}")
        lines.append(f"  Cached CASP routes         : {s['n_cached_casp_routes']}")
        lines.append(f"  Predicted candidate papers : {s['n_predicted_candidate_papers']} (awaiting for digitalization)")

        if s["targets_by_journal"]:
            lines.append("  Papers by journal:")
            for journal, count in sorted(s["targets_by_journal"].items(), key=lambda kv: -kv[1]):
                lines.append(f"    {journal:<40} {count}")

        if s["targets_by_contributor"]:
            lines.append("  Papers by contributor:")
            for contributor, count in sorted(s["targets_by_contributor"].items(), key=lambda kv: -kv[1]):
                lines.append(f"    {contributor:<40} {count}")

        return "\n".join(lines)
