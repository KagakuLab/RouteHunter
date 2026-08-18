from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import CandidateStore


@dataclass
class CandidateResult:
    count: int


class CandidateEngine:
    def __init__(self, store: "CandidateStore"):
        self.store = store

    def get(self) -> CandidateResult:
        return CandidateResult(count=self.store.count())
