"""
Contribute module.

Contribute no longer writes directly to the main store. It creates a
Submission -- (target SMILES, paper info, contributor name, optional
comment) -- and places it in a pending queue. An admin reviews it via
admin.review_submission(), which is the only path that publishes data
into the main store. This keeps the published dataset admin-curated
while still letting anyone submit.

Duplicate detection happens at submission time against two things:
    - the main store (already published -- same as before)
    - other PENDING submissions (someone already submitted this exact
      molecule+paper and it's awaiting review)
"""

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .core import PaperRecord, Submission, SubmissionStatus, canonicalize
from .store import RouteHunterStore


class SubmitStatus(str, Enum):
    SUBMITTED = "submitted"
    DUPLICATE_PUBLISHED = "duplicate_published"   # already in the main store
    DUPLICATE_PENDING = "duplicate_pending"       # already awaiting review


@dataclass
class SubmitResult:
    status: SubmitStatus
    submission: Optional[Submission]
    message: str


def submit_contribution(
    store: RouteHunterStore,
    smiles: str,
    paper: PaperRecord,
    contributor_name: str,
    contributor_comment: Optional[str] = None,
) -> SubmitResult:
    """
    Queue a (molecule, paper) contribution for admin review.

    Raises InvalidSMILESError if the SMILES cannot be parsed.
    """
    canon = canonicalize(smiles)  # raises InvalidSMILESError on bad input
    inchikey = canon.inchikey

    # Already published?
    existing_target = store.get_target(inchikey)
    if existing_target is not None and paper.doi in existing_target.paper_dois:
        return SubmitResult(
            status=SubmitStatus.DUPLICATE_PUBLISHED,
            submission=None,
            message=(
                f"Already registered: paper {paper.doi} is already published "
                f"for target {inchikey}."
            ),
        )

    # Already pending review?
    pending_dup = store.find_pending_duplicate(inchikey, paper.doi)
    if pending_dup is not None:
        return SubmitResult(
            status=SubmitStatus.DUPLICATE_PENDING,
            submission=pending_dup,
            message=(
                f"Already submitted: paper {paper.doi} for this target is "
                f"pending review (submission {pending_dup.submission_id})."
            ),
        )

    submission = Submission(
        submission_id=str(uuid.uuid4()),
        smiles=smiles,
        paper=paper,
        contributor_name=contributor_name,
        contributor_comment=contributor_comment,
        status=SubmissionStatus.PENDING,
    )
    store.add_submission(submission)

    return SubmitResult(
        status=SubmitStatus.SUBMITTED,
        submission=submission,
        message=(
            f"Submitted for review (submission {submission.submission_id}). "
            f"Thank you, {contributor_name}."
        ),
    )
