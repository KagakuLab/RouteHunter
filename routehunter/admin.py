"""
Admin module.

The only path that publishes a Submission into the main store. This
is where the new-target / new-route / duplicate wiring logic lives
(previously in contribute.py, before contributions were routed through
a review queue).

Not modeled as authentication/authorization here -- in the prototype
"admin" just means "whoever calls review_submission". Add an actual
permission check when this grows a real front end.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .core import Target, canonicalize, SubmissionStatus
from .store import RouteHunterStore


class ReviewStatus(str, Enum):
    APPROVED_NEW_TARGET = "approved_new_target"
    APPROVED_NEW_ROUTE = "approved_new_route"
    REJECTED = "rejected"


@dataclass
class ReviewResult:
    status: ReviewStatus
    target: Optional[Target]
    message: str


def review_submission(
    store: RouteHunterStore,
    submission_id: str,
    approve: bool,
    admin_notes: Optional[str] = None,
) -> ReviewResult:
    """
    Approve or reject a pending submission.

    On approval: publishes the (target, paper) pair into the main
    store -- creating a new Target if the molecule is new, or adding
    an independent route if the molecule is already known -- and marks
    the paper's route_confirmed=True. On rejection: the submission is
    kept (status=REJECTED, with admin_notes) for audit purposes, and
    nothing is written to the main store.
    """
    submission = store.get_submission(submission_id)
    if submission is None:
        raise ValueError(f"No such submission: {submission_id}")
    if submission.status != SubmissionStatus.PENDING:
        raise ValueError(
            f"Submission {submission_id} was already reviewed "
            f"(status={submission.status.value})."
        )

    if not approve:
        submission.status = SubmissionStatus.REJECTED
        submission.admin_notes = admin_notes
        return ReviewResult(
            status=ReviewStatus.REJECTED,
            target=None,
            message=f"Submission {submission_id} rejected.",
        )

    canon = canonicalize(submission.smiles)
    inchikey = canon.inchikey

    existing_target = store.get_target(inchikey)
    is_new_target = existing_target is None

    if is_new_target:
        target = Target(
            inchikey=inchikey,
            canonical_smiles=canon.canonical_smiles,
            input_smiles=canon.input_smiles,
        )
    else:
        target = existing_target

    paper = submission.paper
    paper.route_confirmed = True

    if paper.doi not in target.paper_dois:
        target.paper_dois.append(paper.doi)
    if inchikey not in paper.target_inchikeys:
        paper.target_inchikeys.append(inchikey)

    store.add_target(target)
    store.add_paper(paper)

    submission.status = SubmissionStatus.APPROVED
    submission.admin_notes = admin_notes

    if is_new_target:
        return ReviewResult(
            status=ReviewStatus.APPROVED_NEW_TARGET,
            target=target,
            message=f"Approved. New target published ({inchikey}).",
        )
    else:
        return ReviewResult(
            status=ReviewStatus.APPROVED_NEW_ROUTE,
            target=target,
            message=(
                f"Approved. Independent route added for existing target "
                f"{inchikey}. This target now has {target.n_routes} "
                f"reported synthesis routes."
            ),
        )
