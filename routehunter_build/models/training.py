from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, ShuffleSplit, RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

CV_SPLITS = 5
CV_REPEATS = 5


def fit_and_evaluate(
    pipeline: Pipeline,
    param_grid: dict,
    X, y,
    scoring: str,
    validation_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, dict]:
    """
    Hyperparameters are selected via GridSearchCV on a single
    train/validation split (ShuffleSplit, n_splits=1) over the full
    data; the returned model is refit on all of (X, y) with those
    hyperparameters.

    Evaluation uses repeated CV_SPLITS-fold CV (CV_REPEATS repeats,
    5x5) rather than a single held-out split: every sample gets
    CV_REPEATS independent out-of-fold probability estimates
    (refitting the already-chosen hyperparameters fresh on each
    fold), averaged per sample. This pooled, whole-dataset (y_val,
    y_prob) is far more stable than a single small held-out split --
    both for the reported precision/recall/F1 (at the default 0.5
    decision boundary) and for picking a precision threshold
    afterward.
    """
    single_split = ShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=random_state)
    search = GridSearchCV(pipeline, param_grid=param_grid, scoring=scoring,
                           cv=single_split, refit=True, n_jobs=-1)
    search.fit(X, y)
    model = search.best_estimator_

    y_val, y_prob = _repeated_cv_probabilities(model, X, y, random_state=random_state)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "y_val": y_val,
        "y_prob": y_prob,
    }
    return model, metrics


def _repeated_cv_probabilities(model, X, y, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs repeated 5-fold CV (5 repeats) with the given (already
    hyperparameter-selected) model, refitting a fresh clone on each
    fold's training portion. Returns (y, y_prob), where y_prob is each
    sample's predicted probability averaged over the 5 out-of-fold
    estimates it received (one per repeat).
    """
    X = np.asarray(X, dtype=object)
    y = np.asarray(y)

    cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=random_state)

    prob_sums = np.zeros(len(y))
    prob_counts = np.zeros(len(y))

    for train_idx, test_idx in cv.split(X, y):
        fold_model = clone(model)
        fold_model.fit(list(X[train_idx]), y[train_idx])
        fold_probs = fold_model.predict_proba(list(X[test_idx]))[:, 1]
        prob_sums[test_idx] += fold_probs
        prob_counts[test_idx] += 1

    y_prob = prob_sums / prob_counts
    return y, y_prob


@dataclass
class ThresholdResult:
    threshold: float
    precision: float
    recall: float


def find_threshold_for_precision(y_true, y_prob, target_precision: float) -> ThresholdResult:
    """Smallest probability threshold that achieves >= target_precision
    on (y_true, y_prob). Raises if it's never reached."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    for precision, recall, threshold in zip(precisions, recalls, thresholds):
        if precision >= target_precision:
            return ThresholdResult(threshold=threshold, precision=precision, recall=recall)
    raise ValueError(f"No threshold reaches precision >= {target_precision}")