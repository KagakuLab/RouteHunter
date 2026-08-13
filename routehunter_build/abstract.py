
import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score


# ---- text combination ---------------------------------------------------
#
# A plain function, not a class -- there's nothing here worth wrapping
# in a custom sklearn transformer (unlike the fingerprinting case).
# Call this exact function at inference time too, so the text handed
# to the model is formatted the same way it was during training.

def combine_text(title, abstract) -> str:
    title = (title or "").strip()
    abstract = "" if pd.isna(abstract) else str(abstract).strip()
    return f"{title}. {abstract}" if abstract else title


# ---- data loading ------------------------------------------------------

def load_abstract_data(csv_path: str) -> tuple[list[str], np.ndarray]:
    """Expects exactly columns TITLE, ABSTRACT, IS_ROUTE, already
    standardized (0/1 integers) upstream -- no format checking here."""
    df = pd.read_csv(csv_path)
    texts = [combine_text(t, a) for t, a in zip(df["title"], df["abstract"])]
    y = df["has_route"].to_numpy()
    return texts, y


# ---- training ------------------------------------------------------------
#
# Splitting is the caller's responsibility (see main()) -- this
# function only fits on whatever (X_train, y_train) it's given.

DEFAULT_PARAM_GRID = {
    "tfidf__max_features": [2000, 5000, 10000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.1, 1.0, 10.0],
}


def train_abstract_model(
    X_train_text: list[str],
    y_train: np.ndarray,
    param_grid: Optional[dict] = None,
    validation_fraction: float = 0.2,
    random_state: int = 42,
) -> Pipeline:
    """
    Fits Pipeline(TfidfVectorizer -> LogisticRegression) on
    (X_train_text, y_train), tuning hyperparameters via GridSearchCV
    scored on a single train/validation split (ShuffleSplit,
    n_splits=1) rather than k-fold CV. Returns the fitted pipeline
    (refit on the full X_train_text with the best hyperparameters).
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)),
    ])

    grid = param_grid if param_grid is not None else DEFAULT_PARAM_GRID
    single_split = ShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=random_state)

    search = GridSearchCV(
        pipeline,
        param_grid=grid,
        scoring="balanced_accuracy",
        cv=single_split,
        refit=True,
        n_jobs=-1,
    )
    search.fit(X_train_text, y_train)

    return search.best_estimator_


def save_abstract_model(pipeline: Pipeline, output_path: str) -> None:
    with open(output_path, "wb") as f:
        pickle.dump(pipeline, f)
