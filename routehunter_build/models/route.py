from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .training import fit_and_evaluate

DEFAULT_PARAM_GRID = {
    "tfidf__max_features": [2000, 5000, 10000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.1, 1.0, 10.0],
}


def combine_text(title, abstract) -> str:
    title = (title or "").strip()
    abstract = "" if pd.isna(abstract) else str(abstract).strip()
    return f"{title}. {abstract}" if abstract else title


def load_route_data(
    csv_path: str,
    title_col: str = "title",
    abstract_col: str = "abstract",
    y_col: str = "has_route",
) -> tuple[list[str], np.ndarray]:
    """Expects y_col already standardized (0/1 integers) upstream --
    no format checking here."""
    df = pd.read_csv(csv_path)
    texts = [combine_text(t, a) for t, a in zip(df[title_col], df[abstract_col])]
    y = df[y_col].to_numpy()
    return texts, y


def train_route_model(
    X: list[str],
    y: np.ndarray,
    param_grid: Optional[dict] = None,
    random_state: int = 42,
    test_size: float = 0.2,
    validation_fraction: float = 0.2,
) -> tuple[Pipeline, dict]:
    """Fits Pipeline(TfidfVectorizer -> LogisticRegression). See
    training.fit_and_evaluate for the two-level split. Returns
    (model, metrics)."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)),
    ])
    grid = param_grid if param_grid is not None else DEFAULT_PARAM_GRID

    model, metrics = fit_and_evaluate(
        pipeline, grid, X, y,
        scoring="balanced_accuracy",
        stratify=True,
        test_size=test_size,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    return model, metrics