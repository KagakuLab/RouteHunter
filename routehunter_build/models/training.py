from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline


def fit_and_evaluate(
    pipeline: Pipeline,
    param_grid: dict,
    X, y,
    scoring: str,
    test_size: float = 0.2,
    validation_fraction: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
) -> tuple[Pipeline, dict]:
    """
    Two levels of split:
      1) outer train/test -- test is held out entirely, touched only
         for the final metric below
      2) inner single-split (ShuffleSplit) within the training data --
         used only for hyperparameter selection via GridSearchCV

    Returns the refit pipeline plus a metrics dict: the inner CV score
    (optimistic -- the same split picked the hyperparameters) and the
    outer held-out score (the honest one), both under `scoring`.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if stratify else None,
    )

    single_split = ShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=random_state)
    search = GridSearchCV(pipeline, param_grid=param_grid, scoring=scoring,
                           cv=single_split, refit=True, n_jobs=-1)
    search.fit(X_train, y_train)

    model = search.best_estimator_
    scorer = get_scorer(scoring)

    metrics = {
        "scoring": scoring,
        "best_params": search.best_params_,
        "cv_score": search.best_score_,
        "test_score": scorer(model, X_test, y_test),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return model, metrics
