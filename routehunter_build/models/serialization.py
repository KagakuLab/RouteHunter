import pickle

from sklearn.pipeline import Pipeline


def save_model(pipeline: Pipeline, output_path: str) -> None:
    """
    Plain pickle, deliberately -- no cloudpickle.
    """
    with open(output_path, "wb") as f:
        pickle.dump(pipeline, f)


def load_model(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)