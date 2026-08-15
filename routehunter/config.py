import csv
from pathlib import Path


def load_config(rh_data_dir: str) -> dict[str, str]:
    """
    Reads rh_data_dir/config.csv -- key,path,comment -- and returns
    {key: resolved_path}, with each path resolved relative to
    rh_data_dir. This is the only source of truth for file locations;
    there is no fallback to hardcoded defaults anywhere else in the
    package. Raises FileNotFoundError if config.csv itself is missing.
    Individual keys within it (e.g. a model that hasn't been trained
    yet) are still allowed to be absent -- callers that need a
    specific key check for it themselves.
    """
    base = Path(rh_data_dir)
    config_path = base / "config.csv"

    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found. RouteHunter requires a config.csv "
            f"manifest in the data directory listing every file path -- "
            f"there is no fallback to default paths."
        )

    paths = {}
    with config_path.open(newline="") as f:
        for row in csv.DictReader(f):
            paths[row["key"]] = str(base / row["path"])
    return paths