import json
import os
import pathlib
import pickle

from sklearn.calibration import CalibratedClassifierCV


def load_model() -> CalibratedClassifierCV:
    model_path: pathlib.Path = pathlib.Path(os.getenv("MODEL_PATH"))

    if not model_path.exists():
        raise FileNotFoundError(f"model path does not exist at path: {model_path}")

    if not model_path.is_file():
        raise FileNotFoundError(f"model path is not a file: {model_path}")

    with model_path.open("rb") as f:
        classifier = pickle.load(f)
    return classifier


def load_columns() -> list[str]:
    columns_path: pathlib.Path = pathlib.Path(os.getenv("COLUMNS_PATH"))
    if not columns_path.exists():
        raise FileNotFoundError(f"{columns_path} does not exists")

    if not columns_path.is_file():
        raise FileNotFoundError(f"{columns_path} does not exists or is not a file")

    with columns_path.open("r") as f:
        columns: list[str] = json.loads(f.readline().replace("'", '"'))
    return columns
