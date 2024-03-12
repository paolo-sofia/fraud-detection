import os
import pathlib

import polars as pl
import pytest
from dotenv import load_dotenv
from fraud_detection.preprocessing.training import load_and_preprocess_identities, load_identities
from fraud_detection.utils.columns import IdentitiesColumns

load_dotenv(dotenv_path="../tests/tests.env", verbose=True, override=True)


@pytest.fixture
def temp_processed_file() -> None:
    yield

    pathlib.Path(os.getenv("PROCESSED_IDENTITIES_PATH")).unlink(missing_ok=True)


def test_load_identities(temp_processed_file) -> None:
    # test file loaded case
    assert load_identities().head(1).collect().shape[0] == 1

    # test preprocessed file loaded case
    load_and_preprocess_identities()
    assert IdentitiesColumns.height in load_identities().columns

    # test file not found case
    os.environ["IDENTITIES_PATH"] = "temp.csv"
    os.environ["PROCESSED_IDENTITIES_PATH"] = "temp.csv"
    assert pl.LazyFrame() == load_identities()


def test_fill_nulls_categorical_columns() -> None:
    pass


def test_process_id_30() -> None:
    pass


def test_process_id_31() -> None:
    pass


def test_process_id_33() -> None:
    pass


def test_preprocess_identities() -> None:
    pass


def test_save_processed_identities_to_file() -> None:
    pass


def test_load_and_preprocess_identities() -> None:
    pass
