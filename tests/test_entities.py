import os
import pathlib
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from dotenv import load_dotenv
from src.fraud_detection.preprocessing.training import (
    load_identities,
    load_preprocessed_identities,
)

load_dotenv(dotenv_path="../tests/tests.env", verbose=True, override=True)


# Mock logger to avoid errors during testing
class MockLogger:
    def error(self, msg):
        pass


logger = MockLogger()


@pytest.fixture
def temp_processed_file() -> None:
    yield

    pathlib.Path(os.getenv("PROCESSED_IDENTITIES_PATH")).unlink(missing_ok=True)


@pytest.fixture
def mock_env(monkeypatch):
    # Setup for environment variable
    monkeypatch.setenv("IDENTITIES_PATH", "/data/identities.csv")
    monkeypatch.setenv("PROCESSED_IDENTITIES_PATH", "/data/processed_identities.csv")


@pytest.mark.parametrize(
    "env_path, file_exists, expected_call_count, test_id",
    [
        ("/valid/path/identities.parquet", True, 1, "happy_path_valid_file"),
        ("/valid/path/empty.parquet", True, 1, "happy_path_empty_file"),
        ("/invalid/path/identities.parquet", False, 1, "error_path_file_not_found"),
    ],
)
def test_load_preprocessed_identities(
    mock_env, env_path, file_exists, expected_call_count, test_id, monkeypatch, tmp_path
):
    # Arrange
    dummy_file = tmp_path / "identities.parquet"
    # sourcery skip: no-conditionals-in-tests
    if file_exists:
        dummy_file.touch()
    monkeypatch.setenv("PROCESSED_IDENTITIES_PATH", str(dummy_file))

    with patch("polars.scan_parquet", return_value=pl.LazyFrame()) as mock_scan_parquet:
        # Act
        result = load_preprocessed_identities()

        # Assert
        if file_exists:
            mock_scan_parquet.assert_called_once_with(str(dummy_file))
        else:
            assert mock_scan_parquet.call_count == expected_call_count

        assert isinstance(result, pl.LazyFrame)


@pytest.fixture
def mock_path_exists(monkeypatch):
    def mock_exists(path):
        return path == "/data/processed_identities.csv"

    monkeypatch.setattr(pathlib.Path, "exists", mock_exists)


@pytest.fixture
def mock_path_not_exists(monkeypatch):
    monkeypatch.setattr(pathlib.Path, "exists", MagicMock(return_value=False))


@pytest.fixture
def mock_scan_csv():
    with patch("polars.scan_csv") as mock:
        mock.return_value = pl.LazyFrame()
        yield mock


@pytest.fixture
def mock_load_preprocessed_identities():
    with patch("fraud_detection.preprocessing.training.load_preprocessed_identities") as mock:
        mock.return_value = pl.LazyFrame()
        yield mock


@pytest.mark.parametrize(
    "test_id, mock_exists_func, expected_call",
    [
        ("happy_path_processed", mock_path_exists, "load_preprocessed_identities"),
        ("happy_path_unprocessed", mock_path_not_exists, "scan_csv"),
    ],
)
def test_load_identities_happy_path(
    test_id, mock_exists_func, expected_call, mock_env, mock_scan_csv, mock_load_preprocessed_identities
):
    # Arrange
    mock_exists_func()

    # Act
    result = load_identities()

    # Assert
    assert isinstance(result, pl.LazyFrame)
    if expected_call == "load_preprocessed_identities":
        mock_load_preprocessed_identities.assert_called_once()
        mock_scan_csv.assert_not_called()
    else:
        mock_scan_csv.assert_called_once_with(os.getenv("IDENTITIES_PATH"))
        mock_load_preprocessed_identities.assert_not_called()


@pytest.mark.parametrize(
    "test_id, exception, mock_logger",
    [
        ("error_file_not_found", FileNotFoundError, logger),
    ],
)
def test_load_identities_error_cases(test_id, exception, mock_logger, mock_env, mock_path_not_exists, mock_scan_csv):
    # Arrange
    mock_path_not_exists()
    mock_scan_csv.side_effect = exception
    with patch("fraud_detection.preprocessing.training.logger", mock_logger) as mock_log:
        # Act
        result = load_identities()

        # Assert
        assert isinstance(result, pl.LazyFrame)
        mock_log.error.assert_called_once()
        mock_scan_csv.assert_called_once_with(os.getenv("IDENTITIES_PATH"))
