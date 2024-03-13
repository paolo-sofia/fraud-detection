import os
import pathlib

import polars as pl
from fraud_detection.preprocessing.identities import load_identities, logger
from fraud_detection.preprocessing.transactions import load_transactions
from fraud_detection.utils.columns import IdentitiesColumns


def save_processed_data_to_disk(data: pl.LazyFrame) -> None:
    if pathlib.Path(os.getenv("PROCESSED_DATA_PATH")).exists():
        logger.warn(
            f"Data have already been processed and saved to {os.getenv('PROCESSED_DATA_PATH')}, "
            f"skipping saving to disk."
        )
        return

    logger.info(f"Saving processed identities to {os.getenv('PROCESSED_DATA_PATH')}")
    data.collect().write_parquet(os.getenv("PROCESSED_DATA_PATH"))


def preprocess_data_for_training() -> pl.LazyFrame:
    if pathlib.Path(os.getenv("PROCESSED_DATA_PATH")).exists():
        return pl.scan_parquet(os.getenv("PROCESSED_DATA_PATH"))

    identities: pl.LazyFrame = load_identities()
    transactions: pl.LazyFrame = load_transactions()

    data: pl.LazyFrame = transactions.join(other=identities, on=IdentitiesColumns.TransactionID, how="left")

    save_processed_data_to_disk(data=data)

    return data
