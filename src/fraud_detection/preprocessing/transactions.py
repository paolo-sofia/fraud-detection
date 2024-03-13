import logging
import os
import pathlib

import polars as pl
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("fraud-detection")


def load_preprocessed_transactions() -> pl.LazyFrame:
    logger.info(f"Loading preprocessed transactions from {os.getenv('TRANSACTIONS_PATH')}")
    try:
        return pl.scan_parquet(os.getenv("PROCESSED_TRANSACTIONS_PATH"))
    except FileNotFoundError as e:
        logger.error(e)
        return pl.LazyFrame()


def load_transactions() -> pl.LazyFrame:
    """Loads and returns the transactions data.

    Returns:
        pl.LazyFrame: The loaded transactions' data.
    """
    if not pathlib.Path(os.getenv("PROCESSED_TRANSACTIONS_PATH")).exists():
        try:
            return pl.scan_csv(os.getenv("TRANSACTIONS_PATH"))
        except FileNotFoundError as e:
            logger.error(e)
            return pl.LazyFrame()

    return load_preprocessed_transactions()


def fill_nulls_categorical_columns(dataframe: pl.LazyFrame) -> pl.LazyFrame:
    """Fills null values in categorical columns with "unknown".

    Args:
        dataframe: The input DataFrame.

    Returns:
        pl.LazyFrame: The DataFrame with null values in categorical columns filled with "unknown".
    """
    transforms: list[pl.Expr] = [pl.col(f"M{index}").fill_null("unknown") for index in range(1, 10)]
    transforms.extend(pl.col(column).fill_null("unknown") for column in ["card4", "card6", "ProductCD"])
    return dataframe.with_columns(
        *transforms,
        pl.col("R_emaildomain").str.split(".").list.first().fill_null("unknown"),
        pl.col("P_emaildomain").str.split(".").list.first().fill_null("unknown"),
    )


def preprocess_transactions(transactions: pl.LazyFrame) -> pl.LazyFrame:
    """Preprocesses the transactions data.

    Args:
        transactions (pl.LazyFrame): The transactions data.

    Returns:
        pl.LazyFrame: The preprocessed transactions data.
    """
    # fill null values of numerical features to their median
    transactions = transactions.with_columns(
        *[
            pl.col(col).fill_null(pl.col(col).median()).shrink_dtype().alias(col)
            for col in transactions.select(pl.col(pl.NUMERIC_DTYPES)).columns
        ]
    )

    return fill_nulls_categorical_columns(transactions)


def save_processed_transactions_to_file(transactions: pl.LazyFrame) -> None:
    """Saves the processed transactions data to a file. If the file already exists, writing is aborted.

    Args:
        transactions (pl.LazyFrame): The processed transactions' data.

    Returns:
        None
    """
    if pathlib.Path(os.getenv("PROCESSED_TRANSACTIONS_PATH")).exists():
        logger.info(
            f"Identities have already been processed and saved to {os.getenv('PROCESSED_TRANSACTIONS_PATH')}, "
            f"skipping saving to disk."
        )
        return

    logger.info(f"Saving processed transactions to {os.getenv('PROCESSED_TRANSACTIONS_PATH')}")
    transactions.collect().write_parquet(os.getenv("PROCESSED_TRANSACTIONS_PATH"))


def load_and_preprocess_transactions() -> pl.LazyFrame:
    """Loads, preprocesses and saves the transactions data.

    Returns:
        pl.LazyFrame: The preprocessed transactions' data.
    """
    transactions: pl.LazyFrame = load_transactions()
    transactions = preprocess_transactions(transactions)
    save_processed_transactions_to_file(transactions)
    return transactions
