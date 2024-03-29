import logging
import os
import pathlib

import polars as pl
from dotenv import load_dotenv

from ..utils.columns import IdentitiesColumns

load_dotenv()
logger = logging.getLogger("fraud-detection")


def load_preprocessed_identities() -> pl.LazyFrame:
    logger.info(f"Loading preprocessed identities from {os.getenv('IDENTITIES_PATH')}")
    try:
        return pl.scan_parquet(os.getenv("PROCESSED_IDENTITIES_PATH"))
    except FileNotFoundError as e:
        logger.error(e)
        return pl.LazyFrame()


def load_identities() -> tuple[pl.LazyFrame, bool]:
    """Loads the identities data.

    Returns a tuple containing the identities data as a `pl.LazyFrame` and a boolean value indicating whether the data
    was preprocessed or not.

    Raises:
        FileNotFoundError: If the identities data file is not found.

    Examples:
        >>> load_identities()
        (pl.LazyFrame, bool)
    """
    if not pathlib.Path(os.getenv("PROCESSED_IDENTITIES_PATH")).exists():
        try:
            return pl.scan_csv(os.getenv("IDENTITIES_PATH")), False
        except FileNotFoundError as e:
            logger.error(e)
            return pl.LazyFrame(), False

    return load_preprocessed_identities(), True


def fill_nulls_categorical_columns(dataframe: pl.LazyFrame) -> pl.LazyFrame:
    """Fills null values in categorical columns of the given dataframe.

    Args:
        dataframe: The input dataframe containing categorical columns with null values.

    Returns:
        A new dataframe with the null values in categorical columns filled.

    Examples:
        >>> fill_nulls_categorical_columns(dataframe)
        pl.LazyFrame
    """
    unknown_columns: list[str] = [
        # IdentitiesColumns.DeviceInfo,
        IdentitiesColumns.DeviceType,
        IdentitiesColumns.id_15,
        IdentitiesColumns.id_28,
        IdentitiesColumns.id_30,
        IdentitiesColumns.id_31,
        IdentitiesColumns.id_35,
        IdentitiesColumns.id_36,
        IdentitiesColumns.id_37,
        IdentitiesColumns.id_38,
    ]
    mode_columns: list[str] = [IdentitiesColumns.id_33]
    not_found_columns: list[str] = [
        IdentitiesColumns.id_12,
        IdentitiesColumns.id_16,
        IdentitiesColumns.id_27,
        IdentitiesColumns.id_29,
    ]

    transforms: list[pl.Expr] = []
    for column in dataframe.columns:
        if column == IdentitiesColumns.id_23:
            transforms.append(
                pl.col(IdentitiesColumns.id_23)
                .fill_null("ip_proxy:hidden")
                .str.to_lowercase()
                .alias(IdentitiesColumns.id_23)
            )
        elif column == IdentitiesColumns.id_34:
            transforms.append(
                pl.col(IdentitiesColumns.id_34)
                .fill_null("match_status:-1")
                .str.to_lowercase()
                .alias(IdentitiesColumns.id_34)
            )
        elif column in unknown_columns:
            transforms.append(pl.col(column).fill_null("unknown").str.to_lowercase().alias(column))
        elif column in mode_columns:
            transforms.append(pl.col(column).fill_null(pl.col(column).mode().str.to_lowercase().alias(column)))
        elif column in not_found_columns:
            transforms.append(pl.col(column).fill_null("not_found").str.to_lowercase().alias(column))

    return dataframe.with_columns(*transforms)


def process_id_30(identities: pl.LazyFrame) -> pl.LazyFrame:
    """Processes the id_30 column in the identities' data.

    It fills null values with the string unknown, and on non-null values, extracts the first part of the string until
    the first number of the version.

    Example:
        input: "android 5.0.0"
        output: "android 5"

    Args:
        identities (pl.LazyFrame): The identities' data.

    Returns:
        pl.LazyFrame: The processed identities data with the id_30 column transformed.
    """
    if IdentitiesColumns.id_30 in identities.columns:
        return identities.with_columns(
            pl.col(IdentitiesColumns.id_30)
            .str.to_lowercase()
            .str.extract(pattern="(^[^\\d]+(\\d+))")
            .fill_null("unknown")
            .alias(IdentitiesColumns.id_30)
        )
    return identities


def process_id_31(identities: pl.LazyFrame) -> pl.LazyFrame:
    """Processes the id_31 column in the identities' data.

    It fills null values with the string unknown, and on non-null values, extracts the first part of the string until
    the first number of the version. It also removes the following strings: "generic", "for android".
    The function also removes all the values that are not browsers, such as "59843 /build 1465416"

    Example:
    input: "Chrome 95.0.0 for android"
    output: "chrome 95"

    Args:
    identities (pl.LazyFrame): The identities' data.

    Returns:
    pl.LazyFrame: The processed identities data with the id_30 column transformed.
    """
    if IdentitiesColumns.id_31 in identities.columns:
        return identities.with_columns(
            pl.when(pl.col(IdentitiesColumns.id_31).str.contains("/", literal=True))
            .then(pl.lit("unknown"))
            .otherwise(pl.col(IdentitiesColumns.id_31).str.to_lowercase().str.extract(pattern="(^[^\\d]+)"))
            .str.replace_all(pattern="generic", value="")
            .str.replace_all(pattern="for android", value="")
            .str.strip_chars()
            .fill_null("unknown")
            .alias(IdentitiesColumns.id_31)
        )
    return identities


def process_id_33(identities: pl.LazyFrame) -> pl.LazyFrame:
    """Processes the id_33 column (screen resolution) in the identities' data.

    It fills null values with the string "-1x-1" which indicates a missing screen resolution, and then transforms
    the string to a struct of 2 values, width and height, that are then converted to columns.

    Example:
    Input
    | id_33     |
    | --------- |
    | 1920x1080 |
    | null      |

    Output
    | width | height |
    | ----- | ------ |
    | 1920  | 1080   |
    | -1    | -1     |

    Args:
    identities (pl.LazyFrame): The identities' data.

    Returns:
    pl.LazyFrame: The processed identities data with the id_30 column transformed.
    """
    if IdentitiesColumns.id_33 in identities.columns:
        return identities.with_columns(
            pl.col(IdentitiesColumns.id_33)
            .fill_null("-1x-1")
            .str.split("x")
            .list.eval(pl.element().cast(pl.Int32, strict=False))
            .list.to_struct(fields=[IdentitiesColumns.width, IdentitiesColumns.height])
            .alias(IdentitiesColumns.id_33)
        ).unnest(columns=[IdentitiesColumns.id_33])
    return identities


def preprocess_identities(identities: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """Preprocesses the identities data.

    Args:
        identities (pl.LazyFrame): The identities data.

    Returns:
        pl.LazyFrame: The preprocessed identities data.
    """
    identities = identities.drop(IdentitiesColumns.DeviceInfo)

    # fill null values of numerical features to their median
    identities = identities.with_columns(
        *[
            pl.col(col).fill_null(pl.col(col).median()).shrink_dtype().alias(col)
            for col in identities.select(pl.col(pl.NUMERIC_DTYPES)).columns
        ]
    )

    identities = fill_nulls_categorical_columns(identities)

    identities = process_id_30(identities)
    identities = process_id_31(identities)
    return process_id_33(identities)


def save_processed_identities_to_file(identities: pl.LazyFrame) -> None:
    """Saves the processed identities data to a file. If the file already exists, writing is aborted.

    Args:
        identities (pl.LazyFrame): The processed identities' data.

    Returns:
        None
    """
    if pathlib.Path(os.getenv("PROCESSED_IDENTITIES_PATH")).exists():
        logger.info(
            f"Identities have already been processed and saved to {os.getenv('PROCESSED_IDENTITIES_PATH')}, "
            f"skipping saving to disk."
        )
        return

    logger.info(f"Saving processed identities to {os.getenv('PROCESSED_IDENTITIES_PATH')}")
    identities.collect().write_parquet(os.getenv("PROCESSED_IDENTITIES_PATH"))


def load_and_preprocess_identities() -> pl.LazyFrame:
    """Loads, preprocesses and saves the identities data.

    Returns:
        pl.LazyFrame: The preprocessed identities' data.
    """
    identities: pl.LazyFrame
    is_processed: bool
    identities, is_processed = load_identities()

    if is_processed:
        return identities

    identities = preprocess_identities(identities)
    identities = identities.with_columns(pl.col(IdentitiesColumns.TransactionID).cast(pl.Int64))
    save_processed_identities_to_file(identities)
    return identities
