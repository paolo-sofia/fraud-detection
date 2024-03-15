import logging

import polars as pl
from dotenv import load_dotenv

from src.fraud_detection.preprocessing.identities import preprocess_identities
from src.fraud_detection.preprocessing.transactions import preprocess_transactions
from src.fraud_detection.utils.columns import IdentitiesColumns

load_dotenv()


def process_id_23_and_id_34(dataframe: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    transforms: list[pl.Expr] = []
    if IdentitiesColumns.id_23 in dataframe.columns:
        transforms.append(
            pl.col(IdentitiesColumns.id_23)
            .str.replace_all(pattern=":", value="_")
            .cast(pl.Categorical)
            .alias(IdentitiesColumns.id_23)
        )
    if IdentitiesColumns.id_34 in dataframe.columns:
        transforms.append(
            pl.col(IdentitiesColumns.id_34)
            .str.replace_all(pattern=":", value="_")
            .cast(pl.Categorical)
            .alias(IdentitiesColumns.id_34)
        )
    return dataframe.with_columns(*transforms)


def prepare_data_for_inference(
    inputs: dict[str, str | int | bool | float], columns_to_select: list[str]
) -> pl.DataFrame:
    if missing_columns := set(columns_to_select).difference(list(inputs.keys())):
        raise ValueError(f"Missing columns: {missing_columns}")

    if additional_columns := set(list(inputs.keys())).difference(columns_to_select):
        logging.warning(
            f"Additional columns were passed as inputs, dropping them. Additional columns passed: {additional_columns}"
        )
        inputs = {k: v for k, v in inputs.items() if k not in additional_columns}

    dataframe: pl.DataFrame = pl.from_dict({k: [v] for k, v in inputs.items()})
    dataframe = dataframe.drop(IdentitiesColumns.TransactionID)

    dataframe = preprocess_identities(dataframe)
    dataframe = preprocess_transactions(dataframe)

    dataframe = process_id_23_and_id_34(dataframe)

    dataframe = dataframe.with_columns(pl.col(pl.NUMERIC_DTYPES).shrink_dtype(), pl.col(pl.String).cast(pl.Categorical))
    return dataframe.select(columns_to_select)


"""{
        "TransactionDT": 86506.0,
        "TransactionAmt": 50.0,
        "ProductCD": "H",
        "card1": 4497.0,
        "card2": 514.0,
        "card3": 150.0,
        "card6": "credit",
        "P_emaildomain": "gmail",
        "R_emaildomain": "unknown",
        "C1": 1.0,
        "C10": 1.0,
        "D2": 97.0,
        "D4": 26.0,
        "D8": 37.875,
        "D10": 15.0,
        "V258": 1.0,
        "id_01": 0.0,
        "id_02": 70787.0,
        "id_06": 0.0,
        "id_19": 542.0,
        "id_20": 144.0,
        "id_30": "android 7",
        "id_31": "samsung browser",
    }"""
