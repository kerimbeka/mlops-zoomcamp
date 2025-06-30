import os
import pickle
from pathlib import Path

import click
import pandas as pd
import polars as pl
from dotenv import load_dotenv

load_dotenv()


CATEGORICAL_COLUMNS = ["PULocationID", "DOLocationID"]


def get_input_path(year, month):
    default_input_pattern = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    input_pattern = os.getenv("INPUT_FILE_PATTERN", default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = "./output/yellow_tripdata_{year:04d}_{month:02d}.parquet"
    output_pattern = os.getenv("OUTPUT_FILE_PATTERN", default_output_pattern)
    return output_pattern.format(year=year, month=month)


def read_data(filepath: Path, s3_endpoint_url: str | None = None) -> pl.LazyFrame:
    if s3_endpoint_url:
        storage_options = {"client_kwargs": {"endpoint_url": s3_endpoint_url}}

        df = pd.read_parquet(filepath, storage_options=storage_options)
    else:
        df = pd.read_parquet(filepath)
    return pl.from_pandas(df)


def prepare_data(
    df: pl.LazyFrame | pl.DataFrame, cat_columns: list[str]
) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        df = df.lazy()

    df = (
        df.with_columns(
            duration=(
                pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")
            ).dt.total_seconds()
            / 60
        )
        .filter((pl.col("duration") >= 1) & (pl.col("duration") <= 60))
        .with_columns(
            [
                pl.col(col).fill_null(-1).fill_nan(-1).cast(pl.String)
                for col in cat_columns
            ]
        )
    )

    return df.collect()


def save_data(
    df: pl.DataFrame, filepath: Path, s3_endpoint_url: str | None = None
) -> None:
    if s3_endpoint_url:
        storage_options = {"client_kwargs": {"endpoint_url": s3_endpoint_url}}

        df.to_pandas().to_parquet(
            filepath,
            engine="pyarrow",
            compression=None,
            index=False,
            storage_options=storage_options,
        )
    else:
        df.to_pandas().to_parquet(
            filepath,
            engine="pyarrow",
            compression=None,
            index=False,
        )


@click.command()
@click.option("--year", type=int, required=True, help="Year to process.")
@click.option("--month", type=int, required=True, help="Month to process.")
def main(year: int, month: int) -> None:
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    s3_endpoint_url = os.getenv("S3_ENDPOINT_URL", None)

    with open("./models/model.bin", "rb") as f_in:
        dv, model = pickle.load(f_in)

    df = read_data(input_file, s3_endpoint_url)
    df = prepare_data(df, CATEGORICAL_COLUMNS)
    dicts = df.select(pl.col(CATEGORICAL_COLUMNS)).to_dicts()
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    print("Sum predicted duration:", y_pred.sum())
    df = df.with_row_index(name="row_id").with_columns(
        (pl.lit(f"{year:04d}/{month:02d}_") + pl.col("row_id").cast(pl.Utf8)).alias(
            "ride_id"
        )
    )

    df_result = df.select(
        ["ride_id", pl.Series(name="predicted_duration", values=y_pred)]
    )

    save_data(df_result, output_file, s3_endpoint_url)


if __name__ == "__main__":
    main()
