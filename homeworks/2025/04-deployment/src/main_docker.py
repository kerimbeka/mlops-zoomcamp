import pickle
from pathlib import Path

import click
import numpy as np
import polars as pl

CATEGORICAL_COLUMNS = ["PULocationID", "DOLocationID"]


def read_dataframe(filepath: Path | str) -> pl.DataFrame:
    df = (
        pl.scan_parquet(filepath)
        .with_columns(
            duration=(
                pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")
            ).dt.total_seconds()
            / 60
        )
        .filter((pl.col("duration") >= 1) & (pl.col("duration") <= 60))
        .with_columns(
            [pl.col(col).fill_nan(-1).cast(pl.String) for col in CATEGORICAL_COLUMNS]
        )
    )

    return df.collect()


@click.command()
@click.option("--year", type=int, required=True, help="Year to process.")
@click.option("--month", type=int, required=True, help="Month to process.")
def main(year: int, month: int) -> None:
    with open("model.bin", "rb") as f_in:
        dv, model = pickle.load(f_in)

    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    df = read_dataframe(input_file)
    dicts = df.select(pl.col(CATEGORICAL_COLUMNS)).to_dicts()
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    print(
        f"The mean of the predicted duration for {year:04d}-{month:02d}: {np.mean(y_pred):.2f}"
    )
    print(
        f"The standard deviation of the predicted duration for {year:04d}-{month:02d}: {np.std(y_pred):.2f}"
    )


if __name__ == "__main__":
    main()
