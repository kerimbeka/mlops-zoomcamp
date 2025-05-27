from pathlib import Path

import mlflow
import polars as pl
from prefect import flow, task
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

mlflow.sklearn.autolog()

CATEGORICAL_COLUMNS = ["PULocationID", "DOLocationID"]


@task
def read_dataframe(filepath: Path) -> pl.DataFrame:
    df = (
        pl.scan_parquet(filepath)
        .with_columns(
            duration=(
                pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")
            ).dt.total_seconds()
            / 60
        )
        .filter((pl.col("duration") >= 1) & (pl.col("duration") <= 60))
        .with_columns([pl.col(col).cast(pl.String) for col in CATEGORICAL_COLUMNS])
    )

    return df.collect()


@task(log_prints=True)
def train_linear_regression(
    df_train: pl.DataFrame,
) -> tuple[DictVectorizer, LinearRegression]:
    X_train_dicts = df_train.select(pl.col(CATEGORICAL_COLUMNS)).to_dicts()
    y_train = df_train.select(pl.col("duration")).to_numpy()

    with mlflow.start_run():
        dv = DictVectorizer()
        X_train = dv.fit_transform(X_train_dicts)
        mlflow.sklearn.log_model(dv, artifact_path="dict_vectorizer")

        lin_reg_model = LinearRegression()
        lin_reg_model.fit(X_train, y_train)

        print("Intercept of the model:", lin_reg_model.intercept_)

    return dv, lin_reg_model


@flow(log_prints=True)
def main(train_path: Path = Path("./data/yellow_tripdata_2023-03.parquet")) -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")

    df = pl.read_parquet(train_path)
    print(df.height)

    df_train = read_dataframe(train_path)

    print(df_train.height)

    dv, lin_reg_model = train_linear_regression(df_train)


if __name__ == "__main__":
    main()
