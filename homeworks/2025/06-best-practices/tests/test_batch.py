import polars as pl
from src.batch import prepare_data, CATEGORICAL_COLUMNS
from datetime import datetime
from polars.testing import assert_frame_equal



def dt(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ["PULocationID", "DOLocationID", "tpep_pickup_datetime", "tpep_dropoff_datetime"]
    schema = {"PULocationID": pl.Int64, "DOLocationID": pl.Int64, "tpep_pickup_datetime": pl.Datetime, "tpep_dropoff_datetime": pl.Datetime}
    df = pl.DataFrame(data, schema=schema, orient="row")
    df_actual = prepare_data(df, CATEGORICAL_COLUMNS)
    print(df_actual.select(columns).head())

    schema = {"PULocationID": pl.String, "DOLocationID": pl.String, "tpep_pickup_datetime": pl.Datetime, "tpep_dropoff_datetime": pl.Datetime}
    df_expected = pl.DataFrame(data=[
        (str(-1), str(-1), dt(1, 1), dt(1, 10)),
        (str(1), str(1), dt(1, 2), dt(1, 10)),
    ], schema=schema,  orient="row")
    
    print(df_expected.head())
    
    
    assert_frame_equal(df_actual.select(columns), df_expected)
