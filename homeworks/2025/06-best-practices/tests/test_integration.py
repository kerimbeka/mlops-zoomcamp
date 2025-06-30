import os

import polars as pl
from dotenv import load_dotenv
from src.batch import get_input_path, save_data

from tests.test_batch import dt


def test_save_data_s3():
    load_dotenv()
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    schema = {
        "PULocationID": pl.Int64,
        "DOLocationID": pl.Int64,
        "tpep_pickup_datetime": pl.Datetime,
        "tpep_dropoff_datetime": pl.Datetime,
    }
    df_input = pl.DataFrame(data, schema=schema, orient="row")

    year = 2023
    month = 1

    output_file = get_input_path(year, month)

    endpoint_url = os.getenv("S3_ENDPOINT_URL", None)

    save_data(df_input, output_file, endpoint_url)
