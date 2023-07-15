import pandas as pd
from datetime import datetime

from batch import prepare_data

def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)

def test_prepare_data():
    data = [
    (None, None, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2), dt(1, 10)),
    (1, 2, dt(2, 2), dt(2, 3)),
    (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
    (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),     
    ]

    columns = ['PUlocationID', 'DOlocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    actual = prepare_data(df, ['PUlocationID', 'DOlocationID'])

    columns_expected = ['PUlocationID', 'DOlocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration']
    data_expected = [
                ('-1', '-1', dt(1, 2), dt(1, 10), 8.0),
                ('1', '-1', dt(1, 2), dt(1, 10), 8.0),
                ('1', '2', dt(2, 2), dt(2, 3), 1.0),     
    ]
    expected = pd.DataFrame(data_expected, columns=columns_expected)


    assert actual.shape[0] == 3
    assert expected.equals(actual)
