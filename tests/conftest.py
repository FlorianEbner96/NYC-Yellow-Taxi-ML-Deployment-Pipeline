import pandas as pd
import pytest


@pytest.fixture
def raw_trips() -> pd.DataFrame:
    """Small raw-like frame. Rows 0-2 are valid; every other row breaks exactly one rule."""
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                [
                    "2025-01-05 08:00",  # 0 valid, 10 min
                    "2025-01-06 17:30",  # 1 valid, 30 min
                    "2025-01-07 23:50",  # 2 valid, missing passenger_count -> kept
                    "2025-01-08 09:00",  # 3 duration 0 min
                    "2025-01-08 10:00",  # 4 duration negative
                    "2025-01-08 11:00",  # 5 duration 61 min
                    "2025-01-09 12:00",  # 6 distance 0
                    "2025-01-09 13:00",  # 7 distance 101
                    "2024-12-31 23:50",  # 8 pickup before January
                    "2025-02-01 00:00",  # 9 pickup in February (end is exclusive)
                ]
            ),
            "tpep_dropoff_datetime": pd.to_datetime(
                [
                    "2025-01-05 08:10",
                    "2025-01-06 18:00",
                    "2025-01-08 00:05",
                    "2025-01-08 09:00",
                    "2025-01-08 09:50",
                    "2025-01-08 12:01",
                    "2025-01-09 12:10",
                    "2025-01-09 13:30",
                    "2025-01-01 00:10",
                    "2025-02-01 00:10",
                ]
            ),
            "passenger_count": [1.0, 2.0, None, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "trip_distance": [2.0, 8.0, 3.0, 1.0, 1.0, 1.0, 0.0, 101.0, 1.0, 1.0],
            "PULocationID": [161, 132, 237, 1, 1, 1, 1, 1, 1, 1],
            "DOLocationID": [236, 230, 238, 1, 1, 1, 1, 1, 1, 1],
        }
    )
