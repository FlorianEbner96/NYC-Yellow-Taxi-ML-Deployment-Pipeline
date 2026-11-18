"""Feature preparation. Used by training AND by the API, so both build features identically."""

import pandas as pd

from taxi_ml.config import DEFAULT_PASSENGER_COUNT, FEATURES, PICKUP_COL


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw trip columns into the model's feature columns (only pickup-time information)."""
    pickup = df[PICKUP_COL]
    out = pd.DataFrame(
        {
            "pickup_hour": pickup.dt.hour,
            "pickup_weekday": pickup.dt.day_of_week,
            "PULocationID": df["PULocationID"],
            "DOLocationID": df["DOLocationID"],
            "passenger_count": df["passenger_count"].fillna(DEFAULT_PASSENGER_COUNT),
            "trip_distance": df["trip_distance"],
        }
    )
    return out[FEATURES]
