"""Target definition and row filters. Pure functions: no file paths, no downloads."""

import pandas as pd

from taxi_ml.config import (
    DROPOFF_COL,
    MAX_DISTANCE_MILES,
    MAX_DURATION_MIN,
    MAX_SPEED_MPH,
    MIN_DURATION_MIN,
    PICKUP_COL,
    TARGET,
)


def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with `duration` = dropoff minus pickup, in minutes."""
    out = df.copy()
    out[TARGET] = (out[DROPOFF_COL] - out[PICKUP_COL]).dt.total_seconds() / 60
    return out


def clean(df: pd.DataFrame, start: str = "2025-01-01", end: str = "2025-02-01") -> pd.DataFrame:
    """Add the target and keep only plausible trips picked up in [start, end).

    Rows with a missing passenger_count are kept on purpose (15% of the data);
    they are filled in features.py.
    """
    out = add_duration(df)
    keep = (
        out[PICKUP_COL].ge(start)
        & out[PICKUP_COL].lt(end)
        & out[TARGET].between(MIN_DURATION_MIN, MAX_DURATION_MIN)
        & (out["trip_distance"] > 0)
        & (out["trip_distance"] <= MAX_DISTANCE_MILES)
        # speed <= max, written without dividing: distance <= max_speed * hours
        & (out["trip_distance"] <= MAX_SPEED_MPH * out[TARGET] / 60)
    )
    return out[keep]
