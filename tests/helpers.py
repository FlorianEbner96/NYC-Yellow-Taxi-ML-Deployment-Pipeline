"""Shared test helpers (imported by several test files)."""

import numpy as np
import pandas as pd

from taxi_ml.config import TARGET

SMALL_PARAMS = {"n_estimators": 5, "min_samples_leaf": 2}


def make_processed(n: int = 400) -> pd.DataFrame:
    """Processed-like data where duration clearly depends on distance."""
    rng = np.random.default_rng(0)
    distance = rng.uniform(0.5, 15, n)
    return pd.DataFrame(
        {
            "pickup_hour": rng.integers(0, 24, n),
            "pickup_weekday": rng.integers(0, 7, n),
            "PULocationID": rng.integers(1, 265, n),
            "DOLocationID": rng.integers(1, 265, n),
            "passenger_count": rng.integers(1, 5, n).astype(float),
            "trip_distance": distance,
            TARGET: 3 + 2.5 * distance + rng.normal(0, 1, n),
        }
    )
