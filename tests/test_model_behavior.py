"""Behavioral tests of the TRAINED model (not of the code): does it behave like a taxi trip?

Three kinds, as in the course material:
  * Minimum functionality tests (MFT): simple cases with an obvious answer.
  * Invariance tests (INV): a change that should NOT matter must not change the prediction much.
  * Directional expectation tests (DIR): a change that should push the prediction one way.

By default the model is the newest run in the local MLflow store, so these tests judge a fresh
candidate before it is promoted (baseline now, tuned model later). They are skipped, with a reason, if no model or
no processed data exists. Set TAXI_TEST_MODEL=production to test the promoted model instead of the
newest run, and TAXI_TEST_MLFLOW_DIR to use a store somewhere else.

A Random Forest is not smooth, so single pairs of trips can break a direction by chance.
Directional tests therefore check the share of 5,000 sampled trips, not every trip.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from taxi_ml.config import FEATURES, MAX_DURATION_MIN, MIN_DURATION_MIN, MLFLOW_DIR, PROCESSED_DIR, PROCESSED_NAME
from taxi_ml.registry import load_latest_model, load_production_model

pytestmark = pytest.mark.behavior

# A typical trip context: Tuesday 14:00, Midtown -> Upper East Side, one passenger.
CONTEXT = {"pickup_hour": 14, "pickup_weekday": 1, "PULocationID": 161, "DOLocationID": 236, "passenger_count": 1}


@pytest.fixture(scope="module")
def model():
    mlflow_dir = Path(os.environ.get("TAXI_TEST_MLFLOW_DIR", MLFLOW_DIR))
    load = load_production_model if os.environ.get("TAXI_TEST_MODEL") == "production" else load_latest_model
    try:
        return load(mlflow_dir)
    except FileNotFoundError as error:
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def trips():
    """5,000 real trips (features only) used as the starting point for INV and DIR tests."""
    path = PROCESSED_DIR / PROCESSED_NAME
    if not path.exists():
        pytest.skip(f"{path} not found. Run: uv run python -m taxi_ml.pipeline")
    return pd.read_parquet(path)[FEATURES].sample(5000, random_state=0).astype("float64").reset_index(drop=True)


def predict_trip(model, distance: float, **overrides) -> float:
    """Prediction for one trip in the typical context, with some values overridden."""
    row = {**CONTEXT, **overrides, "trip_distance": distance}
    return float(model.predict(pd.DataFrame([row])[FEATURES].astype("float64"))[0])


def changed(trips: pd.DataFrame, **new_values) -> pd.DataFrame:
    """Copy of the trips with some columns set to a fixed value (or computed from the old ones)."""
    out = trips.copy()
    for column, value in new_values.items():
        out[column] = value(out) if callable(value) else value
    return out


# ---------- minimum functionality tests: simple cases with an obvious answer ----------


def test_mft_predictions_stay_inside_the_range_the_model_was_trained_on(model, trips):
    predictions = model.predict(trips)
    assert predictions.min() >= MIN_DURATION_MIN * 0.5  # nothing negative or near zero
    assert predictions.max() <= MAX_DURATION_MIN


def test_mft_a_very_short_trip_is_predicted_short(model):
    assert predict_trip(model, distance=0.3) < 10


def test_mft_a_long_trip_is_predicted_long(model):
    assert predict_trip(model, distance=20) > 25


def test_mft_a_10_mile_trip_takes_clearly_longer_than_a_half_mile_trip(model):
    assert predict_trip(model, distance=10) - predict_trip(model, distance=0.5) > 10


def test_mft_every_hour_and_weekday_gives_a_valid_prediction(model):
    predictions = [
        predict_trip(model, distance=3, pickup_hour=hour, pickup_weekday=weekday)
        for hour in range(24)
        for weekday in range(7)
    ]
    assert all(MIN_DURATION_MIN <= p <= MAX_DURATION_MIN for p in predictions)


def test_mft_extreme_but_possible_inputs_give_valid_predictions(model):
    extremes = pd.DataFrame(
        {
            "pickup_hour": [0, 23, 12],
            "pickup_weekday": [0, 6, 3],
            "PULocationID": [1, 265, 132],
            "DOLocationID": [265, 1, 132],
            "passenger_count": [0, 6, 1],
            "trip_distance": [0.01, 100.0, 50.0],
        }
    )[FEATURES].astype("float64")
    predictions = model.predict(extremes)
    assert pd.Series(predictions).between(0, MAX_DURATION_MIN).all()


# ---------- invariance tests: this should NOT matter ----------


@pytest.mark.parametrize("passengers", [2, 3, 4])
def test_inv_passenger_count_barely_changes_the_prediction(model, trips, passengers):
    before = model.predict(changed(trips, passenger_count=1))
    after = model.predict(changed(trips, passenger_count=passengers))
    difference = pd.Series(after - before).abs()
    # tolerance: well below the model's ~4 minute error
    assert difference.mean() < 0.5
    assert difference.quantile(0.95) < 1.0


def test_inv_the_same_trip_always_gets_the_same_prediction(model, trips):
    # Not bit-for-bit: the forest adds up its trees in parallel threads, so the last decimals
    # (about 1e-14 minutes) can differ between calls. Anything bigger would be a real problem.
    assert np.allclose(model.predict(trips), model.predict(trips.copy()), rtol=0, atol=1e-9)


# ---------- directional expectation tests: this should push the prediction one way ----------


def test_dir_more_distance_does_not_make_the_trip_shorter(model, trips):
    before = model.predict(trips)
    after = model.predict(changed(trips, trip_distance=lambda t: t["trip_distance"] * 1.5))
    difference = pd.Series(after - before)
    assert (difference >= -0.5).mean() >= 0.95  # at least 95% do not get shorter by more than 30 s
    assert difference.mean() > 1  # and on average a 50% longer trip is clearly longer


def test_dir_a_trip_gets_longer_step_by_step_with_distance(model):
    distances = [0.5, 1, 2, 5, 10, 15]
    predictions = [predict_trip(model, d) for d in distances]
    assert predictions == sorted(predictions)


def test_dir_rush_hour_is_slower_than_the_middle_of_the_night(model, trips):
    rush = model.predict(changed(trips, pickup_hour=17))
    night = model.predict(changed(trips, pickup_hour=3))
    difference = pd.Series(rush - night)
    assert difference.mean() > 0
    assert (difference > 0).mean() >= 0.9


def test_dir_a_sunday_is_not_slower_than_a_weekday_rush_hour(model, trips):
    sunday = model.predict(changed(trips, pickup_weekday=6, pickup_hour=17))
    thursday = model.predict(changed(trips, pickup_weekday=3, pickup_hour=17))
    assert sunday.mean() <= thursday.mean()
