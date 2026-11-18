import pandas as pd
import pytest

from taxi_ml.config import FEATURES
from taxi_ml.features import build_features


def test_build_features_returns_exactly_the_feature_columns_in_order(raw_trips):
    assert build_features(raw_trips).columns.tolist() == FEATURES


def test_build_features_extracts_hour_and_weekday_from_pickup(raw_trips):
    out = build_features(raw_trips)
    # 2025-01-05 is a Sunday (6), 2025-01-06 a Monday (0)
    assert out.loc[0, "pickup_hour"] == 8
    assert out.loc[0, "pickup_weekday"] == 6
    assert out.loc[1, "pickup_hour"] == 17
    assert out.loc[1, "pickup_weekday"] == 0


def test_build_features_fills_missing_passenger_count_with_the_constant(raw_trips):
    out = build_features(raw_trips)
    assert out.loc[2, "passenger_count"] == 1
    assert not out.isna().any().any()


def test_build_features_never_contains_columns_known_only_after_the_trip(raw_trips):
    raw_trips["tpep_dropoff_datetime"] = pd.Timestamp("2025-01-05 09:00")
    raw_trips["total_amount"] = 99.0
    cols = set(build_features(raw_trips).columns)
    assert not cols & {"tpep_dropoff_datetime", "total_amount", "duration"}


def test_build_features_does_not_modify_the_input(raw_trips):
    before = raw_trips.copy()
    build_features(raw_trips)
    pd.testing.assert_frame_equal(raw_trips, before)


def test_build_features_fails_clearly_when_a_column_is_missing(raw_trips):
    with pytest.raises(KeyError):
        build_features(raw_trips.drop(columns=["trip_distance"]))
