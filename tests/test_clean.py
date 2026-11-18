import pandas as pd

from taxi_ml.clean import add_duration, clean


def test_add_duration_is_dropoff_minus_pickup_in_minutes(raw_trips):
    out = add_duration(raw_trips)
    assert out.loc[0, "duration"] == 10
    assert out.loc[1, "duration"] == 30
    assert out.loc[2, "duration"] == 15  # crosses midnight


def test_add_duration_does_not_modify_the_input(raw_trips):
    add_duration(raw_trips)
    assert "duration" not in raw_trips.columns


def test_clean_keeps_only_the_valid_rows(raw_trips):
    out = clean(raw_trips)
    assert out.index.tolist() == [0, 1, 2]


def test_clean_drops_non_positive_and_too_short_durations(raw_trips):
    out = clean(raw_trips)
    assert (out["duration"] >= 1).all()
    assert not {3, 4} & set(out.index)  # 0 min and negative


def test_clean_drops_durations_over_60_minutes(raw_trips):
    assert 5 not in clean(raw_trips).index


def test_clean_keeps_a_duration_of_exactly_60_minutes(raw_trips):
    raw_trips.loc[5, "tpep_dropoff_datetime"] = pd.Timestamp("2025-01-08 12:00")
    assert 5 in clean(raw_trips).index


def test_clean_drops_zero_and_too_long_distances(raw_trips):
    out = clean(raw_trips)
    assert not {6, 7} & set(out.index)


def test_clean_drops_trips_picked_up_outside_january(raw_trips):
    out = clean(raw_trips)
    assert not {8, 9} & set(out.index)


def test_clean_keeps_rows_with_missing_passenger_count(raw_trips):
    out = clean(raw_trips)
    assert 2 in out.index
    assert out.loc[2, "passenger_count"] != out.loc[2, "passenger_count"]  # still NaN; filled in features


def test_clean_on_an_empty_frame_returns_an_empty_frame(raw_trips):
    out = clean(raw_trips.iloc[0:0])
    assert out.empty
    assert "duration" in out.columns
