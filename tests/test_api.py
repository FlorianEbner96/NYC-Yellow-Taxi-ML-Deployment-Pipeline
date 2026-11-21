import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from taxi_ml.features import build_features
from taxi_ml.pipeline import build_dataset

VALID = {
    "pickup_datetime": "2025-01-14T17:30:00",
    "PULocationID": 161,
    "DOLocationID": 236,
    "passenger_count": 2,
    "trip_distance": 2.4,
}


class RecordingModel:
    """Stands in for the real model: remembers the features it was given, always predicts 12.345."""

    def __init__(self):
        self.seen = None

    def predict(self, X):
        self.seen = X.copy()
        return np.full(len(X), 12.345)


@pytest.fixture
def model():
    return RecordingModel()


@pytest.fixture
def client(model):
    with TestClient(create_app(lambda: (model, "7"))) as client:
        yield client


# ---------- /predict: the happy path ----------


def test_predict_returns_the_duration_and_the_model_version(client):
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    assert response.json() == {"predicted_duration_minutes": 12.35, "model_version": "7"}


def test_health_reports_the_model_version(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "7"}


def test_passenger_count_is_optional_and_defaults_to_one(client, model):
    body = {k: v for k, v in VALID.items() if k != "passenger_count"}
    assert client.post("/predict", json=body).status_code == 200
    assert model.seen.loc[0, "passenger_count"] == 1


def test_predict_accepts_the_limits_of_the_valid_ranges(client):
    for change in [
        {"passenger_count": 1}, {"passenger_count": 6},
        {"PULocationID": 1}, {"DOLocationID": 265},
        {"trip_distance": 0.01}, {"trip_distance": 100},
    ]:
        assert client.post("/predict", json={**VALID, **change}).status_code == 200, change


# ---------- training and serving build identical features (the step 6 goal) ----------


def test_the_api_builds_the_same_features_as_the_training_pipeline(client, model):
    raw_trip = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2025-01-14 17:30"]),
            "tpep_dropoff_datetime": pd.to_datetime(["2025-01-14 17:45"]),  # makes it a valid training row
            "passenger_count": [2.0],
            "trip_distance": [2.4],
            "PULocationID": [161],
            "DOLocationID": [236],
        }
    )
    pipeline_features, _ = build_dataset(raw_trip)
    client.post("/predict", json=VALID)
    pd.testing.assert_frame_equal(
        model.seen.reset_index(drop=True), pipeline_features.reset_index(drop=True), check_dtype=False
    )


def test_the_api_passes_exactly_the_feature_columns_to_the_model(client, model):
    client.post("/predict", json=VALID)
    assert model.seen.columns.tolist() == build_features(
        pd.DataFrame({"tpep_pickup_datetime": pd.to_datetime(["2025-01-14"]), "PULocationID": [1],
                      "DOLocationID": [1], "passenger_count": [1.0], "trip_distance": [1.0]})
    ).columns.tolist()


# ---------- /predict: invalid input is rejected with 422 and the model is never called ----------


@pytest.mark.parametrize(
    "change",
    [
        {"passenger_count": 0},
        {"passenger_count": 7},
        {"passenger_count": 1.5},
        {"trip_distance": 0},
        {"trip_distance": -1},
        {"trip_distance": 100.1},
        {"PULocationID": 0},
        {"PULocationID": 266},
        {"DOLocationID": 0},
        {"DOLocationID": 266},
        {"pickup_datetime": "not a date"},
        {"trip_distance": "far"},
    ],
    ids=str,
)
def test_invalid_values_are_rejected(client, model, change):
    response = client.post("/predict", json={**VALID, **change})
    assert response.status_code == 422
    assert model.seen is None


@pytest.mark.parametrize("missing", ["pickup_datetime", "PULocationID", "DOLocationID", "trip_distance"])
def test_missing_required_fields_are_rejected(client, model, missing):
    body = {k: v for k, v in VALID.items() if k != missing}
    response = client.post("/predict", json=body)
    assert response.status_code == 422
    assert missing in response.text
    assert model.seen is None


def test_the_error_message_names_the_field_that_is_wrong(client):
    response = client.post("/predict", json={**VALID, "passenger_count": 9})
    assert "passenger_count" in response.text


# ---------- start-up ----------


def test_the_app_refuses_to_start_when_there_is_no_model():
    def no_model():
        raise FileNotFoundError("No model is promoted to 'production' yet.")

    with pytest.raises(FileNotFoundError, match="promoted"):
        with TestClient(create_app(no_model)):
            pass


def test_the_model_is_loaded_once_not_per_request(model):
    calls = []

    def loader():
        calls.append(1)
        return model, "1"

    with TestClient(create_app(loader)) as client:
        for _ in range(3):
            client.post("/predict", json=VALID)
    assert len(calls) == 1


def test_unknown_fields_are_rejected_instead_of_silently_ignored(client, model):
    response = client.post("/predict", json={**VALID, "passengers": 3})  # typo of passenger_count
    assert response.status_code == 422
    assert "passengers" in response.text
    assert model.seen is None


# ---------- smoke test with the real promoted model ----------


def test_the_real_promoted_model_gives_a_plausible_duration():
    from taxi_ml.registry import load_production_model

    try:
        load_production_model()
    except FileNotFoundError as error:
        pytest.skip(str(error))

    from app.main import app

    with TestClient(app) as client:
        short = client.post("/predict", json={**VALID, "trip_distance": 0.5}).json()
        long = client.post("/predict", json={**VALID, "trip_distance": 10}).json()
    assert 1 <= short["predicted_duration_minutes"] < long["predicted_duration_minutes"] <= 60
