import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

from helpers import SMALL_PARAMS, make_processed
from taxi_ml.config import FEATURES, TARGET
from taxi_ml.train import load_processed, rmse, run, split_data, tracking_uri, train_baseline


@pytest.fixture
def processed():
    return make_processed()


@pytest.fixture
def processed_file(tmp_path, processed):
    path = tmp_path / "processed.parquet"
    processed.to_parquet(path)
    return path


# ---------- unit tests ----------


def test_load_processed_returns_features_and_target(processed_file):
    X, y = load_processed(processed_file)
    assert X.columns.tolist() == FEATURES
    assert y.name == TARGET
    assert len(X) == len(y) == 400


def test_load_processed_gives_a_clear_error_when_the_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="pipeline"):
        load_processed(tmp_path / "nope.parquet")


def test_split_data_uses_20_percent_for_validation_by_default(processed):
    X, y = processed[FEATURES], processed[TARGET]
    X_train, X_val, y_train, y_val = split_data(X, y)
    assert len(X_val) == len(y_val) == 80
    assert len(X_train) == len(y_train) == 320


def test_split_data_has_no_overlap_and_loses_no_rows(processed):
    X, y = processed[FEATURES], processed[TARGET]
    X_train, X_val, _, _ = split_data(X, y)
    assert set(X_train.index).isdisjoint(X_val.index)
    assert set(X_train.index) | set(X_val.index) == set(X.index)


def test_split_data_keeps_features_and_target_aligned(processed):
    X, y = processed[FEATURES], processed[TARGET]
    X_train, _, y_train, _ = split_data(X, y)
    pd.testing.assert_series_equal(y_train, y.loc[X_train.index])


def test_split_data_is_reproducible(processed):
    X, y = processed[FEATURES], processed[TARGET]
    first = split_data(X, y)[1].index
    second = split_data(X, y)[1].index
    assert first.equals(second)


def test_rmse_matches_the_hand_calculation():
    # errors 3 and 4 -> sqrt((9 + 16) / 2)
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))


def test_rmse_of_a_perfect_prediction_is_zero():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0


def test_train_baseline_returns_a_fitted_model(processed):
    model = train_baseline(processed[FEATURES], processed[TARGET], SMALL_PARAMS)
    check_is_fitted(model)
    assert len(model.predict(processed[FEATURES])) == len(processed)


def test_train_baseline_overrides_the_default_params(processed):
    model = train_baseline(processed[FEATURES], processed[TARGET], SMALL_PARAMS)
    assert model.n_estimators == 5
    assert model.min_samples_leaf == 2


def test_train_baseline_beats_predicting_the_mean(processed):
    X_train, X_val, y_train, y_val = split_data(processed[FEATURES], processed[TARGET])
    model = train_baseline(X_train, y_train, SMALL_PARAMS)
    mean_prediction = np.full(len(y_val), y_train.mean())
    assert rmse(y_val, model.predict(X_val)) < rmse(y_val, mean_prediction)


# ---------- integration tests: the whole run, logged to a temporary MLflow store ----------


@pytest.fixture
def mlflow_dir(tmp_path):
    return tmp_path / "mlflow"


def test_run_logs_params_metrics_and_the_model_to_mlflow(processed_file, mlflow_dir):
    run_id = run(processed_path=processed_file, mlflow_dir=mlflow_dir, params=SMALL_PARAMS)

    mlflow.set_tracking_uri(tracking_uri(mlflow_dir))
    logged = mlflow.get_run(run_id)
    assert logged.data.params["n_estimators"] == "5"
    assert logged.data.params["n_train"] == "320"
    assert logged.data.params["n_val"] == "80"
    for metric in ("rmse_val", "rmse_train", "rmse_val_mean_baseline"):
        assert metric in logged.data.metrics


def test_run_logs_a_validation_rmse_better_than_the_mean_baseline(processed_file, mlflow_dir):
    run_id = run(processed_path=processed_file, mlflow_dir=mlflow_dir, params=SMALL_PARAMS)
    mlflow.set_tracking_uri(tracking_uri(mlflow_dir))
    metrics = mlflow.get_run(run_id).data.metrics
    assert metrics["rmse_val"] < metrics["rmse_val_mean_baseline"]


def test_the_logged_model_can_be_loaded_and_predicts(processed_file, processed, mlflow_dir):
    run_id = run(processed_path=processed_file, mlflow_dir=mlflow_dir, params=SMALL_PARAMS)
    mlflow.set_tracking_uri(tracking_uri(mlflow_dir))
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    predictions = model.predict(processed[FEATURES].head(3))
    assert predictions.shape == (3,)


def test_run_without_a_processed_file_points_to_the_pipeline(tmp_path, mlflow_dir):
    with pytest.raises(FileNotFoundError, match="pipeline"):
        run(processed_path=tmp_path / "missing.parquet", mlflow_dir=mlflow_dir)



def test_run_creates_the_database_and_artifacts_inside_mlflow_dir(processed_file, mlflow_dir):
    run(processed_path=processed_file, mlflow_dir=mlflow_dir, params=SMALL_PARAMS)
    assert (mlflow_dir / "mlflow.db").exists()
    assert any((mlflow_dir / "mlartifacts").rglob("MLmodel"))


def test_two_runs_go_into_the_same_experiment(processed_file, mlflow_dir):
    first = run(processed_path=processed_file, mlflow_dir=mlflow_dir, params=SMALL_PARAMS)
    second = run(processed_path=processed_file, mlflow_dir=mlflow_dir, params=SMALL_PARAMS)
    mlflow.set_tracking_uri(tracking_uri(mlflow_dir))
    assert mlflow.get_run(first).info.experiment_id == mlflow.get_run(second).info.experiment_id
    assert first != second
