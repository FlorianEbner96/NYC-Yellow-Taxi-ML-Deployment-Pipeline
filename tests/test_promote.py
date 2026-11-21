import mlflow
import pytest
from mlflow import MlflowClient

from helpers import make_processed
from taxi_ml.config import PRODUCTION_ALIAS, REGISTERED_MODEL_NAME
from taxi_ml.promote import best_run, promote
from taxi_ml.train import run, tracking_uri

GOOD = {"n_estimators": 10, "min_samples_leaf": 2}  # learns the distance signal
BAD = {"n_estimators": 3, "min_samples_leaf": 150}  # only 2 leaves per tree on 320 rows: close to the mean


@pytest.fixture
def mlflow_dir(tmp_path):
    return tmp_path / "mlflow"


@pytest.fixture
def data_path(tmp_path):
    path = tmp_path / "processed.parquet"
    make_processed().to_parquet(path)
    return path


def production_run_id(mlflow_dir) -> str:
    client = MlflowClient(tracking_uri=tracking_uri(mlflow_dir))
    return client.get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS).run_id


def test_best_run_is_the_one_with_the_lowest_validation_rmse_not_the_newest(mlflow_dir, data_path):
    good = run(processed_path=data_path, mlflow_dir=mlflow_dir, params=GOOD)
    run(processed_path=data_path, mlflow_dir=mlflow_dir, params=BAD)  # newest, but worse
    run_id, rmse_val, candidates = best_run(mlflow_dir)
    assert run_id == good
    assert rmse_val == min(c["rmse_val"] for c in candidates)


def test_best_run_lists_every_candidate_it_compared(mlflow_dir, data_path):
    run(processed_path=data_path, mlflow_dir=mlflow_dir, params=GOOD)
    run(processed_path=data_path, mlflow_dir=mlflow_dir, params=BAD)
    _, _, candidates = best_run(mlflow_dir)
    assert len(candidates) == 2
    assert [c["rmse_val"] for c in candidates] == sorted(c["rmse_val"] for c in candidates)


def test_promote_points_the_production_alias_at_the_best_run(mlflow_dir, data_path):
    good = run(processed_path=data_path, mlflow_dir=mlflow_dir, params=GOOD)
    run(processed_path=data_path, mlflow_dir=mlflow_dir, params=BAD)
    promote(mlflow_dir)
    assert production_run_id(mlflow_dir) == good


def test_promote_returns_the_registered_version_number(mlflow_dir, data_path):
    run(processed_path=data_path, mlflow_dir=mlflow_dir, params=GOOD)
    assert promote(mlflow_dir) == 1


def test_promoting_twice_does_not_create_a_duplicate_version(mlflow_dir, data_path):
    run(processed_path=data_path, mlflow_dir=mlflow_dir, params=GOOD)
    assert promote(mlflow_dir) == promote(mlflow_dir) == 1
    client = MlflowClient(tracking_uri=tracking_uri(mlflow_dir))
    assert len(client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")) == 1


def test_a_worse_new_run_does_not_take_over_production(mlflow_dir, data_path):
    good = run(processed_path=data_path, mlflow_dir=mlflow_dir, params=GOOD)
    promote(mlflow_dir)
    run(processed_path=data_path, mlflow_dir=mlflow_dir, params=BAD)
    assert promote(mlflow_dir) == 1
    assert production_run_id(mlflow_dir) == good


def test_a_better_new_run_moves_the_alias_to_a_new_version(mlflow_dir, data_path):
    run(processed_path=data_path, mlflow_dir=mlflow_dir, params=BAD)
    assert promote(mlflow_dir) == 1
    better = run(processed_path=data_path, mlflow_dir=mlflow_dir, params=GOOD)
    assert promote(mlflow_dir) == 2
    assert production_run_id(mlflow_dir) == better


def test_the_promoted_model_can_be_loaded_by_alias(mlflow_dir, data_path):
    run(processed_path=data_path, mlflow_dir=mlflow_dir, params=GOOD)
    promote(mlflow_dir)
    mlflow.set_tracking_uri(tracking_uri(mlflow_dir))
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}")
    assert model.predict(make_processed(3).drop(columns="duration")).shape == (3,)


def test_promote_without_any_training_points_to_train(tmp_path):
    with pytest.raises(FileNotFoundError, match="taxi_ml.train"):
        promote(tmp_path / "empty")
