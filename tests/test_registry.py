import pytest

from helpers import SMALL_PARAMS, make_processed
from taxi_ml.config import FEATURES
from taxi_ml.registry import latest_run_id, load_latest_model
from taxi_ml.train import run


@pytest.fixture
def trained(tmp_path):
    """A real training run in a temporary MLflow folder. Returns (mlflow_dir, run_id)."""
    data = make_processed()
    path = tmp_path / "processed.parquet"
    data.to_parquet(path)
    mlflow_dir = tmp_path / "mlflow"
    return mlflow_dir, run(processed_path=path, mlflow_dir=mlflow_dir, params=SMALL_PARAMS)


def test_latest_run_id_returns_the_run_that_was_trained(trained):
    mlflow_dir, run_id = trained
    assert latest_run_id(mlflow_dir) == run_id


def test_latest_run_id_returns_the_newest_of_two_runs(trained, tmp_path):
    mlflow_dir, first = trained
    path = tmp_path / "processed.parquet"
    second = run(processed_path=path, mlflow_dir=mlflow_dir, params=SMALL_PARAMS)
    assert second != first
    assert latest_run_id(mlflow_dir) == second


def test_load_latest_model_predicts_one_value_per_row(trained):
    mlflow_dir, _ = trained
    model = load_latest_model(mlflow_dir)
    X = make_processed(10)[FEATURES]
    assert model.predict(X).shape == (10,)


def test_load_latest_model_accepts_float_features_like_the_signature(trained):
    mlflow_dir, _ = trained
    model = load_latest_model(mlflow_dir)
    X = make_processed(5)[FEATURES].astype("float64")
    assert len(model.predict(X)) == 5


def test_latest_run_id_without_any_training_points_to_train(tmp_path):
    with pytest.raises(FileNotFoundError, match="taxi_ml.train"):
        latest_run_id(tmp_path / "empty")


def test_asking_for_a_model_does_not_create_a_database(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        latest_run_id(empty)
    assert not (empty / "mlflow.db").exists()



def test_load_production_model_returns_the_promoted_model_not_the_newest(tmp_path):
    from taxi_ml.promote import promote
    from taxi_ml.registry import load_production_model

    path = tmp_path / "processed.parquet"
    make_processed().to_parquet(path)
    mlflow_dir = tmp_path / "mlflow"
    good = {"n_estimators": 10, "min_samples_leaf": 2}
    bad = {"n_estimators": 3, "min_samples_leaf": 150}
    run(processed_path=path, mlflow_dir=mlflow_dir, params=good)
    run(processed_path=path, mlflow_dir=mlflow_dir, params=bad)  # newest, but worse
    promote(mlflow_dir)

    X = make_processed(50)[FEATURES]
    production = load_production_model(mlflow_dir).predict(X)
    newest = load_latest_model(mlflow_dir).predict(X)
    assert production.shape == (50,)
    assert not (production == newest).all()  # a different model than the newest run


def test_load_production_model_before_any_promotion_points_to_promote(trained):
    from taxi_ml.registry import load_production_model

    mlflow_dir, _ = trained
    with pytest.raises(FileNotFoundError, match="taxi_ml.promote"):
        load_production_model(mlflow_dir)


def test_load_production_model_without_a_database_points_to_train(tmp_path):
    from taxi_ml.registry import load_production_model

    with pytest.raises(FileNotFoundError, match="taxi_ml.train"):
        load_production_model(tmp_path / "empty")


def test_production_version_is_the_version_number_of_the_promoted_model(trained):
    from taxi_ml.promote import promote
    from taxi_ml.registry import production_version

    mlflow_dir, _ = trained
    promote(mlflow_dir)
    assert production_version(mlflow_dir) == "1"


def test_production_version_before_any_promotion_points_to_promote(trained):
    from taxi_ml.registry import production_version

    mlflow_dir, _ = trained
    with pytest.raises(FileNotFoundError, match="taxi_ml.promote"):
        production_version(mlflow_dir)
