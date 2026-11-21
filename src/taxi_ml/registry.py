"""Find and load trained models from the local MLflow store.

* load_latest_model(): the newest finished run, to judge a fresh candidate (behavioral tests).
* load_production_model(): the version promoted with `python -m taxi_ml.promote`, used by the API.
"""

from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from taxi_ml.config import EXPERIMENT_NAME, MLFLOW_DIR, PRODUCTION_ALIAS, REGISTERED_MODEL_NAME
from taxi_ml.train import tracking_uri

NOT_TRAINED = "No trained model found in {where}. Train one first: uv run python -m taxi_ml.train"
NOT_PROMOTED = "No model is promoted to '{alias}' yet. Promote the best run: uv run python -m taxi_ml.promote"


def open_store(mlflow_dir: Path = MLFLOW_DIR) -> MlflowClient:
    """Client for the local MLflow database; fails clearly when nothing was trained yet."""
    # check first: opening a missing SQLite file would silently create an empty database
    if not (mlflow_dir / "mlflow.db").exists():
        raise FileNotFoundError(NOT_TRAINED.format(where=mlflow_dir))
    return MlflowClient(tracking_uri=tracking_uri(mlflow_dir))


def latest_run_id(mlflow_dir: Path = MLFLOW_DIR) -> str:
    """Id of the newest finished training run."""
    client = open_store(mlflow_dir)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = (
        client.search_runs(
            [experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if experiment
        else []
    )
    if not runs:
        raise FileNotFoundError(NOT_TRAINED.format(where=mlflow_dir))
    return runs[0].info.run_id


def load_latest_model(mlflow_dir: Path = MLFLOW_DIR):
    """The sklearn model of the newest finished run."""
    run_id = latest_run_id(mlflow_dir)
    mlflow.set_tracking_uri(tracking_uri(mlflow_dir))
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


def production_version(mlflow_dir: Path = MLFLOW_DIR) -> str:
    """Version number of the model the 'production' alias points at, e.g. "3"."""
    client = open_store(mlflow_dir)
    try:
        return str(client.get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS).version)
    except MlflowException:
        raise FileNotFoundError(NOT_PROMOTED.format(alias=PRODUCTION_ALIAS)) from None


def load_production_model(mlflow_dir: Path = MLFLOW_DIR):
    """The sklearn model the 'production' alias points at."""
    production_version(mlflow_dir)  # fails clearly if nothing was promoted
    mlflow.set_tracking_uri(tracking_uri(mlflow_dir))
    return mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}")
