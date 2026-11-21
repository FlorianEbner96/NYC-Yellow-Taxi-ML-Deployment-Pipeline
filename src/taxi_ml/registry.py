"""Find and load the trained model from the local MLflow store.

Used by the behavioral tests now and by the API later.
"""

from pathlib import Path

import mlflow
from mlflow import MlflowClient

from taxi_ml.config import EXPERIMENT_NAME, MLFLOW_DIR
from taxi_ml.train import tracking_uri

NOT_TRAINED = "No trained model found in {where}. Train one first: uv run python -m taxi_ml.train"


def latest_run_id(mlflow_dir: Path = MLFLOW_DIR) -> str:
    """Id of the newest finished training run."""
    # check first: opening a missing SQLite file would silently create an empty database
    if not (mlflow_dir / "mlflow.db").exists():
        raise FileNotFoundError(NOT_TRAINED.format(where=mlflow_dir))
    client = MlflowClient(tracking_uri=tracking_uri(mlflow_dir))
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
