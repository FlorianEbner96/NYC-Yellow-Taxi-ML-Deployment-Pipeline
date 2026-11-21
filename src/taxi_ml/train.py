"""Train the baseline RandomForestRegressor, evaluate it with RMSE and log the run to MLflow.

Run with: uv run python -m taxi_ml.train   (needs the pipeline to have run first)
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from taxi_ml.config import (
    EXPERIMENT_NAME,
    FEATURES,
    MLFLOW_DIR,
    PROCESSED_DIR,
    PROCESSED_NAME,
    RANDOM_SEED,
    RF_PARAMS,
    SKOPS_TRUSTED_TYPES,
    TARGET,
    VAL_SIZE,
)


def load_processed(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read the pipeline output and return (features X, target y)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the pipeline first: uv run python -m taxi_ml.pipeline")
    df = pd.read_parquet(path)
    return df[FEATURES], df[TARGET]


def split_data(X, y, test_size: float = VAL_SIZE, seed: int = RANDOM_SEED):
    """Random train/validation split. Returns X_train, X_val, y_train, y_val."""
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def train_baseline(X_train, y_train, params: dict | None = None) -> RandomForestRegressor:
    """Fit a RandomForestRegressor. `params` override the defaults in config.RF_PARAMS."""
    model = RandomForestRegressor(**{**RF_PARAMS, **(params or {})})
    return model.fit(X_train, y_train)


def rmse(y_true, y_pred) -> float:
    """Root mean squared error, in the unit of the target (minutes)."""
    return float(root_mean_squared_error(y_true, y_pred))


def tracking_uri(mlflow_dir: Path) -> str:
    """SQLite database that stores runs, params and metrics."""
    return f"sqlite:///{(mlflow_dir / 'mlflow.db').as_posix()}"


def run(
    processed_path: Path | None = None,
    mlflow_dir: Path = MLFLOW_DIR,
    params: dict | None = None,
) -> str:
    """Split, train, evaluate and log one MLflow run. Returns the run id.

    MLflow writes mlflow.db and the model files (mlartifacts/) into `mlflow_dir`.
    """
    X, y = load_processed(processed_path or PROCESSED_DIR / PROCESSED_NAME)
    X_train, X_val, y_train, y_val = split_data(X, y)

    mlflow.set_tracking_uri(tracking_uri(mlflow_dir))
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        # fixed artifact folder, so models do not end up wherever the command was started
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=(mlflow_dir / "mlartifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="baseline-random-forest") as active:
        model = train_baseline(X_train, y_train, params)

        mlflow.log_params({**model.get_params(), "n_train": len(X_train), "n_val": len(X_val),
                           "val_size": VAL_SIZE, "split_seed": RANDOM_SEED, "features": ",".join(FEATURES)})
        mlflow.log_metrics(
            {
                "rmse_val": rmse(y_val, model.predict(X_val)),
                "rmse_train": rmse(y_train, model.predict(X_train)),
                # context: what you get by always predicting the average trip duration
                "rmse_val_mean_baseline": rmse(y_val, np.full(len(y_val), y_train.mean())),
            }
        )
        # floats in the signature: the API may send whole numbers, and ints would reject missing values
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_val.head(5).astype("float64"),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

        metrics = mlflow.get_run(active.info.run_id).data.metrics
        print(f"validation RMSE {metrics['rmse_val']:.3f} min (train {metrics['rmse_train']:.3f}, "
              f"always-predict-the-mean {metrics['rmse_val_mean_baseline']:.3f}) | run {active.info.run_id}")
        return active.info.run_id


if __name__ == "__main__":
    run()
