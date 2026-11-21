"""Promote the best training run: register its model and point the 'production' alias at it.

Run with: uv run python -m taxi_ml.promote

"Best" = lowest validation RMSE, which is only a fair comparison between runs that used the same
cleaning rules and the same validation split. The candidates are printed so this stays visible.
"""

from pathlib import Path

import mlflow

from taxi_ml.config import EXPERIMENT_NAME, MLFLOW_DIR, PRODUCTION_ALIAS, REGISTERED_MODEL_NAME
from taxi_ml.registry import NOT_TRAINED, open_store
from taxi_ml.train import tracking_uri


def best_run(mlflow_dir: Path = MLFLOW_DIR) -> tuple[str, float, list[dict]]:
    """Return (run id, rmse_val, all candidates sorted from best to worst)."""
    client = open_store(mlflow_dir)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = (
        client.search_runs(
            [experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED' and metrics.rmse_val >= 0",
            order_by=["metrics.rmse_val ASC"],
        )
        if experiment
        else []
    )
    if not runs:
        raise FileNotFoundError(NOT_TRAINED.format(where=mlflow_dir))
    candidates = [
        {"run_id": r.info.run_id, "run_name": r.info.run_name, "rmse_val": r.data.metrics["rmse_val"]} for r in runs
    ]
    return candidates[0]["run_id"], candidates[0]["rmse_val"], candidates


def promote(mlflow_dir: Path = MLFLOW_DIR) -> int:
    """Register the best run's model (once) and point 'production' at it. Returns the version number."""
    run_id, rmse_val, candidates = best_run(mlflow_dir)
    client = open_store(mlflow_dir)

    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    existing = next((v for v in versions if v.run_id == run_id), None)  # promoting twice adds no duplicate
    if existing:
        version = int(existing.version)
    else:
        mlflow.set_tracking_uri(tracking_uri(mlflow_dir))
        version = int(mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL_NAME).version)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, str(version))

    print(f"compared {len(candidates)} run(s):")
    for c in candidates:
        print(f"  {'->' if c['run_id'] == run_id else '  '} {c['run_id'][:8]}  {c['run_name']}  rmse_val {c['rmse_val']:.3f}")
    print(f"'{PRODUCTION_ALIAS}' now points to {REGISTERED_MODEL_NAME} version {version} (rmse_val {rmse_val:.3f})")
    return version


if __name__ == "__main__":
    promote()
