"""Settings shared by the pipeline, training and the API. No logic here."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
YEAR = 2025
MONTH = 1

# Cleaning rules (see notebooks/01_explore_yellow_taxi.ipynb)
MIN_DURATION_MIN = 1
MAX_DURATION_MIN = 60
MAX_DISTANCE_MILES = 100
# API input limits (the API only accepts what the model was trained on)
MIN_PASSENGERS = 1
MAX_PASSENGERS = 6
MAX_LOCATION_ID = 265  # TLC taxi zones are numbered 1 to 265

MAX_SPEED_MPH = 80  # faster is physically impossible in the city: wrong distance or timestamps

# Rows with a missing passenger_count are kept and filled with this constant.
# A constant (not the data's median) so training and the API fill identically.
DEFAULT_PASSENGER_COUNT = 1

PROCESSED_NAME = f"yellow_{YEAR}-{MONTH:02d}_clean.parquet"

# Training
RANDOM_SEED = 42
VAL_SIZE = 0.2
# Baseline Random Forest. Unlimited depth on 2.7M rows gives 500+ MB models, so the
# trees are kept small with min_samples_leaf and max_samples (see README, step 3).
RF_PARAMS = {
    "n_estimators": 50,
    "min_samples_leaf": 100,
    "max_samples": 0.5,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

# MLflow: SQLite database + artifact folder inside MLFLOW_DIR. (The ./mlruns file store is
# in maintenance mode and refused by MLflow 3.17.)
MLFLOW_DIR = ROOT
EXPERIMENT_NAME = "yellow-taxi-duration"
# MLflow saves sklearn models with skops, which refuses unknown types. Every tree model contains
# this one; we train and load our own models, so we trust exactly this type and nothing else.
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]
# Model Registry: the API serves whichever version the "production" alias points at.
REGISTERED_MODEL_NAME = "yellow-taxi-duration"
PRODUCTION_ALIAS = "production"

PICKUP_COL ="tpep_pickup_datetime"
DROPOFF_COL = "tpep_dropoff_datetime"
TARGET = "duration"

FEATURES = [
    "pickup_hour",
    "pickup_weekday",
    "PULocationID",
    "DOLocationID",
    "passenger_count",
    "trip_distance",
]
