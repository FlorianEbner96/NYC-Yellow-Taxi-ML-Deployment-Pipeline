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

# Rows with a missing passenger_count are kept and filled with this constant.
# A constant (not the data's median) so training and the API fill identically.
DEFAULT_PASSENGER_COUNT = 1

PICKUP_COL = "tpep_pickup_datetime"
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
