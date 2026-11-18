# NYC Yellow Taxi: Trip Duration Model, from Data to API

Train, track and locally serve a model that predicts the duration (in minutes) of New York City yellow taxi trips, using the January 2025 trip records.

Stack: `pandas` (data pipeline), `scikit-learn` `RandomForestRegressor` (model), `MLflow` (experiment tracking), `FastAPI` (serving).
The assignment is in [project-description.md](project-description.md).

> This README is written step by step while the project is built. Each section below documents one step: what was done, why, and what came out.
> **Status:** step 1 (exploration) done. Next: data pipeline, then model training, then API.

## Setup

Requirements: [uv](https://docs.astral.sh/uv/getting-started/installation/) and the Python version in [`.python-version`](.python-version).

```bash
uv sync
```

Open notebooks with the kernel from `.venv/` that `uv sync` creates.

## Project structure

```
data/raw/               downloaded parquet files, never edited (ignored by git)
notebooks/
  01_explore_yellow_taxi.ipynb   exploration only, not part of the pipeline
src/                    (planned) data pipeline, features, training
app/                    (planned) FastAPI app
tests/                  (planned)
```

This table is updated whenever a file is added.

## Log of steps

### Step 1: Data exploration (`notebooks/01_explore_yellow_taxi.ipynb`)

**Why a notebook first:** the cleaning rules and the feature list should come from looking at the data, not from guessing. The notebook is exploration only; its decisions are moved into `src/` as the pipeline in step 2.

**Data:** `yellow_tripdata_2025-01.parquet` from the [NYC TLC trip record data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), downloaded from `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet` into `data/raw/` (59 MB, 3,475,226 rows, 20 columns).

**Target:** `duration` = `tpep_dropoff_datetime - tpep_pickup_datetime`, in minutes.

#### Exploration findings

| Finding | Numbers | Consequence |
|---|---|---|
| Duration has impossible and extreme values | min -51,472 min, max 5,626 min; 2,051 rows with `duration <= 0`, 40,111 under 1 min, 31,867 over 60 min, 20 over 24 h. Median 11.7 min, 99th percentile 58.9 min | Keep only trips of 1 to 60 minutes. Extremes would dominate the RMSE |
| Distance has errors | 90,893 rows with distance 0, 162 over 100 miles (max 276,424) | Keep distance in (0, 100] miles |
| Month window | 22 rows have a pickup outside January (21 in Dec 2024, 1 in Feb 2025) | Filter on the pickup date, as in the Green Taxi project |
| One block of rows has five empty fields | `passenger_count`, `RatecodeID`, `store_and_fwd_flag`, `congestion_surcharge`, `Airport_fee` are all missing in the same 15.5% of rows (540,149) | Do **not** drop them, that would remove 16% of the data. Fill `passenger_count` with its median (1) |
| Passenger count | 24,656 rows with 0 passengers, 18 with more than 6 | Kept for now (they do not make the duration wrong) |
| Fares are unreliable | 145,516 rows with `fare_amount <= 0`, min -900 | Not used as features anyway (see leakage) |
| Distance carries strong signal | correlation of distance and duration (1 to 120 min trips) is 0.81 | A baseline should already work well |
| Location IDs | 261 pickup and 260 dropoff zones; the 20 most common pickup-dropoff pairs cover only 6.4% of trips | Treat as categories, not a lookup of routes |

With the three cleaning rules (month window, duration 1 to 60 min, distance in (0, 100]) **2,852,634 rows remain** if rows with a missing `passenger_count` are also dropped (82.1%). Keeping those rows instead (decision above) leaves more.

#### Feature decision and leakage

The API is called before a trip ends, so it can only use what is known at pickup. Features chosen for the baseline:

- pickup hour and pickup weekday (from `tpep_pickup_datetime`),
- `PULocationID`, `DOLocationID`,
- `passenger_count`,
- `trip_distance`.

Not used: `tpep_dropoff_datetime` (it defines the target), and all fare, tip, toll, surcharge and `payment_type` columns, because they are only known after the trip.

**Limitation to document in the results:** `trip_distance` is measured by the meter during the trip, so strictly it is not known at pickup. It is the strongest predictor, so the baseline uses it, but the final write-up should say so, and the model can be compared against a version without it.

## Reproduce the exploration

```bash
uv sync
mkdir -p data/raw
curl -L -o data/raw/yellow_tripdata_2025-01.parquet https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet
```

Then run `notebooks/01_explore_yellow_taxi.ipynb`. (Step 2 replaces the manual download with a pipeline step.)

## Questions from the assignment

1. What is the RMSE of your final model? *(to be answered after training)*
2. What would you do differently if you had more time? *(to be answered at the end)*
