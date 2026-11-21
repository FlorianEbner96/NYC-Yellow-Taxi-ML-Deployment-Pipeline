"""FastAPI app: serves the promoted ("production") model.

Run with: uv run uvicorn app.main:app --port 8000   (docs at http://127.0.0.1:8000/docs)
"""

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, Request

from app.schemas import HealthResponse, PredictionResponse, TripRequest
from taxi_ml.features import build_features
from taxi_ml.registry import load_production_model, production_version


def load_production() -> tuple[Any, str]:
    """The promoted model and its version number."""
    return load_production_model(), production_version()


def create_app(load_model: Callable[[], tuple[Any, str]] = load_production) -> FastAPI:
    """Build the app. `load_model` is a parameter so tests can use a stand-in model."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # loaded once at start-up; if there is no model the app refuses to start
        app.state.model, app.state.model_version = load_model()
        yield

    app = FastAPI(title="NYC Yellow Taxi trip duration", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        return {"status": "ok", "model_version": request.app.state.model_version}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(trip: TripRequest, request: Request):
        raw = pd.DataFrame(
            {
                "tpep_pickup_datetime": pd.to_datetime([trip.pickup_datetime]),
                "PULocationID": [trip.PULocationID],
                "DOLocationID": [trip.DOLocationID],
                "passenger_count": [float(trip.passenger_count)],
                "trip_distance": [trip.trip_distance],
            }
        )
        features = build_features(raw).astype("float64")  # same function as in training
        minutes = float(request.app.state.model.predict(features)[0])
        return {"predicted_duration_minutes": round(minutes, 2), "model_version": request.app.state.model_version}

    return app


app = create_app()
