"""Request and response shapes of the API. Pydantic rejects anything outside the ranges the model knows."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from taxi_ml.config import MAX_DISTANCE_MILES, MAX_LOCATION_ID, MAX_PASSENGERS, MIN_PASSENGERS


class TripRequest(BaseModel):
    """What is known when a trip starts (plus the distance, see README for that limitation)."""

    model_config = ConfigDict(
        extra="forbid",  # a misspelled field is an error, not silently ignored
        json_schema_extra={
            "example": {
                "pickup_datetime": "2025-01-14T17:30:00",
                "PULocationID": 161,
                "DOLocationID": 236,
                "passenger_count": 2,
                "trip_distance": 2.4,
            }
        },
    )

    pickup_datetime: datetime = Field(description="Pickup time in New York local time, ISO 8601")
    PULocationID: int = Field(ge=1, le=MAX_LOCATION_ID, description="TLC pickup zone id")
    DOLocationID: int = Field(ge=1, le=MAX_LOCATION_ID, description="TLC dropoff zone id")
    passenger_count: int = Field(default=1, ge=MIN_PASSENGERS, le=MAX_PASSENGERS)
    trip_distance: float = Field(gt=0, le=MAX_DISTANCE_MILES, description="Trip distance in miles")


class PredictionResponse(BaseModel):
    predicted_duration_minutes: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
