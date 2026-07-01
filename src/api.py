"""FastAPI service for EnergyTypeNet predictions."""

from contextlib import asynccontextmanager
from typing import Dict

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import load_artifact, predict_dataframe


class BuildingFeatures(BaseModel):
    square_footage: float = Field(..., gt=0)
    number_of_occupants: float = Field(..., ge=0)
    appliances_used: float = Field(..., ge=0)
    average_temperature: float
    day_of_week: str = 'Weekday'
    energy_consumption: float = Field(..., ge=0)


artifact = None


def get_model_artifact() -> dict:
    """Load the model once and reuse it for later requests."""
    global artifact

    if artifact is None:
        artifact = load_artifact()

    return artifact


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model_artifact()
    yield


app = FastAPI(title='EnergyTypeNet API', version='1.0.0', lifespan=lifespan)


@app.get('/health')
def health() -> Dict[str, str]:
    return {'status': 'ok'}


@app.post('/predict')
def predict(features: BuildingFeatures) -> Dict:
    row = pd.DataFrame([{
        'Square Footage': features.square_footage,
        'Number of Occupants': features.number_of_occupants,
        'Appliances Used': features.appliances_used,
        'Average Temperature': features.average_temperature,
        'Day of Week': features.day_of_week,
        'Energy Consumption': features.energy_consumption,
    }])

    result = predict_dataframe(row, get_model_artifact())[0]

    return result


# Future autoencoder endpoint:
# @app.post('/anomaly_score')
# def anomaly_score(features: BuildingFeatures) -> Dict[str, float | bool]:
#     """Return reconstruction error, anomaly threshold and anomaly flag."""
