"""FastAPI service for EnergyTypeNet predictions and explanations."""

from contextlib import asynccontextmanager
from importlib.util import find_spec
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.data import CLASSES, FEATURE_COLS
from src.predict import load_artifact, predict_dataframe

try:
    from src.explainability import (
        explain_dataset_globally,
        explain_single_prediction,
    )
except Exception as exc:  # pragma: no cover - protects API startup
    explain_dataset_globally = None
    explain_single_prediction = None
    _EXPLAINABILITY_IMPORT_ERROR: Exception | None = exc
else:
    _EXPLAINABILITY_IMPORT_ERROR = None


LOGGER = logging.getLogger(__name__)
FEATURE_NAMES = list(FEATURE_COLS['all'])
CLASS_NAMES = list(CLASSES)
TRAINING_DATA_PATH = Path(__file__).resolve().parents[1] / 'data' / 'train_energy_data.csv'

artifact: dict | None = None
BACKGROUND_DATA: np.ndarray | None = None
_GLOBAL_EXPLANATION_CACHE: dict[str, Any] | None = None


class BuildingFeatures(BaseModel):
    square_footage: float = Field(..., gt=0)
    number_of_occupants: float = Field(..., ge=0)
    appliances_used: float = Field(..., ge=0)
    average_temperature: float
    day_of_week: str = 'Weekday'
    energy_consumption: float = Field(..., ge=0)


class ExplainRequest(BuildingFeatures):
    """Prediction input plus optional explanation controls."""

    include_shap: bool = True
    include_lime: bool = False


class ExplainResponse(BaseModel):
    """Prediction response enriched with optional local explanations."""

    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(alias='class')
    probabilities: dict[str, float]
    shap_explanation: dict[str, Any] | None = None
    lime_explanation: dict[str, Any] | None = None
    summary: str
    computation_time: float = Field(..., ge=0)
    explanation_available: bool


def is_package_available(package: str) -> bool:
    """Return whether an optional explanation package can be imported."""

    try:
        return find_spec(package) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def get_model_artifact() -> dict:
    """Load the model once and reuse it for later requests."""
    global artifact

    if artifact is None:
        artifact = load_artifact()
    return artifact


def get_background_data() -> np.ndarray:
    """Load and cache a deterministic 100-row explanation background."""
    global BACKGROUND_DATA

    if BACKGROUND_DATA is None:
        frame = pd.read_csv(TRAINING_DATA_PATH, usecols=FEATURE_NAMES)
        BACKGROUND_DATA = frame.loc[:, FEATURE_NAMES].head(100).to_numpy(dtype=float)
    return BACKGROUND_DATA


def _request_frame(features: BuildingFeatures) -> pd.DataFrame:
    return pd.DataFrame([{
        'Square Footage': features.square_footage,
        'Number of Occupants': features.number_of_occupants,
        'Appliances Used': features.appliances_used,
        'Average Temperature': features.average_temperature,
        'Day of Week': features.day_of_week,
        'Energy Consumption': features.energy_consumption,
    }])


def _best_model(model_artifact: dict) -> tuple[Any, str]:
    """Select the highest-test-accuracy model when an artifact stores several."""

    models = model_artifact.get('models')
    if isinstance(models, dict) and models:
        scores = model_artifact.get('test_accuracy', model_artifact.get('test_scores', {}))
        if isinstance(scores, dict) and scores:
            name = max(models, key=lambda key: float(scores.get(key, float('-inf'))))
        else:
            name = next(iter(models))
        return models[name], str(name)
    return model_artifact['model'], str(model_artifact.get('best_name', 'best_model'))


def _model_inputs(model_artifact: dict, row: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    feature_set = str(model_artifact.get('feature_set', 'all'))
    names = list(FEATURE_COLS.get(feature_set, FEATURE_NAMES))
    sample = row.loc[:, names].to_numpy(dtype=float)
    background_frame = pd.DataFrame(get_background_data(), columns=FEATURE_NAMES)
    background = background_frame.loc[:, names].to_numpy(dtype=float)
    return names, sample, background


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model_artifact()
    get_background_data()
    yield


app = FastAPI(title='EnergyTypeNet API', version='1.1.0', lifespan=lifespan)


@app.get('/health')
def health() -> Dict[str, str | bool]:
    return {
        'status': 'ok',
        'shap_available': is_package_available('shap'),
        'lime_available': is_package_available('lime'),
    }


@app.post('/predict')
def predict(features: BuildingFeatures) -> Dict:
    return predict_dataframe(_request_frame(features), get_model_artifact())[0]


@app.post('/predict/explain', response_model=ExplainResponse, response_model_by_alias=True)
def predict_explain(features: ExplainRequest) -> ExplainResponse:
    """Return a prediction even when optional explanation computation fails."""

    started = perf_counter()
    row = _request_frame(features)
    model_artifact = get_model_artifact()
    prediction = predict_dataframe(row, model_artifact)[0]
    shap_payload = None
    lime_payload = None
    summary = 'Prediction completed; no explanation method was requested.'

    if features.include_shap or features.include_lime:
        if explain_single_prediction is None:
            summary = 'Prediction completed, but the explainability module is unavailable.'
            LOGGER.warning('Explainability module unavailable: %s', _EXPLAINABILITY_IMPORT_ERROR)
        else:
            try:
                model, _ = _best_model(model_artifact)
                names, sample, background = _model_inputs(model_artifact, row)
                classes = list(model_artifact.get('classes', CLASS_NAMES))
                result = explain_single_prediction(
                    model=model,
                    sample=sample,
                    feature_names=names,
                    class_names=classes,
                    background_data=background,
                    task_type='classification',
                    use_shap=features.include_shap,
                    use_lime=features.include_lime,
                )
                data = result.to_dict()
                summary = result.summary_text()
                if features.include_shap and data.get('shap_values') is not None:
                    shap_payload = {
                        'feature_names': data['feature_names'],
                        'feature_values': data['feature_values'],
                        'shap_values': data['shap_values'],
                        'base_value': data['shap_base_value'],
                        'predicted_class': str(data['predicted_class']),
                        'explainer_type': data['shap_explainer_type'],
                    }
                if features.include_lime and data.get('lime_weights') is not None:
                    lime_payload = {
                        'feature_names': data['feature_names'],
                        'feature_values': data['feature_values'],
                        'lime_weights': data['lime_weights'],
                        'intercept': data['lime_intercept'],
                        'local_r2': data['lime_local_r2'],
                    }
            except Exception as exc:  # explanation must never break prediction delivery
                LOGGER.warning('Local explanation failed: %s', exc, exc_info=True)
                summary = f'Prediction completed, but the optional explanation was unavailable: {exc}'

    return ExplainResponse(
        class_name=str(prediction['class']),
        probabilities={str(key): float(value) for key, value in prediction['probabilities'].items()},
        shap_explanation=shap_payload,
        lime_explanation=lime_payload,
        summary=summary,
        computation_time=perf_counter() - started,
        explanation_available=shap_payload is not None or lime_payload is not None,
    )


@app.get('/explain/global')
def explain_global() -> dict[str, Any]:
    """Return cached global SHAP importance with per-class breakdowns."""
    global _GLOBAL_EXPLANATION_CACHE

    if not is_package_available('shap'):
        raise HTTPException(
            status_code=503,
            detail={
                'message': 'SHAP is required for global explanations.',
                'install_command': 'python -m pip install shap',
            },
        )
    if explain_dataset_globally is None:
        raise HTTPException(
            status_code=503,
            detail={'message': 'The explainability module is unavailable.'},
        )
    if _GLOBAL_EXPLANATION_CACHE is not None:
        return _GLOBAL_EXPLANATION_CACHE

    try:
        model_artifact = get_model_artifact()
        model, model_name = _best_model(model_artifact)
        feature_set = str(model_artifact.get('feature_set', 'all'))
        names = list(FEATURE_COLS.get(feature_set, FEATURE_NAMES))
        background_frame = pd.DataFrame(get_background_data(), columns=FEATURE_NAMES)
        dataset = background_frame.loc[:, names].to_numpy(dtype=float)
        classes = list(model_artifact.get('classes', CLASS_NAMES))
        importance = explain_dataset_globally(
            model=model,
            dataset=dataset,
            feature_names=names,
            class_names=classes,
        )
        feature_map: dict[str, Any] = {}
        per_class_columns = [
            column for column in importance.columns
            if str(column).startswith('mean_abs_shap_')
        ]
        for _, row in importance.iterrows():
            feature_map[str(row['feature'])] = {
                'mean_abs_shap': float(row['mean_abs_shap']),
                'per_class': {
                    str(column).removeprefix('mean_abs_shap_'): float(row[column])
                    for column in per_class_columns
                },
            }
        _GLOBAL_EXPLANATION_CACHE = {
            'model': model_name,
            'feature_importance': feature_map,
        }
        return _GLOBAL_EXPLANATION_CACHE
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.warning('Global explanation failed: %s', exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                'message': f'Global SHAP explanation is temporarily unavailable: {exc}',
                'install_command': 'python -m pip install shap',
            },
        ) from exc


# Future autoencoder endpoint:
# @app.post('/anomaly_score')
# def anomaly_score(features: BuildingFeatures) -> Dict[str, float | bool]:
#     """Return reconstruction error, anomaly threshold and anomaly flag."""
