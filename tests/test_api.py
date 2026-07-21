from fastapi.testclient import TestClient
import numpy as np

import src.api as api
from src.api import app


class DummyModel:
    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.2, 0.7, 0.1]])


def test_health_endpoint():
    client = TestClient(app)

    response = client.get('/health')

    assert response.status_code == 200
    result = response.json()
    assert result['status'] == 'ok'
    assert isinstance(result['shap_available'], bool)
    assert isinstance(result['lime_available'], bool)


def test_predict_endpoint_returns_probabilities(monkeypatch):
    monkeypatch.setattr(
        api,
        'artifact',
        {
            'feature_set': 'core',
            'model': DummyModel(),
            'classes': ['Residential', 'Commercial', 'Industrial'],
        },
    )

    client = TestClient(app)
    payload = {
        'square_footage': 25000,
        'number_of_occupants': 20,
        'appliances_used': 30,
        'average_temperature': 72,
        'day_of_week': 'Weekday',
        'energy_consumption': 4100,
    }

    response = client.post('/predict', json=payload)
    result = response.json()

    assert response.status_code == 200
    assert result['class'] in {'Residential', 'Commercial', 'Industrial'}
    assert set(result['probabilities']) == {'Residential', 'Commercial', 'Industrial'}
