"""Focused tests for model explainability helpers and API fallbacks."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import src.api as api
import src.explainability as explainability
from src.api import ExplainResponse, app
from src.explainability import (
    ExplanationResult,
    SHAPExplainerType,
    compare_shap_lime,
    detect_explainer_type,
    explain_single_prediction,
    extract_pipeline_parts,
)


@pytest.fixture
def classification_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def _result(*, shap_values=None, lime_weights=None):
    return ExplanationResult(
        sample_index=0,
        feature_names=['a', 'b'],
        feature_values=[1.0, 2.0],
        predicted_class='yes',
        predicted_probability=0.8,
        class_names=['no', 'yes'],
        shap_values=shap_values,
        shap_base_value=0.2,
        shap_explainer_type='linear',
        lime_weights=lime_weights,
        lime_intercept=0.1,
        lime_local_r2=0.9,
        timestamp='2026-07-20T00:00:00+00:00',
    )


def test_detect_tree_explainer(classification_data):
    X, y = classification_data
    assert detect_explainer_type(RandomForestClassifier().fit(X, y)) is SHAPExplainerType.TREE


def test_detect_linear_explainer(classification_data):
    X, y = classification_data
    assert detect_explainer_type(LogisticRegression().fit(X, y)) is SHAPExplainerType.LINEAR


def test_detect_kernel_explainer(classification_data):
    X, y = classification_data
    assert detect_explainer_type(MLPClassifier(max_iter=10).fit(X, y)) is SHAPExplainerType.KERNEL


def test_detect_xgboost_by_module_name():
    FakeXGB = type('XGBClassifier', (), {'__module__': 'xgboost.sklearn'})
    assert detect_explainer_type(FakeXGB()) is SHAPExplainerType.XGBOOST


def test_extract_pipeline_parts_applies_preprocessing(classification_data):
    X, y = classification_data
    pipeline = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
    estimator, transform = extract_pipeline_parts(pipeline)
    assert isinstance(estimator, LogisticRegression)
    assert transform(X[:2]).shape == (2, 4)


def test_extract_raw_estimator_uses_identity(classification_data):
    X, y = classification_data
    model = LogisticRegression().fit(X, y)
    estimator, transform = extract_pipeline_parts(model)
    assert estimator is model
    assert np.array_equal(transform(X[:2]), X[:2])


def test_explanation_result_serialises():
    payload = _result(shap_values=[0.4, -0.2]).to_dict()
    assert payload['predicted_class'] == 'yes'
    assert payload['shap_values'] == [0.4, -0.2]


def test_explanation_result_summary_mentions_prediction():
    assert 'yes' in _result(shap_values=[0.4, -0.2]).summary_text()


def test_compare_shap_lime_agreement():
    frame = compare_shap_lime(_result(shap_values=[0.4, -0.2], lime_weights=[0.3, -0.1]))
    assert frame['sign_agreement'].tolist() == [True, True]


def test_compare_shap_lime_missing_values():
    frame = compare_shap_lime(_result())
    assert frame['shap_value'].isna().all()
    assert frame['lime_weight'].isna().all()


def test_single_prediction_survives_missing_shap(monkeypatch, classification_data):
    X, y = classification_data
    model = LogisticRegression().fit(X, y)
    monkeypatch.setattr(explainability, '_import_shap', lambda: (_ for _ in ()).throw(ImportError('missing')))
    result = explain_single_prediction(
        model, X[:1], ['a', 'b', 'c', 'd'], ['no', 'yes'], X[:20],
        task_type='classification', use_shap=True, use_lime=False,
    )
    assert result.predicted_class in {'no', 'yes'}
    assert result.shap_values is None


def test_lime_supports_mixed_dataframe_pipeline(monkeypatch):
    class FakeExplanation:
        local_exp = {1: [(0, 0.25), (1, -0.1)]}
        intercept = {1: 0.4}
        local_pred = np.array([0.6])
        score = 0.9

    class FakeLimeExplainer:
        def __init__(self, **kwargs):
            self.feature_names = kwargs['feature_names']

        def explain_instance(self, row, prediction_function, **kwargs):
            perturbed = np.vstack([row, [75, 1]])
            probabilities = prediction_function(perturbed)
            assert probabilities.shape == (2, 2)
            return FakeExplanation()

    monkeypatch.setattr(
        explainability,
        '_import_lime_tabular',
        lambda: SimpleNamespace(LimeTabularExplainer=FakeLimeExplainer),
    )
    X = pd.DataFrame({
        'temperature': [60, 65, 70, 75, 80, 85, 55, 68],
        'building': ['office', 'home', 'office', 'shop', 'home', 'shop', 'home', 'office'],
    })
    y = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    preprocessing = ColumnTransformer([
        ('numeric', StandardScaler(), ['temperature']),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), ['building']),
    ])
    model = make_pipeline(preprocessing, LogisticRegression()).fit(X, y)

    result = explain_single_prediction(
        model,
        X.iloc[[2]],
        list(X.columns),
        ['low', 'high'],
        X,
        task_type='classification',
        use_shap=False,
        use_lime=True,
    )

    assert result.lime_weights is not None
    assert len(result.lime_weights) == 2
    assert result.feature_values == [70, 'office']


def test_explain_response_accepts_alias():
    response = ExplainResponse(
        **{'class': 'Commercial'}, probabilities={'Commercial': 1.0},
        summary='ok', computation_time=0.01, explanation_available=False,
    )
    assert response.class_name == 'Commercial'


def test_health_reports_optional_packages():
    result = TestClient(app).get('/health').json()
    assert isinstance(result['shap_available'], bool)
    assert isinstance(result['lime_available'], bool)


class _DummyModel:
    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.2, 0.7, 0.1]])


@pytest.fixture
def api_artifact(monkeypatch):
    artifact = {
        'feature_set': 'core',
        'model': _DummyModel(),
        'classes': ['Residential', 'Commercial', 'Industrial'],
    }
    monkeypatch.setattr(api, 'artifact', artifact)
    monkeypatch.setattr(api, 'BACKGROUND_DATA', np.ones((100, len(api.FEATURE_NAMES))))
    monkeypatch.setattr(api, '_GLOBAL_EXPLANATION_CACHE', None)
    return artifact


def _payload(**extra):
    return {
        'square_footage': 25000,
        'number_of_occupants': 20,
        'appliances_used': 30,
        'average_temperature': 72,
        'day_of_week': 'Weekday',
        'energy_consumption': 4100,
        **extra,
    }


def test_explained_prediction_returns_prediction_when_explanation_fails(monkeypatch, api_artifact):
    monkeypatch.setattr(api, 'explain_single_prediction', lambda **kwargs: (_ for _ in ()).throw(RuntimeError('boom')))
    response = TestClient(app).post('/predict/explain', json=_payload())
    assert response.status_code == 200
    assert response.json()['class'] == 'Commercial'
    assert response.json()['explanation_available'] is False


def test_explained_prediction_without_methods(api_artifact):
    response = TestClient(app).post(
        '/predict/explain', json=_payload(include_shap=False, include_lime=False),
    )
    assert response.status_code == 200
    assert response.json()['explanation_available'] is False


def test_explained_prediction_maps_shap_payload(monkeypatch, api_artifact):
    monkeypatch.setattr(api, 'explain_single_prediction', lambda **kwargs: _result(shap_values=[0.4, -0.2]))
    response = TestClient(app).post('/predict/explain', json=_payload())
    assert response.status_code == 200
    assert response.json()['explanation_available'] is True
    assert response.json()['shap_explanation']['explainer_type'] == 'linear'


def test_global_explanation_missing_shap_returns_503(monkeypatch, api_artifact):
    monkeypatch.setattr(api, 'is_package_available', lambda package: False)
    response = TestClient(app).get('/explain/global')
    assert response.status_code == 503
    assert 'python -m pip install shap' in response.json()['detail']['install_command']


def test_global_explanation_is_cached(monkeypatch, api_artifact):
    monkeypatch.setattr(api, 'is_package_available', lambda package: True)
    calls = {'count': 0}

    def fake_global(**kwargs):
        calls['count'] += 1
        return pd.DataFrame({
            'feature': ['Square Footage', 'Energy Consumption'],
            'mean_abs_shap': [0.8, 0.2],
            'mean_abs_shap_Residential': [0.7, 0.1],
        })

    monkeypatch.setattr(api, 'explain_dataset_globally', fake_global)
    client = TestClient(app)
    assert client.get('/explain/global').status_code == 200
    assert client.get('/explain/global').status_code == 200
    assert calls['count'] == 1


def test_global_explanation_failure_returns_503(monkeypatch, api_artifact):
    monkeypatch.setattr(api, 'is_package_available', lambda package: True)
    monkeypatch.setattr(api, 'explain_dataset_globally', lambda **kwargs: (_ for _ in ()).throw(RuntimeError('bad')))
    response = TestClient(app).get('/explain/global')
    assert response.status_code == 503
    assert 'temporarily unavailable' in response.json()['detail']['message']
