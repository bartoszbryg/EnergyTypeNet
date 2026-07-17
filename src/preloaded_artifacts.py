"""Build and validate deployment artifacts for the bundled Streamlit demo."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import sklearn
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data import load_features
from src.models import AttentionClassifier, LogisticRegressionSoftmax


SCHEMA_VERSION = 1
DEFAULT_ARTIFACT_PATH = Path("artifacts/streamlit_preloaded.joblib")
DATA_PATHS = (
    Path("data/train_energy_data.csv"),
    Path("data/test_energy_data.csv"),
)
MODEL_NAMES = (
    "LR (sklearn)",
    "MLP",
    "XGBoost",
    "AttentionNet",
    "LR Softmax",
)


class PreloadedArtifactError(RuntimeError):
    """Raised when a precomputed artifact cannot safely be used."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_hashes() -> dict[str, str]:
    return {path.as_posix(): _file_sha256(path) for path in DATA_PATHS}


def runtime_versions() -> dict[str, str]:
    return {
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "joblib": joblib.__version__,
    }


def train_energy_models(scaler: StandardScaler) -> dict:
    """Fit the five models used by the preloaded EnergyTypeNet dashboard."""
    X_train, y_train = load_features(DATA_PATHS[0], "core")
    X_scaled = scaler.transform(X_train)

    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=10, max_iter=1000, random_state=42),
    )
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(20, 20),
            activation="relu",
            alpha=0.01,
            max_iter=1200,
            early_stopping=True,
            random_state=42,
        ),
    )
    xgb = XGBClassifier(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=100,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        verbosity=0,
        random_state=42,
    )
    attention = AttentionClassifier(w=2.0).fit(X_scaled, y_train)
    softmax = LogisticRegressionSoftmax(
        eta=0.01,
        n_iter=1000,
        alpha=0.01,
        random_state=42,
    ).fit(X_scaled, y_train)

    for model in (lr, mlp, xgb):
        model.fit(X_train, y_train)

    return {
        "LR (sklearn)": (lr, X_train, False),
        "MLP": (mlp, X_train, False),
        "XGBoost": (xgb, X_train, False),
        "AttentionNet": (attention, X_scaled, True),
        "LR Softmax": (softmax, X_scaled, True),
    }


def _comparison(models: dict, scaler: StandardScaler) -> list[dict]:
    X_train, y_train = load_features(DATA_PATHS[0], "core")
    X_test, y_test = load_features(DATA_PATHS[1], "core")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []

    for name, (model, _, is_scaled) in models.items():
        X_cv = X_train_scaled if is_scaled else X_train
        scores = cross_val_score(model, X_cv, y_train, cv=cv, scoring="accuracy")
        X_eval = X_test_scaled if is_scaled else X_test
        rows.append(
            {
                "Model": name,
                "CV Mean": float(scores.mean()),
                "CV Std": float(scores.std()),
                "Test Acc": float(accuracy_score(y_test, model.predict(X_eval))),
            }
        )

    return sorted(rows, key=lambda row: row["CV Mean"], reverse=True)


def _learning_curves(models: dict) -> dict[str, dict[str, np.ndarray]]:
    X_train, y_train = load_features(DATA_PATHS[0], "core")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    curves = {}

    for name, (model, _, _) in models.items():
        if not hasattr(model, "named_steps"):
            continue
        train_sizes, train_scores, validation_scores = learning_curve(
            model,
            X_train,
            y_train,
            cv=cv,
            train_sizes=np.linspace(0.1, 1.0, 7),
            scoring="accuracy",
            n_jobs=1,
        )
        curves[name] = {
            "train_sizes": train_sizes,
            "train_mean": train_scores.mean(axis=1),
            "train_std": train_scores.std(axis=1),
            "validation_mean": validation_scores.mean(axis=1),
            "validation_std": validation_scores.std(axis=1),
        }

    return curves


def build_artifact() -> dict:
    X_train, _ = load_features(DATA_PATHS[0], "core")
    scaler = StandardScaler().fit(X_train)
    models = train_energy_models(scaler)
    return {
        "schema_version": SCHEMA_VERSION,
        "data_hashes": data_hashes(),
        "runtime_versions": runtime_versions(),
        "models": models,
        "comparison": _comparison(models, scaler),
        "learning_curves": _learning_curves(models),
    }


def save_artifact(path: Path = DEFAULT_ARTIFACT_PATH) -> Path:
    artifact = build_artifact()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)
    return path


def load_artifact(path: Path = DEFAULT_ARTIFACT_PATH) -> dict:
    if not path.is_file():
        raise PreloadedArtifactError(f"artifact is missing: {path}")

    try:
        artifact = joblib.load(path)
    except Exception as exc:
        raise PreloadedArtifactError(f"artifact could not be loaded: {exc}") from exc

    if not isinstance(artifact, dict):
        raise PreloadedArtifactError("artifact root must be a dictionary")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise PreloadedArtifactError("artifact schema version is incompatible")
    if artifact.get("data_hashes") != data_hashes():
        raise PreloadedArtifactError("bundled dataset has changed")
    if artifact.get("runtime_versions") != runtime_versions():
        raise PreloadedArtifactError(
            "artifact library versions differ from the current runtime"
        )

    models = artifact.get("models")
    if not isinstance(models, dict) or tuple(models) != MODEL_NAMES:
        raise PreloadedArtifactError("artifact model set is incomplete or reordered")
    if not isinstance(artifact.get("comparison"), list):
        raise PreloadedArtifactError("artifact comparison results are missing")
    if not isinstance(artifact.get("learning_curves"), dict):
        raise PreloadedArtifactError("artifact learning curves are missing")

    return artifact
