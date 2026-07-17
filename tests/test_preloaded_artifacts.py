from pathlib import Path

import joblib
import pytest

from src.preloaded_artifacts import (
    DEFAULT_ARTIFACT_PATH,
    MODEL_NAMES,
    PreloadedArtifactError,
    data_hashes,
    load_artifact,
    runtime_versions,
)


def test_bundled_streamlit_artifact_is_valid_and_complete():
    artifact = load_artifact(DEFAULT_ARTIFACT_PATH)

    assert artifact["data_hashes"] == data_hashes()
    assert artifact["runtime_versions"] == runtime_versions()
    assert tuple(artifact["models"]) == MODEL_NAMES
    assert {row["Model"] for row in artifact["comparison"]} == set(MODEL_NAMES)
    assert set(artifact["learning_curves"]) == {
        "LR (sklearn)",
        "MLP",
    }


def test_bundled_models_can_predict_after_deserialization():
    artifact = load_artifact(DEFAULT_ARTIFACT_PATH)

    for model, training_data, _is_scaled in artifact["models"].values():
        sample = training_data[:2]
        assert model.predict(sample).shape == (2,)
        assert model.predict_proba(sample).shape == (2, 3)


def test_missing_artifact_uses_guarded_error(tmp_path: Path):
    with pytest.raises(PreloadedArtifactError, match="artifact is missing"):
        load_artifact(tmp_path / "missing.joblib")


def test_incompatible_schema_uses_guarded_error(tmp_path: Path):
    path = tmp_path / "incompatible.joblib"
    joblib.dump({"schema_version": -1}, path)

    with pytest.raises(PreloadedArtifactError, match="schema version"):
        load_artifact(path)
