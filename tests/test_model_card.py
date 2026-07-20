"""Tests for model-card collection, rendering, and optional export paths."""

import sys

import numpy as np
import pandas as pd
import pytest

from src.automl import PreparedDataset, profile_dataset
from src.chat_agent import ChatHistory, ChatMessage
from src.model_card import (
    ModelCardData,
    add_chat_explanations,
    collect_from_automl,
    export_model_card_bytes,
    render_full_model_card,
    select_notable_exchanges,
    validate_model_card,
)


def _prepared() -> PreparedDataset:
    return PreparedDataset(
        X=pd.DataFrame({"size": [10.0, 20.0, 30.0, 40.0]}),
        y=np.array([0, 0, 1, 1]),
        task_type="classification",
        target_col="label",
        feature_cols=["size"],
        numeric_cols=["size"],
        categorical_cols=[],
        classes=["no", "yes"],
    )


def _valid_card(**overrides) -> ModelCardData:
    values = {
        "dataset_name": "Synthetic",
        "task_type": "classification",
        "model_results": [
            {
                "model": "Logistic Regression",
                "cv_accuracy": 0.875,
                "test_accuracy": 0.75,
                "test_f1_macro": 0.7333,
            }
        ],
        "best_model_name": "Logistic Regression",
        "best_model_score": 0.875,
        "limitations": "Custom limitations for this synthetic example.",
    }
    values.update(overrides)
    return ModelCardData(**values)


def test_model_card_data_defaults():
    card = ModelCardData()

    assert card.dataset_name is None
    assert card.task_type is None
    assert card.class_names is None
    assert card.feature_names == []
    assert card.feature_importance == {}
    assert card.mutual_information == {}
    assert card.model_results == []
    assert card.per_class_metrics == {}
    assert card.regression_metrics == {}
    assert card.model_configuration == {}
    assert card.chat_explanations == []
    assert card.exporter_version == "1.0.0"


def test_collect_from_automl_populates_core_fields():
    prepared = _prepared()
    profile = profile_dataset(
        pd.DataFrame({"size": [10, 20, 30, 40], "label": ["no", "no", "yes", "yes"]})
    )
    results = pd.DataFrame(
        [{"model": "Demo", "cv_accuracy": 0.8, "test_accuracy": 0.75}]
    )
    ranking = pd.DataFrame([{"feature": "size", "mutual_information": 0.4}])

    card = collect_from_automl(
        profile, prepared, results, {}, ranking, dataset_name="Tiny data"
    )

    assert card.dataset_name == "Tiny data"
    assert card.task_type == "classification"
    assert card.n_rows == 4
    assert card.n_columns == 2
    assert card.model_results[0]["model"] == "Demo"
    assert card.model_results[0]["cv_score"] == pytest.approx(0.8)


def test_collect_from_automl_handles_none_results():
    card = collect_from_automl(None, None, None, None, None, "Partial")

    assert card.dataset_name == "Partial"
    assert card.model_results == []


def test_add_chat_explanations_keeps_only_grounded_assistant_messages():
    messages = [
        {"role": "user", "content": "Which model?", "grounded": False},
        {
            "role": "assistant",
            "content": "Computed answer",
            "grounded": True,
            "timestamp": "2026-01-01T00:00:00+00:00",
        },
        {"role": "assistant", "content": "Speculation", "grounded": False},
    ]

    updated = add_chat_explanations(_valid_card(), messages)

    assert len(updated.chat_explanations) == 1
    assert updated.chat_explanations[0]["role"] == "assistant"
    assert updated.chat_explanations[0]["grounded"] is True
    assert updated.chat_explanations[0]["question"] == "Which model?"


def test_select_notable_exchanges_returns_at_most_five_complete_pairs():
    history = ChatHistory(max_turns=10)
    for index in range(7):
        history.add(ChatMessage(role="user", content=f"Model question {index}"))
        history.add(
            ChatMessage(
                role="assistant",
                content=f"Grounded model answer {index}",
                source="deterministic",
                grounded=True,
            )
        )

    selected = select_notable_exchanges(history)

    assert len(selected) <= 10
    assert len(selected) % 2 == 0
    assert all(selected[index]["role"] == "user" for index in range(0, len(selected), 2))
    assert all(
        selected[index]["role"] == "assistant"
        for index in range(1, len(selected), 2)
    )


@pytest.mark.parametrize(
    ("card", "expected"),
    [
        (_valid_card(dataset_name=""), "Dataset name"),
        (_valid_card(task_type="clustering"), "Task type"),
        (_valid_card(model_results=[]), "model result"),
    ],
)
def test_validate_model_card_returns_required_errors(card, expected):
    errors, _ = validate_model_card(card)

    assert any(expected.lower() in error.lower() for error in errors)


def test_validate_model_card_warns_when_chat_is_missing():
    _, warnings = validate_model_card(_valid_card())

    assert any("chat explanations" in warning.lower() for warning in warnings)


def test_render_full_model_card_returns_markdown_with_heading_and_results():
    markdown = render_full_model_card(_valid_card())

    assert markdown
    assert "# Synthetic Model Card" in markdown
    assert "Logistic Regression" in markdown


def test_render_full_model_card_rejects_invalid_card():
    with pytest.raises(ValueError, match="cannot be rendered"):
        render_full_model_card(ModelCardData())


def test_ai_explanations_section_is_omitted_when_empty():
    markdown = render_full_model_card(_valid_card(chat_explanations=[]))

    assert "AI-Assisted Dataset Analysis" not in markdown


def test_ai_explanations_section_is_rendered_when_present():
    explanations = [
        {
            "role": "assistant",
            "content": "The measured result supports this answer.",
            "question": "What is the best model?",
            "grounded": True,
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
    ]

    markdown = render_full_model_card(_valid_card(chat_explanations=explanations))

    assert "## AI-Assisted Dataset Analysis" in markdown
    assert "What is the best model?" in markdown


def test_markdown_byte_export_matches_renderer():
    card = _valid_card()

    exported = export_model_card_bytes(card, format="markdown")

    assert isinstance(exported, bytes)
    assert exported.decode("utf-8") == render_full_model_card(card)


def test_pdf_export_reports_missing_weasyprint(monkeypatch):
    import src.model_card as model_card

    monkeypatch.setattr(model_card, "markdown_to_html", lambda _: "<html></html>")
    monkeypatch.setitem(sys.modules, "weasyprint", None)

    with pytest.raises(ImportError, match="pip install weasyprint"):
        export_model_card_bytes(_valid_card(), format="pdf")
