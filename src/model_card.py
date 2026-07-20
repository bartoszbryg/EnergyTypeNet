"""Pure-Python data collection helpers for model-card exports."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.automl import PreparedDataset


EXPORTER_VERSION = "1.0.0"
DEFAULT_DISTRIBUTION_WARNING = (
    "Model performance may differ on data from different distributions."
)


@dataclass
class ModelCardData:
    """All structured information needed to render a model card."""

    dataset_name: str | None = None
    dataset_description: str | None = None
    n_rows: int | None = None
    n_columns: int | None = None
    n_numeric_features: int | None = None
    n_categorical_features: int | None = None
    missing_cell_count: int | None = None
    duplicate_row_count: int | None = None
    target_column: str | None = None
    task_type: str | None = None
    class_names: list[str] | None = None

    feature_names: list[str] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)
    mutual_information: dict[str, float] = field(default_factory=dict)

    model_results: list[dict[str, Any]] = field(default_factory=list)
    best_model_name: str | None = None
    best_model_score: float | None = None

    per_class_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    regression_metrics: dict[str, float] = field(default_factory=dict)
    model_configuration: dict[str, Any] = field(default_factory=dict)

    chat_explanations: list[dict[str, Any]] = field(default_factory=list)
    export_timestamp: str | None = None
    exporter_version: str = EXPORTER_VERSION
    limitations: str | None = None


def collect_from_automl(
    profile: dict[str, Any] | None,
    prepared: PreparedDataset | None,
    results: pd.DataFrame | None,
    fitted_models: dict[str, Any] | None,
    feature_ranking: pd.DataFrame | None,
    dataset_name: str | None = None,
) -> ModelCardData:
    """Collect model-card data from a complete or partial AutoML run."""
    card = ModelCardData(
        dataset_name=dataset_name,
        export_timestamp=datetime.now(timezone.utc).isoformat(),
    )

    if profile:
        card.n_rows = _as_int(profile.get("n_rows"))
        card.n_columns = _as_int(profile.get("n_columns"))
        card.missing_cell_count = _as_int(profile.get("missing_cells"))
        card.duplicate_row_count = _as_int(profile.get("duplicate_rows"))

    if prepared is not None:
        card.task_type = prepared.task_type
        card.target_column = prepared.target_col
        card.feature_names = [str(name) for name in prepared.feature_cols]
        card.n_numeric_features = len(prepared.numeric_cols)
        card.n_categorical_features = len(prepared.categorical_cols)
        card.class_names = (
            [str(name) for name in prepared.classes]
            if prepared.classes is not None
            else None
        )
        if card.n_rows is None:
            card.n_rows = len(prepared.X)
        if card.n_columns is None:
            card.n_columns = len(prepared.feature_cols) + 1

    card.model_results = _result_records(results)
    score_column = _score_column(results, card.task_type)
    if results is not None and not results.empty and score_column is not None:
        scores = pd.to_numeric(results[score_column], errors="coerce")
        if scores.notna().any():
            best_index = scores.idxmax()
            best_row = results.loc[best_index]
            card.best_model_name = _model_name(best_row)
            card.best_model_score = float(scores.loc[best_index])

    card.feature_importance, card.mutual_information = _feature_scores(
        feature_ranking
    )

    if card.best_model_name and fitted_models:
        best_model = fitted_models.get(card.best_model_name)
        card.model_configuration = _model_configuration(best_model)

    if card.task_type == "regression" and card.best_model_name:
        best_result = next(
            (
                row
                for row in card.model_results
                if str(row.get("model")) == card.best_model_name
            ),
            {},
        )
        metric_columns = {
            "mse": ("test_mse", "mse"),
            "mae": ("test_mae", "mae"),
            "r_squared": ("test_r2", "r2", "r_squared"),
        }
        for output_name, candidates in metric_columns.items():
            value = _first_number(best_result, candidates)
            if value is not None:
                card.regression_metrics[output_name] = value

    card.limitations = _default_limitations(
        card.n_rows,
        card.dataset_name,
    )
    return card


def collect_from_energy_dataset(
    train_results: pd.DataFrame | None,
    fitted_models: dict[str, Any] | None,
    feature_ranking: pd.DataFrame | None,
) -> ModelCardData:
    """Collect a model card for the bundled EnergyTypeNet dataset."""
    card = collect_from_automl(
        profile=None,
        prepared=None,
        results=train_results,
        fitted_models=fitted_models,
        feature_ranking=feature_ranking,
        dataset_name="EnergyTypeNet",
    )
    card.dataset_description = (
        "A building energy type classification dataset with Residential, "
        "Commercial, and Industrial classes."
    )
    card.task_type = "classification"
    card.target_column = "Building Type"
    card.class_names = ["Residential", "Commercial", "Industrial"]
    card.feature_names = [
        "Energy Consumption",
        "Square Footage",
        "Number of Occupants",
        "Appliances Used",
        "Average Temperature",
    ]
    card.n_rows = 1000
    card.n_columns = 7
    card.n_numeric_features = 5
    card.n_categorical_features = 0
    card.missing_cell_count = 0
    card.duplicate_row_count = 0
    card.limitations = _default_limitations(card.n_rows, "EnergyTypeNet")
    return card


def add_chat_explanations(
    card: ModelCardData,
    messages: list[dict[str, Any]] | None,
) -> ModelCardData:
    """Return a copy containing only grounded assistant explanations."""
    selected: list[dict[str, Any]] = []
    pending_question: dict[str, Any] | None = None
    for message in messages or []:
        if message.get("role") == "user":
            pending_question = message
            continue
        if message.get("role") != "assistant" or message.get("grounded") is not True:
            continue
        explanation = dict(message)
        if pending_question is not None:
            explanation.setdefault("question", pending_question.get("content"))
            explanation.setdefault(
                "question_timestamp", pending_question.get("timestamp")
            )
        selected.append(explanation)
        pending_question = None
    selected.sort(key=lambda message: str(message.get("timestamp", "")))
    return replace(card, chat_explanations=selected)


def select_notable_exchanges(chat_history: Any) -> list[dict[str, Any]]:
    """Select up to five representative grounded user-assistant exchanges."""
    if chat_history is None:
        return []

    raw_messages = (
        chat_history.to_dict_list()
        if hasattr(chat_history, "to_dict_list")
        else list(chat_history)
    )
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    pending_user: dict[str, Any] | None = None
    for raw in raw_messages:
        message = dict(raw)
        if message.get("role") == "user":
            pending_user = message
        elif message.get("role") == "assistant" and pending_user is not None:
            if message.get("grounded") is True:
                pairs.append((pending_user, message))
            pending_user = None

    topic_terms = (
        "model",
        "performance",
        "accuracy",
        "score",
        "overfit",
        "feature",
        "importance",
        "predictor",
    )

    def priority(pair: tuple[dict[str, Any], dict[str, Any]]) -> tuple[int, int, str]:
        user, assistant = pair
        text = f"{user.get('content', '')} {assistant.get('content', '')}".lower()
        return (
            int(assistant.get("source") == "deterministic"),
            int(any(term in text for term in topic_terms)),
            str(assistant.get("timestamp", "")),
        )

    selected = sorted(pairs, key=priority, reverse=True)[:5]
    selected.sort(key=lambda pair: str(pair[0].get("timestamp", "")))
    return [message for pair in selected for message in pair]


def validate_model_card(card: ModelCardData) -> tuple[list[str], list[str]]:
    """Return blocking errors and non-blocking completeness warnings."""
    errors: list[str] = []
    warnings: list[str] = []

    if not card.dataset_name or not card.dataset_name.strip():
        errors.append("Dataset name is required.")
    if card.task_type not in {"classification", "regression"}:
        errors.append("Task type must be either 'classification' or 'regression'.")
    if not card.model_results:
        errors.append("At least one model result is required.")

    if not card.feature_importance:
        warnings.append(
            "Feature importance is missing; the feature analysis section will be omitted."
        )
    if not card.chat_explanations:
        warnings.append("No chat explanations are included; the explanations section will be empty.")
    if card.limitations and DEFAULT_DISTRIBUTION_WARNING in card.limitations:
        warnings.append(
            "The limitations text is auto-generated; customize it for the specific use case "
            "and data source."
        )

    return errors, warnings


def render_header(card: ModelCardData) -> str:
    """Render the document title and export metadata."""
    rows = [
        ("Export timestamp", card.export_timestamp),
        ("Task type", card.task_type),
        ("Rows", card.n_rows),
        ("Columns", card.n_columns),
        ("Exporter version", card.exporter_version),
    ]
    lines = [
        f"# {_markdown_text(card.dataset_name)} Model Card",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    lines.extend(
        f"| {_table_text(label)} | {_table_text(_display(value))} |"
        for label, value in rows
    )
    lines.extend(["", "---"])
    return "\n".join(lines)


def render_dataset_overview(card: ModelCardData) -> str:
    """Render dataset provenance, dimensions, and target information."""
    description = card.dataset_description or "No dataset description was provided."
    missing = _display(card.missing_cell_count)
    if card.missing_cell_count == 0:
        missing += " (no missing values)"
    statistics = [
        ("Total samples", card.n_rows),
        ("Total features", len(card.feature_names) if card.feature_names else None),
        ("Numeric features", card.n_numeric_features),
        ("Categorical features", card.n_categorical_features),
        ("Missing cells", missing),
        ("Duplicate rows", card.duplicate_row_count),
    ]
    lines = [
        "## Dataset Overview",
        "",
        _markdown_text(description),
        "",
        "| Statistic | Value |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {_table_text(label)} | {_table_text(_display(value))} |"
        for label, value in statistics
    )
    lines.extend(
        [
            "",
            f"- **Target column:** {_markdown_text(card.target_column)}",
            f"- **Task type:** {_markdown_text(card.task_type)}",
        ]
    )
    if card.task_type == "classification" and card.class_names:
        lines.append(
            "- **Classes:** "
            + ", ".join(_markdown_text(name) for name in card.class_names)
        )
    return "\n".join(lines)


def render_feature_analysis(card: ModelCardData) -> str:
    """Render mutual-information and model feature-importance tables."""
    lines = ["## Feature Analysis"]
    if not card.mutual_information and not card.feature_importance:
        lines.extend(["", "Feature analysis data was not collected."])
        return "\n".join(lines)

    if card.mutual_information:
        lines.extend(
            [
                "",
                "### Mutual Information",
                "",
                "| Feature | Mutual Information |",
                "|---|---:|",
            ]
        )
        for feature, score in _sorted_scores(card.mutual_information):
            lines.append(f"| {_table_text(feature)} | {score:.4f} |")
        lines.extend(
            [
                "",
                "*Mutual information is computed against the target column and "
                "represents non-linear association strength.*",
            ]
        )

    if card.feature_importance:
        lines.extend(
            [
                "",
                "### Best-Model Feature Importance",
                "",
                "| Feature | Importance |",
                "|---|---:|",
            ]
        )
        for feature, score in _sorted_scores(card.feature_importance):
            lines.append(f"| {_table_text(feature)} | {score:.4f} |")
    return "\n".join(lines)


def render_model_results(card: ModelCardData) -> str:
    """Render model comparison and task-specific evaluation details."""
    best_score = _format_number(card.best_model_score)
    lines = [
        "## Model Evaluation Results",
        "",
        f"The best-performing model is **{_markdown_text(card.best_model_name)}** "
        f"with a cross-validation score of **{best_score}**.",
        "",
    ]
    if card.task_type == "regression":
        lines.extend(
            [
                "| Model Name | CV R² | Test R² | Test MAE |",
                "|---|---:|---:|---:|",
            ]
        )
        for result in card.model_results:
            name = _result_value(result, "model", "Model", "name")
            rendered_name = _best_model_cell(name, card.best_model_name)
            lines.append(
                f"| {rendered_name} | "
                f"{_format_number(_result_value(result, 'cv_r2', 'CV R2', 'cv_score'))} | "
                f"{_format_number(_result_value(result, 'test_r2', 'r2', 'test_score'))} | "
                f"{_format_number(_result_value(result, 'test_mae', 'mae'))} |"
            )
    else:
        lines.extend(
            [
                "| Model Name | CV Accuracy | Test Accuracy | Test F1 Macro |",
                "|---|---:|---:|---:|",
            ]
        )
        for result in card.model_results:
            name = _result_value(result, "model", "Model", "name")
            rendered_name = _best_model_cell(name, card.best_model_name)
            lines.append(
                f"| {rendered_name} | "
                f"{_format_number(_result_value(result, 'cv_accuracy', 'CV Mean', 'cv_score'))} | "
                f"{_format_number(_result_value(result, 'test_accuracy', 'Test Acc', 'test_score'))} | "
                f"{_format_number(_result_value(result, 'test_f1_macro', 'f1_macro'))} |"
            )

    lines.extend(["", "### Evaluation Details"])
    if card.task_type == "classification" and card.per_class_metrics:
        lines.extend(
            [
                "",
                "| Class | Precision | Recall | F1 |",
                "|---|---:|---:|---:|",
            ]
        )
        for class_name, metrics in card.per_class_metrics.items():
            lines.append(
                f"| {_table_text(class_name)} | "
                f"{_format_number(metrics.get('precision'))} | "
                f"{_format_number(metrics.get('recall'))} | "
                f"{_format_number(metrics.get('f1', metrics.get('f1_score')))} |"
            )
    elif card.task_type == "regression":
        metrics = card.regression_metrics
        lines.extend(
            [
                "",
                "Regression summary: "
                f"MSE = **{_format_number(metrics.get('mse'))}**, "
                f"MAE = **{_format_number(metrics.get('mae'))}**, and "
                f"R² = **{_format_number(metrics.get('r_squared'))}**.",
            ]
        )
    else:
        lines.extend(["", "Detailed evaluation metrics were not captured for this export."])
    return "\n".join(lines)


def render_model_configuration(card: ModelCardData) -> str:
    """Render the best model's captured configuration."""
    lines = ["## Model Configuration"]
    if not card.model_configuration:
        lines.extend(["", "Configuration details were not captured for this export."])
        return "\n".join(lines)

    lines.extend(["", "```yaml"])
    for name in sorted(card.model_configuration):
        lines.append(
            f"{_yaml_scalar(name)}: {_yaml_scalar(card.model_configuration[name])}"
        )
    lines.append("```")
    return "\n".join(lines)


def render_limitations(card: ModelCardData) -> str:
    """Render intended-use limitations and standard caveats."""
    limitations = card.limitations or "No limitations were provided."
    if _is_default_limitations(limitations):
        limitations_text = "\n".join(
            f"> {line}" if line else ">" for line in limitations.splitlines()
        )
    else:
        limitations_text = _markdown_text(limitations)
    return "\n".join(
        [
            "## Intended Use and Limitations",
            "",
            limitations_text,
            "",
            "### Caveats",
            "",
            "- The model was evaluated on a held-out test set from the same "
            "distribution as the training data; performance on different "
            "distributions is unknown.",
            "- Feature importance reflects associations in the training data and "
            "must not be interpreted as a causal relationship.",
            "- Model selection uses cross-validation performance, which may not "
            "align with the business metric that matters for a specific deployment.",
        ]
    )


def render_chat_explanations(card: ModelCardData) -> str:
    """Render selected grounded dataset-assistant exchanges when present."""
    if not card.chat_explanations:
        return ""
    lines = [
        "## AI-Assisted Dataset Analysis",
        "",
        "These explanations were generated by the dataset assistant during an "
        "analysis session and are grounded in the computed statistics shown in "
        "the other sections of this model card.",
    ]
    exchanges = _chat_exchanges(card.chat_explanations)
    for index, (question, answer) in enumerate(exchanges):
        if index:
            lines.extend(["", "---"])
        timestamp = answer.get("timestamp") or question.get("timestamp")
        lines.extend(
            [
                "",
                f"*Question: {_markdown_text(question.get('content') or 'Question not included')}*",
                "",
                f"**Answer:** {_markdown_text(answer.get('content'))}",
                "",
                f"<small>Timestamp: {_markdown_text(timestamp or 'not recorded')}</small>",
            ]
        )
    return "\n".join(lines)


def render_full_model_card(card: ModelCardData) -> str:
    """Validate and render a complete GitHub-friendly Markdown model card."""
    errors, warnings = validate_model_card(card)
    if errors:
        details = "; ".join(errors)
        raise ValueError(
            f"The model card cannot be rendered until these errors are fixed: {details}"
        )

    sections = [
        render_header(card),
        render_dataset_overview(card),
        render_feature_analysis(card),
        render_model_results(card),
        render_model_configuration(card),
        render_limitations(card),
        render_chat_explanations(card),
    ]
    if warnings:
        sections.append(
            "\n".join(
                ["## Export Notes", ""] + [f"- {warning}" for warning in warnings]
            )
        )
    return "\n\n".join(section for section in sections if section).rstrip() + "\n"


def save_model_card(card: ModelCardData, output_path: str) -> str:
    """Validate, render, and save a UTF-8 Markdown model card."""
    markdown = render_full_model_card(card)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return str(path)


def markdown_to_html(markdown_text: str) -> str:
    """Convert Markdown to a complete, print-friendly HTML document."""
    try:
        import markdown as markdown_package
    except ImportError as exc:
        raise ImportError(
            "HTML and PDF model-card export requires the optional 'markdown' "
            "package. Install it with: pip install markdown"
        ) from exc

    body = markdown_package.markdown(
        markdown_text,
        extensions=["tables", "fenced_code"],
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model Card</title>
  <style>
    @page {{ size: A4; margin: 20mm; }}
    * {{ box-sizing: border-box; }}
    body {{
      max-width: 800px;
      margin: 0 auto;
      color: #20242a;
      font-family: Georgia, "Times New Roman", serif;
      font-size: 11pt;
      line-height: 1.55;
    }}
    h1 {{
      color: #172b4d;
      border-bottom: 2px solid #355b85;
      padding-bottom: 0.35rem;
      font-size: 2rem;
    }}
    h2 {{ color: #243b53; font-size: 1.45rem; margin-top: 1.8rem; }}
    h3 {{ color: #334e68; font-size: 1.15rem; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 0.8rem 0 1.2rem;
      font-size: 0.92rem;
    }}
    th, td {{ border: 1px solid #cbd2d9; padding: 0.45rem 0.6rem; }}
    th {{ background: #e9edf2; text-align: left; }}
    tbody tr:nth-child(even) {{ background: #f7f8fa; }}
    code, pre {{ font-family: Consolas, Monaco, monospace; }}
    pre {{
      background: #f1f3f5;
      border: 1px solid #d8dee4;
      border-radius: 4px;
      padding: 0.8rem;
      white-space: pre-wrap;
    }}
    blockquote {{
      margin: 1rem 0;
      padding: 0.65rem 1rem;
      border-left: 4px solid #7895b2;
      background: #f3f6f9;
      color: #46596b;
    }}
    hr {{ border: 0; border-top: 1px solid #cbd2d9; margin: 1.5rem 0; }}
    small {{ color: #687784; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def export_model_card_pdf(card: ModelCardData, output_path: str) -> str:
    """Render and write a model card as a PDF document."""
    path = Path(output_path)
    if path.suffix.lower() != ".pdf":
        raise ValueError("PDF model-card output path must end in '.pdf'.")
    markdown_text = render_full_model_card(card)
    html = markdown_to_html(markdown_text)
    html_class = _load_weasyprint_html()
    path.parent.mkdir(parents=True, exist_ok=True)
    html_class(string=html).write_pdf(str(path))
    return str(path)


def export_model_card_bytes(card: ModelCardData, format: str) -> bytes:
    """Render a model card to Markdown or PDF bytes for browser downloads."""
    normalized_format = format.lower().strip()
    markdown_text = render_full_model_card(card)
    if normalized_format == "markdown":
        return markdown_text.encode("utf-8")
    if normalized_format != "pdf":
        raise ValueError("Model-card format must be either 'markdown' or 'pdf'.")
    html = markdown_to_html(markdown_text)
    html_class = _load_weasyprint_html()
    pdf_bytes = html_class(string=html).write_pdf()
    if not isinstance(pdf_bytes, bytes):
        raise RuntimeError("WeasyPrint did not return PDF bytes.")
    return pdf_bytes


def _load_weasyprint_html() -> Any:
    try:
        from weasyprint import HTML
    except (ImportError, OSError) as exc:
        raise ImportError(
            "PDF model-card export requires the optional 'weasyprint' package. "
            "Install it with: pip install weasyprint. On some systems WeasyPrint "
            "also requires system packages such as libpango on Linux or suitable "
            "fonts on macOS."
        ) from exc
    return HTML


def _display(value: Any) -> str:
    return "Not available" if value is None or value == "" else str(value)


def _markdown_text(value: Any) -> str:
    return _display(value).replace("\r\n", "\n").replace("\r", "\n")


def _table_text(value: Any) -> str:
    return _markdown_text(value).replace("|", "\\|").replace("\n", "<br>")


def _format_number(value: Any) -> str:
    number = _as_float(value)
    return "Not available" if number is None else f"{number:.4f}"


def _sorted_scores(scores: dict[str, float]) -> list[tuple[str, float]]:
    normalized = [
        (str(feature), value)
        for feature, raw_value in scores.items()
        if (value := _as_float(raw_value)) is not None
    ]
    return sorted(normalized, key=lambda item: (-item[1], item[0].lower()))


def _result_value(result: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = result.get(key)
        if value is not None:
            return value
    return None


def _best_model_cell(name: Any, best_model_name: str | None) -> str:
    rendered = _table_text(name)
    if name is not None and best_model_name is not None and str(name) == best_model_name:
        return f"**{rendered}**"
    return rendered


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_yaml_scalar(item) for item in value) + "]"
    if isinstance(value, dict):
        entries = ", ".join(
            f"{_yaml_scalar(key)}: {_yaml_scalar(item)}"
            for key, item in value.items()
        )
        return "{" + entries + "}"
    text = str(value)
    if text and all(character.isalnum() or character in "_-./" for character in text):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _is_default_limitations(text: str | None) -> bool:
    return bool(text and DEFAULT_DISTRIBUTION_WARNING in text)


def _chat_exchanges(
    messages: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    exchanges: list[tuple[dict[str, Any], dict[str, Any]]] = []
    pending_user: dict[str, Any] | None = None
    for raw_message in messages:
        message = dict(raw_message)
        if message.get("role") == "user":
            pending_user = message
        elif message.get("role") == "assistant":
            question_text = message.get("question")
            question = pending_user or {
                "role": "user",
                "content": question_text or "Question not included in export",
                "timestamp": message.get("timestamp"),
            }
            exchanges.append((question, message))
            pending_user = None
    return exchanges


def _result_records(results: pd.DataFrame | None) -> list[dict[str, Any]]:
    if results is None or results.empty:
        return []
    records = []
    for raw_row in results.to_dict(orient="records"):
        row = {str(key): _python_value(value) for key, value in raw_row.items()}
        model = next(
            (row.get(name) for name in ("model", "Model", "name") if row.get(name) is not None),
            None,
        )
        cv_score = next(
            (
                row.get(name)
                for name in ("cv_accuracy", "cv_r2", "CV Mean", "cv_mean")
                if row.get(name) is not None
            ),
            None,
        )
        test_score = next(
            (
                row.get(name)
                for name in ("test_accuracy", "test_r2", "Test Acc", "test_score")
                if row.get(name) is not None
            ),
            None,
        )
        row["model"] = str(model) if model is not None else None
        row["cv_score"] = _as_float(cv_score)
        row["test_score"] = _as_float(test_score)
        records.append(row)
    return records


def _score_column(results: pd.DataFrame | None, task_type: str | None) -> str | None:
    if results is None:
        return None
    preferred = (
        ("cv_accuracy", "CV Mean", "cv_mean")
        if task_type != "regression"
        else ("cv_r2", "CV R2", "cv_mean")
    )
    return next((column for column in preferred if column in results.columns), None)


def _model_name(row: pd.Series) -> str | None:
    for column in ("model", "Model", "name"):
        if column in row and pd.notna(row[column]):
            return str(row[column])
    return None


def _feature_scores(
    ranking: pd.DataFrame | None,
) -> tuple[dict[str, float], dict[str, float]]:
    if ranking is None or ranking.empty or "feature" not in ranking.columns:
        return {}, {}
    importance_column = next(
        (
            name
            for name in ("feature_importance", "importance", "quality_score")
            if name in ranking.columns
        ),
        None,
    )
    mi_column = (
        "mutual_information" if "mutual_information" in ranking.columns else None
    )

    def mapping(column: str | None) -> dict[str, float]:
        if column is None:
            return {}
        output: dict[str, float] = {}
        for _, row in ranking.iterrows():
            value = _as_float(row.get(column))
            if value is not None:
                output[str(row["feature"])] = value
        return output

    mutual_information = mapping(mi_column)
    feature_importance = mapping(importance_column)
    if not feature_importance:
        feature_importance = dict(mutual_information)
    return feature_importance, mutual_information


def _model_configuration(model: Any) -> dict[str, Any]:
    if model is None or not hasattr(model, "get_params"):
        return {}
    try:
        return {
            str(key): _python_value(value)
            for key, value in model.get_params(deep=False).items()
        }
    except (AttributeError, TypeError, ValueError):
        return {}


def _default_limitations(n_rows: int | None, dataset_name: str | None) -> str:
    size = f"{n_rows:,} rows" if n_rows is not None else "an unspecified number of rows"
    domain = dataset_name or "the uploaded dataset's domain"
    return (
        f"This model was trained on {size} from {domain}. "
        f"{DEFAULT_DISTRIBUTION_WARNING} Validate performance, representation, and "
        "potential bias for the intended population before deployment."
    )


def _first_number(row: dict[str, Any], candidates: tuple[str, ...]) -> float | None:
    for candidate in candidates:
        value = _as_float(row.get(candidate))
        if value is not None:
            return value
    return None


def _as_int(value: Any) -> int | None:
    try:
        return None if value is None or pd.isna(value) else int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None or pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return None


def _python_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, list, dict, tuple)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return repr(value)
