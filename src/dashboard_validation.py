"""Validation helpers for the Streamlit custom-classification workflow."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ClassificationTargetValidation:
    """Result of validating a column for stratified classification."""

    valid: bool
    reason: str | None
    appears_continuous: bool
    n_classes: int
    min_class_count: int


def _class_limit(n_rows: int) -> int:
    """Keep charts/models useful while allowing ordinary multiclass datasets."""
    return min(50, max(10, int(n_rows * 0.20)))


def validate_classification_target(
    series: pd.Series,
    *,
    cv_splits: int = 5,
) -> ClassificationTargetValidation:
    """Validate a target before stratified holdout splitting and cross-validation."""
    clean = series.dropna()
    n_rows = len(clean)
    counts = clean.value_counts(dropna=False)
    n_classes = int(len(counts))
    min_count = int(counts.min()) if n_classes else 0
    unique_ratio = n_classes / n_rows if n_rows else 0.0
    appears_continuous = bool(
        pd.api.types.is_numeric_dtype(clean)
        and n_classes >= 20
        and unique_ratio > 0.10
    )

    if n_rows == 0:
        reason = "The selected target has no non-missing values."
    elif n_classes < 2:
        reason = "The selected target must contain at least two classes."
    elif appears_continuous:
        reason = (
            f"The selected numeric target has {n_classes:,} unique values across "
            f"{n_rows:,} rows and appears to be continuous. This dashboard mode "
            "performs classification; use a categorical target instead. A future "
            "regression workflow would be appropriate for this column."
        )
    elif n_classes > _class_limit(n_rows):
        reason = (
            f"The selected target has {n_classes:,} classes, which is too many for "
            "this interactive classification workflow. Choose a lower-cardinality "
            "categorical target."
        )
    elif min_count < 2:
        reason = (
            "At least one class has only one row, so a stratified train/test split "
            "cannot place that class in both sets."
        )
    elif min_count < cv_splits + 2:
        reason = (
            f"The smallest class has {min_count} rows. At least {cv_splits + 2} rows "
            f"per class are required to support an 80/20 stratified split followed "
            f"by {cv_splits}-fold cross-validation."
        )
    else:
        reason = None

    return ClassificationTargetValidation(
        valid=reason is None,
        reason=reason,
        appears_continuous=appears_continuous,
        n_classes=n_classes,
        min_class_count=min_count,
    )


def recommend_classification_targets(
    df: pd.DataFrame,
    *,
    cv_splits: int = 5,
) -> list[str]:
    """Return valid target columns, preferring categorical and target-like names."""
    candidates = []
    target_words = ("target", "label", "class", "type", "category", "outcome")

    for position, column in enumerate(df.columns):
        validation = validate_classification_target(df[column], cv_splits=cv_splits)
        if not validation.valid:
            continue
        is_categorical = not pd.api.types.is_numeric_dtype(df[column])
        has_target_name = any(word in str(column).lower() for word in target_words)
        candidates.append(
            (
                0 if has_target_name else 1,
                0 if is_categorical else 1,
                validation.n_classes,
                position,
                column,
            )
        )

    candidates.sort()
    return [column for *_score, column in candidates]
