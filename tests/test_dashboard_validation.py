import pandas as pd

from src.dashboard_validation import (
    recommend_classification_targets,
    validate_classification_target,
)


def test_rejects_continuous_numeric_target_with_regression_guidance():
    result = validate_classification_target(pd.Series(range(1000), name="energy"))

    assert not result.valid
    assert result.appears_continuous
    assert "appears to be continuous" in result.reason
    assert "regression" in result.reason


def test_rejects_singleton_class_before_stratified_split():
    result = validate_classification_target(pd.Series(["a"] * 20 + ["b"]))

    assert not result.valid
    assert result.min_class_count == 1
    assert "only one row" in result.reason


def test_rejects_class_too_small_for_holdout_then_five_fold_cv():
    result = validate_classification_target(pd.Series(["a"] * 20 + ["b"] * 6))

    assert not result.valid
    assert "At least 7 rows" in result.reason
    assert "5-fold" in result.reason


def test_accepts_balanced_multiclass_target():
    result = validate_classification_target(
        pd.Series(["commercial"] * 20 + ["industrial"] * 20 + ["residential"] * 20)
    )

    assert result.valid
    assert result.reason is None
    assert result.n_classes == 3


def test_recommends_target_like_categorical_column_first():
    frame = pd.DataFrame(
        {
            "Energy Consumption": range(30),
            "Region": ["north", "south", "west"] * 10,
            "Building Type": ["commercial", "industrial", "residential"] * 10,
        }
    )

    assert recommend_classification_targets(frame) == ["Building Type", "Region"]
