"""Fast unit coverage for schema, drift, and leakage validation."""

import numpy as np
import pandas as pd

from src.data_validation import (
    ValidationIssue,
    ValidationReport,
    compute_categorical_drift,
    compute_numeric_drift,
    run_complete_validation,
    run_leakage_checks,
    run_schema_checks,
)


def test_empty_dataframe_is_error():
    assert any(i.severity == "error" for i in run_schema_checks(pd.DataFrame()))


def test_small_dataframe_is_warning():
    issues = run_schema_checks(pd.DataFrame({"x": range(10), "target": [0, 1] * 5}), "target")
    assert any(i.severity == "warning" and "only 10 rows" in i.message for i in issues)


def test_missing_target_is_error():
    issues = run_schema_checks(pd.DataFrame({"x": range(30), "y": range(30)}), "missing")
    assert any(i.severity == "error" and i.column == "missing" for i in issues)


def test_more_than_ten_percent_missing_is_warning():
    frame = pd.DataFrame({"x": [np.nan] * 4 + list(range(26)), "target": [0, 1] * 15})
    assert any(i.severity == "warning" and i.column == "x" for i in run_schema_checks(frame, "target"))


def test_constant_column_is_warning():
    frame = pd.DataFrame({"constant": [1] * 30, "target": [0, 1] * 15})
    assert any(i.severity == "warning" and i.column == "constant" for i in run_schema_checks(frame, "target"))


def test_single_class_target_is_error():
    frame = pd.DataFrame({"x": range(30), "target": ["one"] * 30})
    assert any(i.severity == "error" and i.column == "target" for i in run_schema_checks(frame, "target"))


def test_empty_dataset_does_not_report_redundant_single_class_error():
    frame = pd.DataFrame(columns=["x", "target"])
    issues = run_schema_checks(frame, "target")
    assert any("empty" in issue.message.lower() for issue in issues)
    assert not any("only one class" in issue.message.lower() for issue in issues)


def test_duplicate_rows_are_warning():
    frame = pd.DataFrame({"x": [1, 2] * 15, "target": [0, 1] * 15})
    assert any(i.severity == "warning" and "duplicate" in i.message for i in run_schema_checks(frame, "target"))


def test_report_passes_without_errors():
    report = ValidationReport([ValidationIssue("warning", "schema", None, "Review")])
    assert report.passed


def test_report_fails_with_error():
    report = ValidationReport([ValidationIssue("error", "schema", None, "Stop")])
    assert not report.passed


def test_report_errors_filter_severity():
    error = ValidationIssue("error", "schema", "x", "Stop")
    report = ValidationReport([error, ValidationIssue("warning", "schema", "y", "Review")])
    assert report.errors() == [error]


def test_report_by_column_filters_column():
    x_issue = ValidationIssue("info", "schema", "x", "Note")
    report = ValidationReport([x_issue, ValidationIssue("info", "schema", "y", "Other")])
    assert report.by_column("x") == [x_issue]


def test_report_markdown_is_non_empty():
    assert ValidationReport().to_markdown().strip()


def test_numeric_drift_detects_clear_shift():
    rng = np.random.default_rng(42)
    reference = pd.DataFrame({"x": rng.normal(0, 1, 200)})
    new = pd.DataFrame({"x": rng.normal(10, 1, 200)})
    assert compute_numeric_drift(reference, new, "x")["drifted"]


def test_numeric_drift_same_distribution_is_not_flagged():
    rng = np.random.default_rng(42)
    values = rng.normal(0, 1, 200)
    reference = pd.DataFrame({"x": values})
    new = pd.DataFrame({"x": values.copy()})
    assert not compute_numeric_drift(reference, new, "x")["drifted"]


def test_categorical_drift_detects_clear_shift():
    reference = pd.DataFrame({"x": ["a"] * 90 + ["b"] * 10})
    new = pd.DataFrame({"x": ["a"] * 10 + ["b"] * 90})
    assert compute_categorical_drift(reference, new, "x")["drifted"]


def test_identical_feature_is_leakage_error():
    frame = pd.DataFrame({"feature": ["a", "b"] * 50, "target": ["a", "b"] * 50})
    issues = run_leakage_checks(frame, "target", "classification")
    assert any(i.severity == "error" and i.column == "feature" for i in issues)


def test_unrelated_features_are_not_suspicious():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({
        "noise": rng.normal(size=400),
        "target": rng.integers(0, 2, size=400),
    })
    assert run_leakage_checks(frame, "target", "classification") == []


def test_unique_categorical_identifier_is_not_mislabeled_as_target_leakage():
    frame = pd.DataFrame({
        "record_id": [f"row-{index}" for index in range(120)],
        "target": ["a", "b", "c"] * 40,
    })
    assert run_leakage_checks(frame, "target", "classification") == []


def test_direct_target_copy_still_wins_over_identifier_suppression():
    labels = [f"class-{index}" for index in range(60)]
    frame = pd.DataFrame({"copy": labels, "target": labels})
    issues = run_leakage_checks(frame, "target", "classification")
    assert any(issue.severity == "error" and issue.column == "copy" for issue in issues)


def test_complete_validation_includes_all_categories():
    new = pd.DataFrame({"identity": [0, 1] * 50, "target": [0, 1] * 50})
    reference = pd.DataFrame({"identity": [0] * 90 + [1] * 10, "target": [0, 1] * 50})
    report = run_complete_validation(new, "target", "classification", reference)
    assert report.categories_checked == ("schema", "leakage", "drift")
    assert {issue.category for issue in report.issues} >= {"schema", "leakage", "drift"}


def test_complete_validation_skips_drift_without_reference():
    frame = pd.DataFrame({"x": range(40), "target": [0, 1] * 20})
    report = run_complete_validation(frame, "target", "classification")
    assert "drift" not in report.categories_checked
    assert not report.by_category("drift")
