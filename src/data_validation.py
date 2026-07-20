"""Reusable schema, drift, and leakage checks for tabular datasets.

The module deliberately contains no Streamlit code.  It returns structured
findings that can be rendered by a dashboard, CLI, test, or model-card export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


VALID_SEVERITIES = {"error", "warning", "info"}
VALID_CATEGORIES = {"schema", "drift", "leakage"}


@dataclass(frozen=True)
class ValidationIssue:
    """One actionable data-validation finding."""

    severity: str
    category: str
    column: str | None
    message: str
    suggestion: str | None = None

    def __post_init__(self) -> None:
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(
                f"severity must be one of {sorted(VALID_SEVERITIES)}, "
                f"got {self.severity!r}"
            )
        if self.category not in VALID_CATEGORIES:
            raise ValueError(
                f"category must be one of {sorted(VALID_CATEGORIES)}, "
                f"got {self.category!r}"
            )


def _plural(count: int, singular: str) -> str:
    return f"{count} {singular if count == 1 else singular + 's'}"


def _summarize_issues(issues: list[ValidationIssue]) -> str:
    errors = sum(issue.severity == "error" for issue in issues)
    warnings = sum(issue.severity == "warning" for issue in issues)
    infos = sum(issue.severity == "info" for issue in issues)

    if not issues:
        return "No issues found"
    if not errors and not warnings:
        return f"Passed with {_plural(infos, 'informational note')}"

    parts = []
    if errors:
        parts.append(_plural(errors, "error"))
    if warnings:
        parts.append(_plural(warnings, "warning"))
    if infos:
        parts.append(_plural(infos, "informational note"))
    return " and ".join(parts) + " found"


@dataclass
class ValidationReport:
    """Aggregate findings and convenience views for one validation run."""

    issues: list[ValidationIssue] = field(default_factory=list)
    passed: bool = field(init=False)
    summary: str = field(init=False)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    categories_checked: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.issues = list(self.issues)
        self.passed = not any(issue.severity == "error" for issue in self.issues)
        self.summary = _summarize_issues(self.issues)

    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def infos(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "info"]

    def by_column(self, column: str) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.column == column]

    def by_category(self, category: str) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.category == category]

    def to_markdown(self) -> str:
        """Render a portable report for dashboards and model cards."""

        lines = [f"**Validation summary:** {self.summary}."]
        sections = (
            ("Errors", self.errors()),
            ("Warnings", self.warnings()),
            ("Information", self.infos()),
        )
        for heading, findings in sections:
            if not findings:
                continue
            lines.extend(["", f"### {heading}"])
            for issue in findings:
                prefix = f"[{issue.column}] " if issue.column else ""
                bullet = f"- {prefix}{issue.message}"
                if issue.suggestion:
                    bullet += f" *{issue.suggestion}*"
                lines.append(bullet)
        return "\n".join(lines)


def run_schema_checks(
    dataframe: pd.DataFrame,
    target_column: str | None = None,
) -> list[ValidationIssue]:
    """Inspect static shape, completeness, cardinality, and range properties."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")

    issues: list[ValidationIssue] = []
    row_count, column_count = dataframe.shape

    if row_count == 0:
        issues.append(ValidationIssue(
            "error",
            "schema",
            None,
            "The dataset is empty and cannot be used for training.",
            "Upload a CSV containing at least 30 usable rows.",
        ))
    elif row_count < 30:
        issues.append(ValidationIssue(
            "warning",
            "schema",
            None,
            f"The dataset has only {row_count} rows; model estimates will be highly unreliable.",
            "Collect more representative samples before relying on model metrics.",
        ))

    if column_count < 2:
        issues.append(ValidationIssue(
            "error",
            "schema",
            None,
            "At least one feature column and one target column are required.",
            "Upload a dataset with two or more columns.",
        ))

    target_exists = target_column is not None and target_column in dataframe.columns
    if target_column is not None and not target_exists:
        issues.append(ValidationIssue(
            "error",
            "schema",
            target_column,
            f"The specified target column '{target_column}' was not found.",
            "Choose an existing target column or correct the CSV header.",
        ))

    for column in dataframe.columns:
        series = dataframe[column]
        missing_rate = float(series.isna().mean()) if row_count else 0.0
        if missing_rate > 0.50:
            issues.append(ValidationIssue(
                "error",
                "schema",
                str(column),
                f"The column is more than half empty ({missing_rate:.1%} missing).",
                "Drop the column or use a justified imputation strategy before training.",
            ))
        elif missing_rate >= 0.10:
            issues.append(ValidationIssue(
                "warning",
                "schema",
                str(column),
                f"The column has {missing_rate:.1%} missing values.",
                "Review the missingness pattern and apply an appropriate imputation strategy.",
            ))
        elif missing_rate >= 0.01:
            issues.append(ValidationIssue(
                "info",
                "schema",
                str(column),
                f"The column has {missing_rate:.1%} missing values.",
                None,
            ))

        unique_count = int(series.nunique(dropna=True))
        if row_count > 0 and target_exists and column == target_column and unique_count <= 1:
            issues.append(ValidationIssue(
                "error",
                "schema",
                str(column),
                "The target contains only one class and cannot be used for classification.",
                "Choose a target containing at least two observed classes.",
            ))
        elif unique_count == 1:
            issues.append(ValidationIssue(
                "warning",
                "schema",
                str(column),
                "The column is constant and carries no information for model training.",
                "Remove this column from the feature set.",
            ))
        elif unique_count < 3 and row_count > 20:
            issues.append(ValidationIssue(
                "info",
                "schema",
                str(column),
                f"The column has very low cardinality ({unique_count} unique values).",
                "Confirm that the limited set of values is expected.",
            ))

        if (
            row_count
            and (
                pd.api.types.is_object_dtype(series.dtype)
                or isinstance(series.dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(series.dtype)
            )
            and unique_count > 0.50 * row_count
        ):
            issues.append(ValidationIssue(
                "warning",
                "schema",
                str(column),
                f"The categorical column has very high cardinality ({unique_count} unique values across {row_count} rows).",
                "Inspect whether it is an identifier; drop it or choose a suitable encoding if appropriate.",
            ))

        if pd.api.types.is_numeric_dtype(series.dtype):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric) >= 2:
                mean = float(numeric.mean())
                std = float(numeric.std(ddof=0))
                if np.isfinite(std) and std > 0:
                    minimum = float(numeric.min())
                    maximum = float(numeric.max())
                    if minimum < mean - 10 * std or maximum > mean + 10 * std:
                        extreme = minimum if minimum < mean - 10 * std else maximum
                        issues.append(ValidationIssue(
                            "warning",
                            "schema",
                            str(column),
                            f"The column contains an extreme value ({extreme:g}) more than 10 standard deviations from its mean.",
                            "Verify whether the value is legitimate or a data-entry error.",
                        ))

    if target_exists:
        target = dataframe[target_column]
        if pd.api.types.is_numeric_dtype(target.dtype) and target.nunique(dropna=True) > 20:
            issues.append(ValidationIssue(
                "info",
                "schema",
                target_column,
                "The numeric target has more than 20 unique values and may represent a regression task.",
                "Confirm whether the desired output is continuous before selecting classification.",
            ))

    duplicate_count = int(dataframe.duplicated().sum())
    if duplicate_count:
        issues.append(ValidationIssue(
            "warning",
            "schema",
            None,
            f"The dataset contains {duplicate_count} duplicate row{'s' if duplicate_count != 1 else ''}.",
            "Remove or investigate duplicates before training to avoid biased evaluation.",
        ))

    return issues


def run_all_schema_checks(
    dataframe: pd.DataFrame,
    target_column: str | None = None,
) -> ValidationReport:
    """Run schema validation and return a structured report."""

    return ValidationReport(
        issues=run_schema_checks(dataframe, target_column),
        categories_checked=("schema",),
    )


def _drift_series(series: pd.Series) -> pd.Series:
    """Return a comparison-safe series while treating missing values explicitly."""

    return series.astype("object").where(series.notna(), "<MISSING>")


def compute_numeric_drift(
    reference: pd.DataFrame,
    new: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    """Measure numeric distribution drift with a two-sample KS test."""

    from scipy.stats import ks_2samp

    reference_values = pd.to_numeric(reference[column], errors="coerce").dropna()
    new_values = pd.to_numeric(new[column], errors="coerce").dropna()
    if reference_values.empty or new_values.empty:
        return {
            "test_name": "Kolmogorov-Smirnov",
            "statistic": None,
            "p_value": None,
            "drifted": False,
            "reference_mean": float(reference_values.mean()) if not reference_values.empty else None,
            "new_mean": float(new_values.mean()) if not new_values.empty else None,
            "reference_std": float(reference_values.std(ddof=0)) if not reference_values.empty else None,
            "new_std": float(new_values.std(ddof=0)) if not new_values.empty else None,
        }

    result = ks_2samp(reference_values, new_values)
    return {
        "test_name": "Kolmogorov-Smirnov",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "drifted": bool(result.pvalue < 0.05),
        "reference_mean": float(reference_values.mean()),
        "new_mean": float(new_values.mean()),
        "reference_std": float(reference_values.std(ddof=0)),
        "new_std": float(new_values.std(ddof=0)),
    }


def compute_categorical_drift(
    reference: pd.DataFrame,
    new: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    """Measure categorical drift using chi-squared and total variation distance."""

    from scipy.stats import chi2_contingency

    reference_values = _drift_series(reference[column])
    new_values = _drift_series(new[column])
    categories = reference_values.unique().tolist()
    categories.extend(value for value in new_values.unique() if value not in categories)
    reference_counts = reference_values.value_counts().reindex(categories, fill_value=0)
    new_counts = new_values.value_counts().reindex(categories, fill_value=0)
    contingency = np.vstack([reference_counts.to_numpy(), new_counts.to_numpy()])

    statistic: float | None
    p_value: float | None
    low_expected_counts = False
    try:
        statistic_raw, p_value_raw, _, expected = chi2_contingency(contingency)
        statistic = float(statistic_raw)
        p_value = float(p_value_raw)
        low_expected_counts = bool((expected < 5).any())
    except ValueError:
        identical = reference_counts.equals(new_counts)
        statistic = 0.0 if identical else None
        p_value = 1.0 if identical else None

    reference_distribution = reference_counts / max(int(reference_counts.sum()), 1)
    new_distribution = new_counts / max(int(new_counts.sum()), 1)
    total_variation_distance = float(
        0.5 * np.abs(reference_distribution - new_distribution).sum()
    )
    return {
        "test_name": "Chi-squared",
        "statistic": statistic,
        "p_value": p_value,
        "drifted": bool(p_value is not None and p_value < 0.05),
        "reference_mean": None,
        "new_mean": None,
        "reference_std": None,
        "new_std": None,
        "total_variation_distance": total_variation_distance,
        "low_expected_counts": low_expected_counts,
    }


def run_drift_checks(
    reference: pd.DataFrame,
    new: pd.DataFrame,
    target_column: str | None = None,
    significance_level: float = 0.05,
) -> list[ValidationIssue]:
    """Compare a new dataset with a reference dataset column by column."""

    if not 0 < significance_level < 1:
        raise ValueError("significance_level must be between 0 and 1")
    if not isinstance(reference, pd.DataFrame) or not isinstance(new, pd.DataFrame):
        raise TypeError("reference and new must be pandas DataFrames")

    issues: list[ValidationIssue] = []
    reference_features = [column for column in reference.columns if column != target_column]
    new_features = [column for column in new.columns if column != target_column]

    for column in reference_features:
        if column not in new.columns:
            issues.append(ValidationIssue(
                "error", "drift", str(column),
                f"The reference column '{column}' is missing from the new dataset.",
                "Restore the required feature or retrain a pipeline with the new schema.",
            ))
            continue

        reference_numeric = pd.api.types.is_numeric_dtype(reference[column].dtype)
        new_numeric = pd.api.types.is_numeric_dtype(new[column].dtype)
        if reference_numeric and new_numeric:
            result = compute_numeric_drift(reference, new, column)
            p_value = result["p_value"]
            if p_value is not None and p_value < significance_level:
                issues.append(ValidationIssue(
                    "warning", "drift", str(column),
                    "Numeric distribution drift detected "
                    f"(KS p={p_value:.3g}; mean {result['reference_mean']:.4g} -> {result['new_mean']:.4g}).",
                    "Investigate the source shift and consider retraining if it reflects production data.",
                ))
        else:
            result = compute_categorical_drift(reference, new, column)
            p_value = result["p_value"]
            if p_value is not None and p_value < significance_level:
                tvd = result["total_variation_distance"]
                detail = f"; total variation distance={tvd:.3f}" if tvd > 0.10 else ""
                issues.append(ValidationIssue(
                    "warning", "drift", str(column),
                    f"Categorical distribution drift detected (chi-squared p={p_value:.3g}{detail}).",
                    "Review category frequencies and consider retraining if the shift is representative.",
                ))
            if result["low_expected_counts"]:
                issues.append(ValidationIssue(
                    "info", "drift", str(column),
                    "Some expected category counts are below 5, so the chi-squared result may be unstable.",
                    "Collect more samples or combine genuinely equivalent rare categories.",
                ))

    for column in new_features:
        if column not in reference.columns:
            issues.append(ValidationIssue(
                "warning", "drift", str(column),
                f"The new column '{column}' was not present in the reference dataset.",
                "Confirm whether the deployed preprocessing pipeline should consume this feature.",
            ))
    return issues


def run_all_drift_checks(
    reference: pd.DataFrame,
    new: pd.DataFrame,
    target_column: str | None = None,
    significance_level: float = 0.05,
) -> ValidationReport:
    """Run distribution drift checks and return a structured report."""

    return ValidationReport(
        issues=run_drift_checks(reference, new, target_column, significance_level),
        categories_checked=("drift",),
    )


def check_column_coverage(
    required_columns: list[str] | tuple[str, ...],
    new: pd.DataFrame,
) -> ValidationReport:
    """Check that all features required by a trained model are present."""

    issues = [
        ValidationIssue(
            "error", "schema", str(column),
            f"Required feature '{column}' is missing.",
            "Provide the feature with the same name used during model training.",
        )
        for column in required_columns
        if column not in new.columns
    ]
    return ValidationReport(issues=issues, categories_checked=("schema",))


def _correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    """Return eta-squared for a categorical feature and numeric target."""

    valid = categories.notna() & measurements.notna()
    groups = categories[valid].astype(str)
    values = pd.to_numeric(measurements[valid], errors="coerce")
    valid_numeric = values.notna()
    groups, values = groups[valid_numeric], values[valid_numeric]
    if values.empty or values.nunique() <= 1:
        return 0.0
    grand_mean = float(values.mean())
    denominator = float(((values - grand_mean) ** 2).sum())
    if denominator <= 0:
        return 0.0
    numerator = sum(
        len(group_values) * (float(group_values.mean()) - grand_mean) ** 2
        for _, group_values in values.groupby(groups)
    )
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _direct_identity_score(feature: pd.Series, target: pd.Series) -> float:
    """Detect row-aligned identity without flagging coincidental equal histograms."""

    valid = feature.notna() & target.notna()
    if not valid.any():
        return 0.0
    feature_values = feature[valid]
    target_values = target[valid]
    feature_codes, _ = pd.factorize(feature_values, sort=True)
    target_codes, _ = pd.factorize(target_values, sort=True)
    if np.array_equal(feature_codes, target_codes):
        return 1.0
    if pd.api.types.is_numeric_dtype(feature_values.dtype):
        numeric_feature = pd.to_numeric(feature_values, errors="coerce")
        numeric_target = pd.to_numeric(target_values, errors="coerce")
        numeric_valid = numeric_feature.notna() & numeric_target.notna()
        if numeric_valid.sum() >= 2:
            correlation = numeric_feature[numeric_valid].corr(numeric_target[numeric_valid])
            if pd.notna(correlation) and abs(float(correlation)) > 0.999:
                return 1.0
    return 0.0


def compute_leakage_scores(
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: str,
) -> dict[str, float]:
    """Estimate feature/target association for leakage diagnostics."""

    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder

    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' was not found")
    normalized_task = task_type.strip().lower()
    if normalized_task not in {"classification", "regression"}:
        raise ValueError("task_type must be 'classification' or 'regression'")

    usable = dataframe.loc[dataframe[target_column].notna()].copy()
    target = usable[target_column]
    scores: dict[str, float] = {}

    if normalized_task == "classification":
        class_count = int(target.nunique(dropna=True))
        if class_count < 2:
            return {str(column): 0.0 for column in usable.columns if column != target_column}
        encoded_target = LabelEncoder().fit_transform(target.astype(str))
        normalization = max(float(np.log(class_count)), np.finfo(float).eps)
        for column in usable.columns:
            if column == target_column:
                continue
            feature = usable[column]
            if _direct_identity_score(feature, target) == 1.0:
                scores[str(column)] = 1.0
                continue
            # Label-encoding a nearly unique identifier makes mutual information
            # spuriously equal to the target entropy. Schema validation already
            # reports these columns as identifier-like, so do not mislabel them
            # as proven target leakage unless the row-aligned identity check above
            # actually found a direct copy of the target.
            if (
                not pd.api.types.is_numeric_dtype(feature.dtype)
                and len(feature) > 0
                and feature.nunique(dropna=True) / len(feature) > 0.50
            ):
                scores[str(column)] = 0.0
                continue
            if pd.api.types.is_numeric_dtype(feature.dtype):
                numeric = pd.to_numeric(feature, errors="coerce")
                fill_value = float(numeric.median()) if numeric.notna().any() else 0.0
                encoded = numeric.fillna(fill_value).to_numpy().reshape(-1, 1)
                discrete = False
            else:
                encoded = LabelEncoder().fit_transform(
                    feature.astype("object").where(feature.notna(), "<MISSING>").astype(str)
                ).reshape(-1, 1)
                discrete = True
            information = float(
                mutual_info_classif(
                    encoded,
                    encoded_target,
                    discrete_features=[discrete],
                    random_state=42,
                )[0]
            )
            scores[str(column)] = float(np.clip(information / normalization, 0.0, 1.0))
    else:
        numeric_target = pd.to_numeric(target, errors="coerce")
        for column in usable.columns:
            if column == target_column:
                continue
            feature = usable[column]
            if _direct_identity_score(feature, target) == 1.0:
                scores[str(column)] = 1.0
            elif pd.api.types.is_numeric_dtype(feature.dtype):
                numeric_feature = pd.to_numeric(feature, errors="coerce")
                valid = numeric_feature.notna() & numeric_target.notna()
                correlation = (
                    numeric_feature[valid].corr(numeric_target[valid])
                    if valid.sum() >= 2 else 0.0
                )
                scores[str(column)] = float(
                    np.clip(abs(correlation) if pd.notna(correlation) else 0.0, 0.0, 1.0)
                )
            else:
                scores[str(column)] = _correlation_ratio(feature, numeric_target)
    return scores


def run_leakage_checks(
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: str,
) -> list[ValidationIssue]:
    """Turn feature/target association scores into actionable findings."""

    issues: list[ValidationIssue] = []
    for column, score in compute_leakage_scores(dataframe, target_column, task_type).items():
        if score > 0.95:
            issues.append(ValidationIssue(
                "error", "leakage", column,
                f"The feature has an extremely strong target association (leakage score {score:.3f}).",
                "Remove it unless it is genuinely available before the prediction is made.",
            ))
        elif score >= 0.80:
            issues.append(ValidationIssue(
                "warning", "leakage", column,
                f"The feature has a very strong target association (leakage score {score:.3f}).",
                "Verify its definition and availability at prediction time before training.",
            ))
        elif score >= 0.60:
            issues.append(ValidationIssue(
                "info", "leakage", column,
                f"The feature has a strong target association (leakage score {score:.3f}).",
                "Review it for possible proxy leakage and validate performance on an untouched split.",
            ))
    return issues


def run_all_leakage_checks(
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: str,
) -> ValidationReport:
    """Run leakage diagnostics and return a structured report."""

    return ValidationReport(
        issues=run_leakage_checks(dataframe, target_column, task_type),
        categories_checked=("leakage",),
    )


def run_complete_validation(
    new: pd.DataFrame,
    target_column: str,
    task_type: str,
    reference: pd.DataFrame | None = None,
) -> ValidationReport:
    """Run available schema, leakage, and optional drift validation layers."""

    categories = ["schema"]
    issues = run_schema_checks(new, target_column)
    if target_column in new.columns:
        categories.append("leakage")
        issues.extend(run_leakage_checks(new, target_column, task_type))
    if reference is not None:
        categories.append("drift")
        issues.extend(run_drift_checks(reference, new, target_column))

    severity_order = {"error": 0, "warning": 1, "info": 2}
    category_order = {"schema": 0, "leakage": 1, "drift": 2}
    issues.sort(key=lambda issue: (
        severity_order[issue.severity],
        category_order[issue.category],
        issue.column or "",
    ))
    report = ValidationReport(issues=issues, categories_checked=tuple(categories))
    report.summary = f"{report.summary}; checks run: {', '.join(categories)}"
    return report
