"""Safe computation tools for the AI Dataset Assistant chat."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.automl import PreparedDataset


@dataclass
class AgentToolResult:
    """Computed result returned by a guarded dataset-agent tool."""

    tool_name: str
    answer: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    source: str = 'tool'


def should_run_computation(
    question: str,
    previous_answer: str | None = None,
) -> bool:
    """Detect whether the user is asking the assistant to compute diagnostics."""
    q = question.lower()
    previous = (previous_answer or '').lower()
    action_terms = [
        'compute',
        'calculate',
        'run',
        'investigate',
        'analysis',
        'analyze',
        'do it',
        'do this',
        'do that',
        'please do',
        'can you do',
    ]
    diagnostic_terms = [
        'feature importance',
        'importance',
        'complexity',
        'correlation',
        'mutual information',
        'overfit',
        'overfitting',
        'leakage',
        'gap',
        'best model',
        'model selection',
        'compact',
        'missing',
        'duplicates',
    ]

    explicit_request = any(term in q for term in action_terms)
    direct_diagnostic = any(term in q for term in diagnostic_terms)
    previous_requested_diagnostic = any(term in previous for term in diagnostic_terms)

    return direct_diagnostic or (explicit_request and previous_requested_diagnostic)


def run_dataset_computation(
    question: str,
    profile: dict[str, Any],
    prepared: PreparedDataset | None,
    results: pd.DataFrame | None,
    feature_ranking: pd.DataFrame | None,
    compact_results: pd.DataFrame | None = None,
    previous_answer: str | None = None,
) -> AgentToolResult | None:
    """Run bounded dataset diagnostics from already computed dashboard objects."""
    if not should_run_computation(question, previous_answer):
        return None

    if prepared is None:
        return AgentToolResult(
            tool_name='dataset_readiness_check',
            answer=(
                'I can run diagnostics after a target and feature set are selected. '
                f"Right now the dataset has {profile.get('n_rows', 'unknown')} rows, "
                f"{profile.get('n_columns', 'unknown')} columns, "
                f"{profile.get('missing_cells', 'unknown')} missing cells and "
                f"{profile.get('duplicate_rows', 'unknown')} duplicate rows."
            ),
        )

    tables: dict[str, pd.DataFrame] = {}
    answer_parts = [_data_quality_summary(profile)]

    if results is not None and not results.empty:
        model_diagnostics = compute_model_diagnostics(prepared, results)
        tables['model_diagnostics'] = model_diagnostics
        answer_parts.append(_model_diagnostics_text(prepared, model_diagnostics))

        complexity = compute_model_complexity(results)
        tables['model_complexity'] = complexity
        answer_parts.append(_complexity_text(complexity))

    if feature_ranking is not None and not feature_ranking.empty:
        feature_diagnostics = compute_feature_diagnostics(prepared, feature_ranking)
        tables['feature_diagnostics'] = feature_diagnostics
        answer_parts.append(_feature_diagnostics_text(feature_diagnostics))

        target_associations = compute_target_associations(prepared, feature_ranking)
        if not target_associations.empty:
            tables['target_associations'] = target_associations
            answer_parts.append(_target_association_text(target_associations))

    if (
        results is not None
        and compact_results is not None
        and not results.empty
        and not compact_results.empty
    ):
        comparison = compare_feature_sets(prepared, results, compact_results)
        tables['feature_set_comparison'] = comparison
        answer_parts.append(_feature_set_comparison_text(prepared, comparison))

    return AgentToolResult(
        tool_name='dataset_diagnostics',
        answer=' '.join(part for part in answer_parts if part),
        tables=tables,
    )


def compute_model_diagnostics(
    prepared: PreparedDataset,
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Compute gaps, ties and generalization checks from baseline results."""
    if prepared.task_type == 'classification':
        metric = 'test_accuracy'
        cv_metric = 'cv_accuracy'
        secondary = 'test_f1_macro'
    else:
        metric = 'test_r2'
        cv_metric = 'cv_r2'
        secondary = 'test_mae'

    rows = []
    best_score = float(results.iloc[0][metric])

    for rank, row in enumerate(results.itertuples(index=False), start=1):
        row_dict = row._asdict()
        test_score = float(row_dict[metric])
        cv_score = float(row_dict[cv_metric])
        rows.append({
            'rank': rank,
            'model': row_dict['model'],
            'test_metric': test_score,
            'cv_metric': cv_score,
            'cv_test_gap': cv_score - test_score,
            'gap_from_best': best_score - test_score,
            'secondary_metric': float(row_dict[secondary]),
            'near_best': abs(best_score - test_score) <= 0.01,
        })

    return pd.DataFrame(rows)


def compute_model_complexity(results: pd.DataFrame) -> pd.DataFrame:
    """Assign transparent model-complexity categories from estimator families."""
    rows = []

    for model_name in results['model'].astype(str):
        lower = model_name.lower()

        if 'dummy' in lower:
            level = 'Very low'
            reason = 'Majority or mean baseline.'
        elif any(token in lower for token in ['logistic', 'ridge']):
            level = 'Low'
            reason = 'Linear model with few moving parts.'
        elif any(token in lower for token in ['knn', 'svm', 'svr']):
            level = 'Medium'
            reason = 'Instance-based or margin model; can be harder to scale or tune.'
        elif any(token in lower for token in ['forest', 'extra', 'boost', 'xgboost', 'hist']):
            level = 'High'
            reason = 'Tree ensemble with many fitted learners.'
        elif any(token in lower for token in ['mlp', 'neural']):
            level = 'High'
            reason = 'Neural network with hidden-layer parameters.'
        else:
            level = 'Medium'
            reason = 'General estimator complexity.'

        rows.append({
            'model': model_name,
            'complexity': level,
            'reason': reason,
        })

    return pd.DataFrame(rows)


def compute_feature_diagnostics(
    prepared: PreparedDataset,
    feature_ranking: pd.DataFrame,
) -> pd.DataFrame:
    """Combine mutual information with missingness and cardinality."""
    rows = []
    mi_total = max(float(feature_ranking['mutual_information'].sum()), 1e-12)
    mi_lookup = dict(zip(
        feature_ranking['feature'].astype(str),
        feature_ranking['mutual_information'].astype(float),
    ))

    for feature in prepared.feature_cols:
        series = prepared.X[feature]
        mutual_info = float(mi_lookup.get(feature, 0.0))
        rows.append({
            'feature': feature,
            'mutual_information': mutual_info,
            'mi_share': mutual_info / mi_total,
            'missing_pct': float(series.isna().mean()),
            'unique': int(series.nunique(dropna=True)),
            'dtype': str(series.dtype),
        })

    return pd.DataFrame(rows).sort_values(
        'mutual_information',
        ascending=False,
    ).reset_index(drop=True)


def compute_target_associations(
    prepared: PreparedDataset,
    feature_ranking: pd.DataFrame,
) -> pd.DataFrame:
    """Compute lightweight target associations for numeric features."""
    rows = []
    y = np.asarray(prepared.y, dtype=float)
    mi_lookup = dict(zip(
        feature_ranking['feature'].astype(str),
        feature_ranking['mutual_information'].astype(float),
    ))

    for feature in prepared.numeric_cols:
        values = pd.to_numeric(prepared.X[feature], errors='coerce')
        values = values.fillna(values.median())

        if values.nunique(dropna=True) <= 1:
            corr = 0.0
        else:
            corr = float(np.corrcoef(values.to_numpy(dtype=float), y)[0, 1])

        if not np.isfinite(corr):
            corr = 0.0

        rows.append({
            'feature': feature,
            'abs_target_correlation': abs(corr),
            'target_correlation': corr,
            'mutual_information': float(mi_lookup.get(feature, 0.0)),
        })

    return pd.DataFrame(rows).sort_values(
        ['abs_target_correlation', 'mutual_information'],
        ascending=False,
    ).reset_index(drop=True)


def compare_feature_sets(
    prepared: PreparedDataset,
    full_results: pd.DataFrame,
    compact_results: pd.DataFrame,
) -> pd.DataFrame:
    """Compare all selected features against the suggested compact set."""
    if prepared.task_type == 'classification':
        metric = 'test_accuracy'
        secondary = 'test_f1_macro'
    else:
        metric = 'test_r2'
        secondary = 'test_mae'

    full = full_results.iloc[0]
    compact = compact_results.iloc[0]

    return pd.DataFrame([
        {
            'feature_set': 'All selected features',
            'best_model': full['model'],
            'primary_metric': float(full[metric]),
            'secondary_metric': float(full[secondary]),
        },
        {
            'feature_set': 'Suggested compact features',
            'best_model': compact['model'],
            'primary_metric': float(compact[metric]),
            'secondary_metric': float(compact[secondary]),
        },
    ])


def _data_quality_summary(profile: dict[str, Any]) -> str:
    return (
        f"Data quality check: {profile.get('n_rows', 'unknown')} rows, "
        f"{profile.get('n_columns', 'unknown')} columns, "
        f"{profile.get('missing_cells', 'unknown')} missing cells and "
        f"{profile.get('duplicate_rows', 'unknown')} duplicate rows."
    )


def _model_diagnostics_text(
    prepared: PreparedDataset,
    diagnostics: pd.DataFrame,
) -> str:
    best = diagnostics.iloc[0]
    near_best = diagnostics[diagnostics['near_best']]['model'].astype(str).tolist()
    tied_text = ''

    if len(near_best) > 1:
        tied_text = (
            f" Models within 0.01 of the best score: {', '.join(near_best)}."
        )

    if prepared.task_type == 'classification':
        caution = ''

        if float(best['test_metric']) >= 0.98 and len(near_best) > 1:
            caution = (
                ' Because several models are near-perfect, treat the score as a '
                'diagnostic signal to check leakage, feature definitions and the '
                'holdout split before calling it a solved problem.'
            )

        return (
            f"Model check: {best['model']} is ranked first with test accuracy "
            f"{best['test_metric']:.3f}, macro F1 {best['secondary_metric']:.3f} "
            f"and CV/test gap {best['cv_test_gap']:.3f}.{tied_text}{caution}"
        )

    return (
        f"Model check: {best['model']} is ranked first with test R2 "
        f"{best['test_metric']:.3f}, MAE {best['secondary_metric']:.3f} "
        f"and CV/test gap {best['cv_test_gap']:.3f}.{tied_text}"
    )


def _complexity_text(complexity: pd.DataFrame) -> str:
    if complexity.empty:
        return ''

    low_complexity = complexity[
        complexity['complexity'].isin(['Very low', 'Low'])
    ]['model'].astype(str).tolist()

    if low_complexity:
        return (
            f"Simpler candidates to prefer when performance is tied: "
            f"{', '.join(low_complexity[:3])}."
        )

    return 'Most top candidates are medium or high complexity, so validation matters.'


def _feature_diagnostics_text(diagnostics: pd.DataFrame) -> str:
    if diagnostics.empty:
        return ''

    top = diagnostics.head(3)
    weak = diagnostics[
        diagnostics['mutual_information'] <= 0.01
    ]['feature'].astype(str).head(3).tolist()
    top_text = ', '.join(
        f"{row.feature} (MI {row.mutual_information:.3f})"
        for row in top.itertuples()
    )
    weak_text = ''

    if weak:
        weak_text = f" Weak measured features: {', '.join(weak)}."

    return f"Feature check: strongest measured features are {top_text}.{weak_text}"


def _target_association_text(associations: pd.DataFrame) -> str:
    if associations.empty:
        return ''

    top = associations.iloc[0]

    return (
        f"Numeric association check: `{top['feature']}` has the largest absolute "
        f"target correlation ({float(top['abs_target_correlation']):.3f})."
    )


def _feature_set_comparison_text(
    prepared: PreparedDataset,
    comparison: pd.DataFrame,
) -> str:
    full = comparison.iloc[0]
    compact = comparison.iloc[1]
    delta = float(compact['primary_metric'] - full['primary_metric'])

    if abs(delta) <= 0.01:
        verdict = 'The compact set performs about the same, so it is easier to explain.'
    elif delta > 0:
        verdict = 'The compact set performs better in this split.'
    elif float(full['primary_metric']) >= 0.98 and abs(delta) >= 0.15:
        verdict = (
            'This gap is a red flag: the all-selected feature set may contain '
            'label-revealing information, or the holdout split may be too easy. '
            'Use the compact score as the more conservative benchmark.'
        )
    else:
        verdict = 'The full feature set performs better in this split.'

    metric_name = 'accuracy' if prepared.task_type == 'classification' else 'R2'

    return (
        f"Feature-set comparison: full {metric_name}={full['primary_metric']:.3f}, "
        f"compact {metric_name}={compact['primary_metric']:.3f}. {verdict}"
    )
