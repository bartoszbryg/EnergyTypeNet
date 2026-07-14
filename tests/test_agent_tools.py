import pandas as pd

from src.agent_tools import (
    compare_feature_sets,
    compute_feature_diagnostics,
    compute_model_diagnostics,
    run_dataset_computation,
    should_run_computation,
)
from src.automl import prepare_dataset, profile_dataset, rank_features


def _sample_classification_context():
    df = pd.DataFrame({
        'energy': [10, 11, 12, 50, 55, 58, 90, 95, 99],
        'size': [100, 110, 120, 500, 520, 530, 900, 930, 950],
        'region': ['A', 'A', 'B', 'B', 'B', 'A', 'C', 'C', 'C'],
        'target': ['low', 'low', 'low', 'mid', 'mid', 'mid', 'high', 'high', 'high'],
    })
    profile = profile_dataset(df)
    prepared = prepare_dataset(
        df,
        'target',
        ['energy', 'size', 'region'],
        'classification',
    )
    ranking = rank_features(prepared)
    results = pd.DataFrame([
        {
            'model': 'Logistic Regression',
            'cv_accuracy': 0.93,
            'cv_f1_macro': 0.92,
            'test_accuracy': 1.0,
            'test_f1_macro': 1.0,
        },
        {
            'model': 'KNN',
            'cv_accuracy': 0.90,
            'cv_f1_macro': 0.88,
            'test_accuracy': 1.0,
            'test_f1_macro': 1.0,
        },
        {
            'model': 'Dummy baseline',
            'cv_accuracy': 0.33,
            'cv_f1_macro': 0.16,
            'test_accuracy': 0.33,
            'test_f1_macro': 0.16,
        },
    ])

    return profile, prepared, ranking, results


def test_should_run_computation_uses_previous_answer_context():
    assert not should_run_computation('Could you do it for me?')

    assert should_run_computation(
        'Could you do it for me?',
        previous_answer='We should compute feature importance and model complexity next.',
    )


def test_compute_model_diagnostics_marks_near_best_models():
    _, prepared, _, results = _sample_classification_context()

    diagnostics = compute_model_diagnostics(prepared, results)

    assert diagnostics.iloc[0]['model'] == 'Logistic Regression'
    assert bool(diagnostics.iloc[0]['near_best']) is True
    assert bool(diagnostics.iloc[1]['near_best']) is True
    assert 'cv_test_gap' in diagnostics.columns


def test_compute_feature_diagnostics_includes_quality_signals():
    _, prepared, ranking, _ = _sample_classification_context()

    diagnostics = compute_feature_diagnostics(prepared, ranking)

    assert set(['feature', 'mutual_information', 'mi_share', 'missing_pct']).issubset(
        diagnostics.columns
    )
    assert diagnostics['mi_share'].sum() > 0


def test_compare_feature_sets_reports_both_feature_sets():
    _, prepared, _, results = _sample_classification_context()
    compact = pd.DataFrame([
        {
            'model': 'Logistic Regression',
            'cv_accuracy': 0.93,
            'cv_f1_macro': 0.92,
            'test_accuracy': 0.95,
            'test_f1_macro': 0.95,
        },
    ])

    comparison = compare_feature_sets(prepared, results, compact)

    assert comparison['feature_set'].tolist() == [
        'All selected features',
        'Suggested compact features',
    ]
    assert comparison.iloc[0]['primary_metric'] == 1.0


def test_run_dataset_computation_returns_tables_and_grounded_answer():
    profile, prepared, ranking, results = _sample_classification_context()
    compact = pd.DataFrame([{
        'model': 'Gradient Boosting',
        'cv_accuracy': 0.48,
        'cv_f1_macro': 0.47,
        'test_accuracy': 0.52,
        'test_f1_macro': 0.51,
    }])

    output = run_dataset_computation(
        'Can you investigate why this is the best model?',
        profile,
        prepared,
        results,
        ranking,
        compact,
    )

    assert output is not None
    assert output.tool_name == 'dataset_diagnostics'
    assert 'Model check' in output.answer
    assert 'Feature check' in output.answer
    assert 'red flag' in output.answer
    assert 'model_diagnostics' in output.tables
    assert 'feature_diagnostics' in output.tables
    assert 'feature_set_comparison' in output.tables
