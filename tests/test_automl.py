import pandas as pd

from src.automl import (
    answer_dataset_question,
    cluster_analysis,
    generate_dataset_report,
    guess_task_type,
    prepare_dataset,
    profile_dataset,
    rank_features,
    recommend_features,
    suggest_targets,
    train_baselines,
)


def test_profile_and_target_suggestions():
    df = pd.DataFrame({
        'feature_a': [1, 2, 3, 4],
        'feature_b': ['x', 'x', 'y', 'y'],
        'Building Type': ['A', 'A', 'B', 'B'],
    })

    profile = profile_dataset(df)
    suggestions = suggest_targets(df)

    assert profile['n_rows'] == 4
    assert profile['n_columns'] == 3
    assert suggestions[0]['column'] == 'Building Type'
    assert suggestions[0]['task_type'] == 'classification'


def test_risk_level_is_preferred_sample_target():
    df = pd.DataFrame({
        'HVAC Type': ['Electric', 'Gas', 'Hybrid', 'Electric'],
        'Maintenance Visits': [1, 3, 8, 2],
        'Risk Level': ['Low', 'Medium', 'High', 'Low'],
    })

    suggestions = suggest_targets(df)

    assert suggestions[0]['column'] == 'Risk Level'


def test_regression_requires_numeric_target():
    df = pd.DataFrame({
        'feature': [1, 2, 3],
        'target': ['a', 'b', 'c'],
    })

    try:
        prepare_dataset(df, 'target', ['feature'], 'regression')
    except ValueError as exc:
        assert 'must be numeric' in str(exc)
    else:
        raise AssertionError('Expected non-numeric regression target to fail')


def test_classification_requires_repeated_class_examples():
    df = pd.DataFrame({
        'feature': [1, 2, 3, 4],
        'target': [10, 20, 30, 40],
    })

    try:
        prepare_dataset(df, 'target', ['feature'], 'classification')
    except ValueError as exc:
        assert 'only one row' in str(exc)
    else:
        raise AssertionError('Expected one-row-per-class target to fail')


def test_prepare_classification_dataset_and_rank_features():
    df = pd.DataFrame({
        'energy': [10, 11, 50, 55, 12, 53],
        'size': [100, 120, 500, 520, 110, 510],
        'type': ['small', 'small', 'large', 'large', 'small', 'large'],
    })

    prepared = prepare_dataset(df, 'type', ['energy', 'size'], 'classification')
    ranking = rank_features(prepared)

    assert prepared.task_type == 'classification'
    assert prepared.classes == ['large', 'small']
    assert list(ranking.columns) == ['feature', 'mutual_information']


def test_recommend_features_marks_informative_columns():
    df = pd.DataFrame({
        'signal': [0, 0, 0, 1, 1, 1, 2, 2, 2],
        'weak_noise': [5, 4, 6, 5, 4, 6, 5, 4, 6],
        'target': ['low', 'low', 'low', 'mid', 'mid', 'mid', 'high', 'high', 'high'],
    })

    prepared = prepare_dataset(
        df,
        'target',
        ['signal', 'weak_noise'],
        'classification',
    )
    recommendations = recommend_features(prepared)

    assert recommendations.iloc[0]['feature'] == 'signal'
    assert recommendations.iloc[0]['recommendation'] in {'Strong', 'Moderate'}


def test_train_classification_baselines_small_dataset():
    df = pd.DataFrame({
        'x1': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        'x2': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        'target': ['low', 'low', 'low', 'mid', 'mid', 'mid',
                   'high', 'high', 'high', 'high', 'high', 'high'],
    })

    prepared = prepare_dataset(df, 'target', ['x1', 'x2'], 'classification')
    results, models = train_baselines(prepared, cv=2)

    assert not results.empty
    assert 'test_accuracy' in results.columns
    assert 'Logistic Regression' in models


def test_train_regression_baselines_and_report():
    df = pd.DataFrame({
        'x1': list(range(20)),
        'x2': [value * 2 for value in range(20)],
        'target': [value * 3.5 for value in range(20)],
    })

    prepared = prepare_dataset(df, 'target', ['x1', 'x2'], 'regression')
    profile = profile_dataset(df)
    results, _ = train_baselines(prepared, cv=2)
    ranking = rank_features(prepared)
    report = generate_dataset_report(profile, suggest_targets(df), prepared, results, ranking)

    assert guess_task_type(df['target']) == 'regression'
    assert not results.empty
    assert 'test_r2' in results.columns
    assert 'Dataset Report' in report


def test_dataset_report_distinguishes_full_and_compact_results():
    profile = {
        'n_rows': 100,
        'n_columns': 4,
        'missing_cells': 0,
        'duplicate_rows': 0,
    }
    target_suggestions = [{'column': 'target', 'task_type': 'classification'}]
    df = pd.DataFrame({
        'feature_a': [0, 1, 2, 0, 1, 2],
        'feature_b': [10, 11, 12, 13, 14, 15],
        'target': ['a', 'a', 'b', 'b', 'c', 'c'],
    })
    prepared = prepare_dataset(
        df,
        'target',
        ['feature_a', 'feature_b'],
        'classification',
    )
    results = pd.DataFrame([{
        'model': 'Logistic Regression',
        'test_accuracy': 1.0,
        'test_f1_macro': 1.0,
    }])
    compact_results = pd.DataFrame([{
        'model': 'KNN',
        'test_accuracy': 0.5,
        'test_f1_macro': 0.45,
    }])

    report = generate_dataset_report(
        profile,
        target_suggestions,
        prepared,
        results,
        None,
        compact_results,
    )

    assert 'All-selected-feature baseline result' in report
    assert 'Suggested-compact-feature baseline result' in report
    assert 'Test accuracy: **1.000**' in report
    assert 'Compact test accuracy: **0.500**' in report
    assert 'Diagnostic warning' in report
    assert 'label-revealing features' in report


def test_answer_overfitting_question_uses_cv_test_gap():
    df = pd.DataFrame({
        'signal': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        'noise': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'],
        'target': ['low', 'low', 'low', 'mid', 'mid', 'mid',
                   'high', 'high', 'high', 'high', 'high', 'high'],
    })
    profile = profile_dataset(df)
    prepared = prepare_dataset(df, 'target', ['signal', 'noise'], 'classification')
    ranking = rank_features(prepared)
    results = pd.DataFrame([{
        'model': 'Logistic Regression',
        'cv_accuracy': 0.95,
        'cv_f1_macro': 0.95,
        'test_accuracy': 0.70,
        'test_f1_macro': 0.70,
    }])

    answer = answer_dataset_question(
        'When I apply all selected features, will the model overfit?',
        profile,
        prepared,
        results,
        ranking,
    )

    assert 'CV accuracy' in answer
    assert 'test accuracy' in answer
    assert 'overfitting' in answer.lower()


def test_answer_best_features_does_not_return_best_model():
    df = pd.DataFrame({
        'strong_feature': [0, 0, 1, 1, 2, 2],
        'weak_feature': [1, 2, 1, 2, 1, 2],
        'target': ['a', 'a', 'b', 'b', 'c', 'c'],
    })
    profile = profile_dataset(df)
    prepared = prepare_dataset(
        df,
        'target',
        ['strong_feature', 'weak_feature'],
        'classification',
    )
    ranking = rank_features(prepared)
    results = pd.DataFrame([{
        'model': 'Logistic Regression',
        'test_accuracy': 1.0,
        'test_f1_macro': 1.0,
    }])

    answer = answer_dataset_question(
        'What are the best features?', profile, prepared, results, ranking
    )

    assert 'strongest measured feature candidates' in answer
    assert 'MI ' in answer
    assert 'best baseline' not in answer


def test_cluster_analysis_returns_labels_and_diagnostics():
    df = pd.DataFrame({
        'x': [-3, -2.8, -3.1, 0.0, 0.2, -0.1, 3.0, 3.2, 2.8],
        'y': [-3, -3.2, -2.8, 3.0, 2.8, 3.2, -3.0, -2.9, -3.1],
    })

    result = cluster_analysis(df, max_k=4)

    assert result['best_k'] in {2, 3, 4}
    assert len(result['kmeans_labels']) == len(df)
    assert len(result['gmm_labels']) == len(df)
    assert set(result['silhouette_scores']).issubset({2, 3, 4})
    assert len(result['figure'].axes) == 2

