import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from src.models import (
    AdaBoostClassifierCustom,
    BaggingClassifierCustom,
    BaggingRegressorCustom,
    DecisionTreeClassifierCustom,
    DecisionTreeRegressorCustom,
)


def classification_split():
    X, y = make_classification(
        n_samples=300,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        class_sep=1.5,
        random_state=42,
    )

    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)


def test_bagging_default_base_estimator_is_custom_tree():
    X_train, _, y_train, _ = classification_split()
    clf = BaggingClassifierCustom(n_estimators=5, random_state=0)
    clf.fit(X_train, y_train)

    assert all(isinstance(estimator, DecisionTreeClassifierCustom) for estimator in clf.estimators_)


def test_bagging_predict_shape_and_valid_labels():
    X_train, X_test, y_train, y_test = classification_split()
    clf = BaggingClassifierCustom(n_estimators=10, random_state=0)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    assert pred.shape == y_test.shape
    assert set(np.unique(pred)).issubset(set(clf.classes_))


def test_bagging_predict_proba_shape_and_rows_sum_to_one():
    X_train, X_test, y_train, _ = classification_split()
    clf = BaggingClassifierCustom(n_estimators=10, random_state=0)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)

    assert proba.shape == (X_test.shape[0], len(clf.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_bagging_accuracy_above_baseline():
    X_train, X_test, y_train, y_test = classification_split()
    clf = BaggingClassifierCustom(n_estimators=25, random_state=0)
    clf.fit(X_train, y_train)

    assert clf.score(X_test, y_test) > 0.65


def test_bagging_custom_base_estimator_with_max_depth_works():
    X_train, X_test, y_train, y_test = classification_split()
    clf = BaggingClassifierCustom(
        base_estimator=DecisionTreeClassifierCustom(max_depth=3),
        n_estimators=12,
        random_state=0,
    )
    clf.fit(X_train, y_train)

    assert clf.predict(X_test).shape == y_test.shape
    assert all(estimator.max_depth == 3 for estimator in clf.estimators_)


def test_bagging_max_features_half_uses_half_of_available_features():
    X_train, _, y_train, _ = classification_split()
    clf = BaggingClassifierCustom(n_estimators=7, max_features=0.5, random_state=0)
    clf.fit(X_train, y_train)

    assert all(len(features) == 3 for features in clf.estimators_features_)


def test_bagging_oob_score_requires_bootstrap():
    X_train, _, y_train, _ = classification_split()
    clf = BaggingClassifierCustom(bootstrap=False, oob_score=True, random_state=0)

    with pytest.raises(ValueError, match='OOB scoring requires bootstrap=True'):
        clf.fit(X_train, y_train)


def test_bagging_oob_score_is_valid_when_enabled():
    X_train, _, y_train, _ = classification_split()
    clf = BaggingClassifierCustom(n_estimators=20, oob_score=True, random_state=0)
    clf.fit(X_train, y_train)

    assert clf.oob_score_ is not None
    assert 0.0 <= clf.oob_score_ <= 1.0


def test_bagging_feature_importances_match_input_width_and_sum_to_one():
    X_train, _, y_train, _ = classification_split()
    clf = BaggingClassifierCustom(n_estimators=12, random_state=0)
    clf.fit(X_train, y_train)

    importances = clf.feature_importances_

    assert importances.shape == (X_train.shape[1],)
    assert np.isclose(importances.sum(), 1.0)


def test_bagging_n_estimators_is_respected():
    X_train, _, y_train, _ = classification_split()
    clf = BaggingClassifierCustom(n_estimators=13, random_state=0)
    clf.fit(X_train, y_train)

    assert len(clf.estimators_) == 13


def test_bagging_repr_contains_class_name_and_estimator_count():
    clf = BaggingClassifierCustom(n_estimators=13, random_state=0)

    assert 'BaggingClassifierCustom' in repr(clf)
    assert 'n_estimators=13' in repr(clf)


def test_bagging_without_bootstrap_has_no_repeated_sample_indices():
    X_train, _, y_train, _ = classification_split()
    clf = BaggingClassifierCustom(n_estimators=5, bootstrap=False, random_state=0)
    clf.fit(X_train, y_train)

    for sample_idx in clf.estimators_samples_:
        assert len(sample_idx) == len(np.unique(sample_idx))


def test_adaboost_default_base_estimator_is_decision_stump():
    X_train, _, y_train, _ = classification_split()
    clf = AdaBoostClassifierCustom(n_estimators=5, random_state=0)
    clf.fit(X_train, y_train)

    assert all(isinstance(estimator, DecisionTreeClassifierCustom) for estimator in clf.estimators_)
    assert all(estimator.max_depth == 1 for estimator in clf.estimators_)


def test_adaboost_predict_shape_and_valid_labels():
    X_train, X_test, y_train, y_test = classification_split()
    clf = AdaBoostClassifierCustom(n_estimators=20, random_state=0)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    assert pred.shape == y_test.shape
    assert set(np.unique(pred)).issubset(set(clf.classes_))


def test_adaboost_predict_proba_rows_sum_to_one():
    X_train, X_test, y_train, _ = classification_split()
    clf = AdaBoostClassifierCustom(n_estimators=20, random_state=0)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)

    assert proba.shape == (X_test.shape[0], len(clf.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_adaboost_accuracy_above_baseline():
    X_train, X_test, y_train, y_test = classification_split()
    clf = AdaBoostClassifierCustom(n_estimators=40, learning_rate=0.5, random_state=0)
    clf.fit(X_train, y_train)

    assert clf.score(X_test, y_test) > 0.65


def test_adaboost_tracks_weights_errors_and_staged_predictions():
    X_train, X_test, y_train, _ = classification_split()
    clf = AdaBoostClassifierCustom(n_estimators=12, random_state=0)
    clf.fit(X_train, y_train)

    staged = list(clf.staged_predict(X_test))

    assert len(clf.estimator_weights_) == len(clf.estimators_)
    assert len(clf.estimator_errors_) == len(clf.estimators_)
    assert len(staged) == len(clf.estimators_)
    assert all(pred.shape == (X_test.shape[0],) for pred in staged)


def test_adaboost_high_learning_rate_stops_without_crashing():
    X, y = make_classification(
        n_samples=45,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        random_state=7,
    )
    clf = AdaBoostClassifierCustom(n_estimators=15, learning_rate=2.0, random_state=0)
    clf.fit(X, y)

    assert 1 <= len(clf.estimators_) <= 15
    assert clf.predict(X).shape == y.shape


def test_bagging_regressor_default_base_estimator_is_custom_tree():
    X, y = make_regression(n_samples=180, n_features=6, noise=10.0, random_state=42)
    reg = BaggingRegressorCustom(n_estimators=5, random_state=0)
    reg.fit(X, y)

    assert all(isinstance(estimator, DecisionTreeRegressorCustom) for estimator in reg.estimators_)


def test_bagging_regressor_predict_shape_and_positive_r2():
    X, y = make_regression(n_samples=220, n_features=6, noise=5.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    reg = BaggingRegressorCustom(
        base_estimator=DecisionTreeRegressorCustom(max_depth=6),
        n_estimators=20,
        random_state=0,
    )
    reg.fit(X_train, y_train)

    pred = reg.predict(X_test)

    assert pred.shape == y_test.shape
    assert reg.score(X_test, y_test) > 0.0


def test_bagging_regressor_feature_importances_and_oob_score():
    X, y = make_regression(n_samples=220, n_features=6, noise=5.0, random_state=42)
    reg = BaggingRegressorCustom(n_estimators=20, oob_score=True, random_state=0)
    reg.fit(X, y)

    assert reg.feature_importances_.shape == (X.shape[1],)
    assert np.isclose(reg.feature_importances_.sum(), 1.0)
    assert reg.oob_score_ is not None
    assert isinstance(reg.oob_score_, float)
