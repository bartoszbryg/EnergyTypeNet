import numpy as np
from sklearn.datasets import make_blobs

from src.models import (
    AdalineGD,
    AttentionClassifier,
    LinearRegressionGD,
    LinearRegressionNormal,
    LogisticRegressionSoftmax,
    Perceptron,
)


def test_attention_predict_proba_shape_and_sum():
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [5.0, 5.0],
        [6.0, 6.0],
    ])
    y = np.array([0, 0, 1, 1])
    model = AttentionClassifier(w=1.0).fit(X, y)

    proba = model.predict_proba(np.array([[0.5, 0.5], [5.5, 5.5]]))

    assert proba.shape == (2, 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(2))


def test_softmax_loss_decreases():
    X = np.array([
        [0.0, 0.0],
        [0.2, 0.1],
        [2.0, 2.0],
        [2.2, 2.1],
        [4.0, 4.0],
        [4.2, 4.1],
    ])
    y = np.array([0, 0, 1, 1, 2, 2])
    model = LogisticRegressionSoftmax(eta=0.1, n_iter=80, alpha=0.0)
    model.fit(X, y)

    assert model.loss_[0] > model.loss_[-1]
    assert model.predict(X).shape == y.shape


def test_linear_regression_gd_learns_simple_line():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = 3 * X.ravel() + 2

    model = LinearRegressionGD(learning_rate=0.001, n_iterations=5000)
    model.fit(X, y)

    assert model.loss_history_[0] > model.loss_history_[-1]
    np.testing.assert_allclose(model.predict([[10.0]]), [32.0], atol=0.5)


def test_linear_regression_normal_solves_simple_line():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = 3 * X.ravel() + 2

    model = LinearRegressionNormal()
    model.fit(X, y)

    np.testing.assert_allclose(model.predict([[10.0]]), [32.0], atol=1e-8)


def test_perceptron_separates_blobs():
    X, y = make_blobs(
        n_samples=80,
        centers=[[-2, -2], [2, 2]],
        cluster_std=0.4,
        random_state=42,
    )

    model = Perceptron(learning_rate=1.0, n_iterations=20)
    model.fit(X, y)

    assert model.errors_[-1] == 0
    assert model.score(X, y) == 1.0


def test_adaline_separates_blobs():
    X, y = make_blobs(
        n_samples=80,
        centers=[[-2, -2], [2, 2]],
        cluster_std=0.4,
        random_state=42,
    )

    model = AdalineGD(learning_rate=0.05, n_iterations=100)
    model.fit(X, y)

    assert model.loss_history_[0] > model.loss_history_[-1]
    assert model.score(X, y) > 0.95
