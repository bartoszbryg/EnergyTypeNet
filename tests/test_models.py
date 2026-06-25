import numpy as np

from src.models import AttentionClassifier, LogisticRegressionSoftmax


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
