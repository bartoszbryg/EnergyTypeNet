"""Custom classification models for EnergyTypeNet."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class AttentionClassifier(ClassifierMixin, BaseEstimator):
    """Kernel-weighted nearest-neighbor classifier."""

    def __init__(self, w: float = 1.0):
        self.w = w

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AttentionClassifier':
        self.X_train_ = X
        self.y_train_ = y.ravel()
        self.classes_ = np.unique(self.y_train_)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        diff = X[:, np.newaxis, :] - self.X_train_[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))

        weights = np.exp(-dist / self.w)
        weights /= weights.sum(axis=1, keepdims=True) + 1e-12

        return np.stack(
            [weights[:, self.y_train_ == c].sum(axis=1) for c in self.classes_],
            axis=1,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class LogisticRegressionOvR(ClassifierMixin, BaseEstimator):
    """One-vs-rest multiclass logistic regression trained with gradient descent."""

    def __init__(
        self,
        eta: float = 0.0001,
        n_iter: int = 1000,
        alpha: float = 0.0,
        random_state: int = 42,
    ):
        self.eta = eta
        self.n_iter = n_iter
        self.alpha = alpha
        self.random_state = random_state

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def _fit_binary(self, X: np.ndarray, y_bin: np.ndarray, rng: np.random.RandomState):
        w = rng.normal(0.0, 0.01, size=1 + X.shape[1])
        losses = []

        for _ in range(self.n_iter):
            net = X @ w[1:] + w[0]
            output = self._sigmoid(net)
            errors = y_bin - output

            w[1:] += self.eta * (X.T @ errors - self.alpha * w[1:])
            w[0] += self.eta * errors.sum()

            output = np.clip(output, 1e-10, 1 - 1e-10)
            loss = (
                -y_bin @ np.log(output)
                - (1 - y_bin) @ np.log(1 - output)
                + (self.alpha / 2) * np.sum(w[1:] ** 2)
            )
            losses.append(loss)

        return w, losses

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionOvR':
        self.classes_ = np.unique(y)
        rng = np.random.RandomState(self.random_state)

        self.weights_: list = []
        self.losses_: list = []

        for c in self.classes_:
            y_binary = (y == c).astype(float)
            w, losses = self._fit_binary(X, y_binary, rng)

            self.weights_.append(w)
            self.losses_.append(losses)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = np.column_stack([
            self._sigmoid(X @ w[1:] + w[0])
            for w in self.weights_
        ])

        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = np.column_stack([
            self._sigmoid(X @ w[1:] + w[0])
            for w in self.weights_
        ])
        scores /= scores.sum(axis=1, keepdims=True) + 1e-12

        return scores


class LogisticRegressionSoftmax(ClassifierMixin, BaseEstimator):
    """Multinomial logistic regression trained with softmax cross-entropy."""

    def __init__(
        self,
        eta: float = 0.01,
        n_iter: int = 1000,
        alpha: float = 0.0,
        random_state: int = 42,
    ):
        self.eta = eta
        self.n_iter = n_iter
        self.alpha = alpha
        self.random_state = random_state

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)

        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionSoftmax':
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        rng = np.random.RandomState(self.random_state)

        self.W_ = rng.normal(0.0, 0.01, size=(n_classes, n_features))
        self.b_ = np.zeros(n_classes)
        self.loss_: list = []

        Y = np.zeros((n_samples, n_classes))
        for i, c in enumerate(self.classes_):
            Y[:, i] = y == c

        for _ in range(self.n_iter):
            P = self._softmax(X @ self.W_.T + self.b_)
            dL = (P - Y) / n_samples

            self.W_ -= self.eta * (dL.T @ X + self.alpha * self.W_)
            self.b_ -= self.eta * dL.sum(axis=0)

            P_clip = np.clip(P, 1e-10, 1.0)
            loss = (
                -np.sum(Y * np.log(P_clip)) / n_samples
                + (self.alpha / 2) * np.sum(self.W_ ** 2)
            )
            self.loss_.append(loss)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._softmax(X @ self.W_.T + self.b_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
