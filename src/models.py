"""Custom classification and regression models for EnergyTypeNet."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class LinearRegressionGD(RegressorMixin, BaseEstimator):
    """Ordinary least squares regression trained with batch gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        fit_intercept: bool = True,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept

    def __repr__(self) -> str:
        return (
            f'LinearRegressionGD(learning_rate={self.learning_rate}, '
            f'n_iterations={self.n_iterations}, fit_intercept={self.fit_intercept})'
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionGD':
        X_fit = self._add_intercept(X)
        y = np.asarray(y, dtype=float).ravel()

        self.weights_ = np.zeros(X_fit.shape[1])
        self.loss_history_: list[float] = []

        for _ in range(self.n_iterations):
            predictions = X_fit @ self.weights_
            errors = predictions - y
            gradient = (2.0 / X_fit.shape[0]) * (X_fit.T @ errors)

            self.weights_ -= self.learning_rate * gradient
            self.loss_history_.append(float(np.mean(errors ** 2)))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._add_intercept(X) @ self.weights_

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        if not self.fit_intercept:
            return X

        return np.c_[np.ones(X.shape[0]), X]


class LinearRegressionNormal(RegressorMixin, BaseEstimator):
    """Ordinary least squares regression solved with the normal equation."""

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept

    def __repr__(self) -> str:
        return f'LinearRegressionNormal(fit_intercept={self.fit_intercept})'

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionNormal':
        X_fit = self._add_intercept(X)
        y = np.asarray(y, dtype=float).ravel()

        xtx = X_fit.T @ X_fit

        try:
            if np.linalg.cond(xtx) > 1 / np.finfo(float).eps:
                raise np.linalg.LinAlgError
            self.weights_ = np.linalg.inv(xtx) @ X_fit.T @ y
        except np.linalg.LinAlgError:
            self.weights_ = np.linalg.lstsq(X_fit, y, rcond=None)[0]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._add_intercept(X) @ self.weights_

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        if not self.fit_intercept:
            return X

        return np.c_[np.ones(X.shape[0]), X]


class Perceptron(ClassifierMixin, BaseEstimator):
    """Classic Rosenblatt Perceptron for binary classification."""

    def __init__(
        self,
        learning_rate: float = 1.0,
        n_iterations: int = 100,
        zero_based: bool = True,
        random_state: int | None = None,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.zero_based = zero_based
        self.random_state = random_state

    def __repr__(self) -> str:
        return (
            f'Perceptron(learning_rate={self.learning_rate}, '
            f'n_iterations={self.n_iterations}, zero_based={self.zero_based})'
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Perceptron':
        X_fit = np.c_[np.ones(X.shape[0]), np.asarray(X, dtype=float)]
        y_train = self._encode_y(y)

        self.weights_ = np.zeros(X_fit.shape[1])
        self.errors_: list[int] = []
        self.classes_ = np.array([0, 1]) if self.zero_based else np.array([-1, 1])

        for _ in range(self.n_iterations):
            errors = 0

            for xi, target in zip(X_fit, y_train):
                y_hat = 1 if xi @ self.weights_ >= 0.0 else 0
                update = self.learning_rate * (target - y_hat)

                if update != 0.0:
                    self.weights_ += update * xi
                    errors += 1

            self.errors_.append(errors)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_fit = np.c_[np.ones(X.shape[0]), np.asarray(X, dtype=float)]
        predictions = (X_fit @ self.weights_ >= 0.0).astype(int)

        if self.zero_based:
            return predictions

        return np.where(predictions == 1, 1, -1)

    def _encode_y(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).ravel()

        if self.zero_based:
            return y.astype(int)

        return (y == 1).astype(int)


class AdalineGD(ClassifierMixin, BaseEstimator):
    """Adaptive Linear Neuron trained with batch gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 50,
        fit_intercept: bool = True,
        random_state: int | None = None,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def __repr__(self) -> str:
        return (
            f'AdalineGD(learning_rate={self.learning_rate}, '
            f'n_iterations={self.n_iterations}, fit_intercept={self.fit_intercept})'
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdalineGD':
        X_fit = self._add_intercept(X)
        y_train = np.asarray(y, dtype=float).ravel()

        self.weights_ = np.zeros(X_fit.shape[1])
        self.loss_history_: list[float] = []
        self.classes_ = np.array([0, 1])

        for _ in range(self.n_iterations):
            net_input = X_fit @ self.weights_
            errors = y_train - net_input

            self.weights_ += self.learning_rate * (X_fit.T @ errors) / X_fit.shape[0]
            self.loss_history_.append(float(np.mean(errors ** 2)))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._add_intercept(X) @ self.weights_ >= 0.5).astype(int)

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        if not self.fit_intercept:
            return X

        return np.c_[np.ones(X.shape[0]), X]
    

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
