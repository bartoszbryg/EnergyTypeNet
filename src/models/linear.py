"""Custom classification and regression models for EnergyTypeNet."""

import inspect
from dataclasses import dataclass
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, clone
from sklearn.kernel_approximation import RBFSampler

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
