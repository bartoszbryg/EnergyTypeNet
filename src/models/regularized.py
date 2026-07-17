"""Custom regularized linear models."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

__all__ = ["RidgeRegressionCustom", "LassoRegressionCustom", "ElasticNetCustom", "RegularizedLogisticRegression"]

class RidgeRegressionCustom(RegressorMixin, BaseEstimator):
    """Ridge regression solved in closed form with an unregularized intercept."""

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def __repr__(self) -> str:
        return f'RidgeRegressionCustom(alpha={self.alpha}, fit_intercept={self.fit_intercept})'

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeRegressionCustom':
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if self.fit_intercept:
            self.X_mean_ = X.mean(axis=0)
            self.y_mean_ = y.mean()
            X_fit = X - self.X_mean_
            y_fit = y - self.y_mean_
        else:
            self.X_mean_ = np.zeros(X.shape[1])
            self.y_mean_ = 0.0
            X_fit = X
            y_fit = y

        penalty = self.alpha * np.eye(X_fit.shape[1])
        self.coef_ = np.linalg.pinv(X_fit.T @ X_fit + penalty) @ X_fit.T @ y_fit
        self.intercept_ = self.y_mean_ - self.X_mean_ @ self.coef_ if self.fit_intercept else 0.0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

class LassoRegressionCustom(RegressorMixin, BaseEstimator):
    """Lasso regression optimized with coordinate descent."""

    def __init__(
        self,
        alpha: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True,
        track_path: bool = False,
    ):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.track_path = track_path

    def __repr__(self) -> str:
        return (
            f'LassoRegressionCustom(alpha={self.alpha}, max_iter={self.max_iter}, '
            f'tol={self.tol}, fit_intercept={self.fit_intercept})'
        )

    @staticmethod
    def soft_threshold(z: float | np.ndarray, gamma: float) -> float | np.ndarray:
        return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LassoRegressionCustom':
        X_fit, y_fit = self._center_data(X, y)
        n_samples, n_features = X_fit.shape

        self.coef_ = np.zeros(n_features)
        self.coef_path_: list[np.ndarray] = []
        prediction = X_fit @ self.coef_

        for iteration in range(1, self.max_iter + 1):
            old_coef = self.coef_.copy()

            for j in range(n_features):
                prediction -= X_fit[:, j] * self.coef_[j]
                rho = X_fit[:, j] @ (y_fit - prediction)
                z = np.sum(X_fit[:, j] ** 2)
                self.coef_[j] = self.soft_threshold(rho, self.alpha * n_samples) / (z + 1e-12)
                prediction += X_fit[:, j] * self.coef_[j]

            if self.track_path:
                self.coef_path_.append(self.coef_.copy())

            if np.max(np.abs(self.coef_ - old_coef)) < self.tol:
                self.converged_ = True
                self.n_iter_ = iteration
                break
        else:
            self.converged_ = False
            self.n_iter_ = self.max_iter

        self.intercept_ = self.y_mean_ - self.X_mean_ @ self.coef_ if self.fit_intercept else 0.0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def _center_data(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if self.fit_intercept:
            self.X_mean_ = X.mean(axis=0)
            self.y_mean_ = y.mean()
            return X - self.X_mean_, y - self.y_mean_

        self.X_mean_ = np.zeros(X.shape[1])
        self.y_mean_ = 0.0
        return X, y

class ElasticNetCustom(RegressorMixin, BaseEstimator):
    """ElasticNet regression optimized with coordinate descent."""

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def __repr__(self) -> str:
        return (
            f'ElasticNetCustom(alpha={self.alpha}, l1_ratio={self.l1_ratio}, '
            f'max_iter={self.max_iter}, tol={self.tol})'
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ElasticNetCustom':
        if not 0.0 <= self.l1_ratio <= 1.0:
            raise ValueError('l1_ratio must be between 0 and 1')

        X_fit, y_fit = self._center_data(X, y)
        n_samples, n_features = X_fit.shape

        self.coef_ = np.zeros(n_features)
        prediction = X_fit @ self.coef_

        for iteration in range(1, self.max_iter + 1):
            old_coef = self.coef_.copy()

            for j in range(n_features):
                prediction -= X_fit[:, j] * self.coef_[j]
                rho = X_fit[:, j] @ (y_fit - prediction)
                z = np.sum(X_fit[:, j] ** 2)
                l1_penalty = self.alpha * self.l1_ratio * n_samples
                l2_penalty = self.alpha * (1.0 - self.l1_ratio) * n_samples
                self.coef_[j] = LassoRegressionCustom.soft_threshold(rho, l1_penalty) / (z + l2_penalty + 1e-12)
                prediction += X_fit[:, j] * self.coef_[j]

            if np.max(np.abs(self.coef_ - old_coef)) < self.tol:
                self.converged_ = True
                self.n_iter_ = iteration
                break
        else:
            self.converged_ = False
            self.n_iter_ = self.max_iter

        self.intercept_ = self.y_mean_ - self.X_mean_ @ self.coef_ if self.fit_intercept else 0.0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def _center_data(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if self.fit_intercept:
            self.X_mean_ = X.mean(axis=0)
            self.y_mean_ = y.mean()
            return X - self.X_mean_, y - self.y_mean_

        self.X_mean_ = np.zeros(X.shape[1])
        self.y_mean_ = 0.0
        return X, y

class RegularizedLogisticRegression(ClassifierMixin, BaseEstimator):
    """One-vs-rest logistic regression with selectable regularization."""

    def __init__(
        self,
        penalty: str = 'l2',
        C: float = 1.0,
        l1_ratio: float = 0.5,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        random_state: int = 42,
    ):
        self.penalty = penalty
        self.C = C
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state

    def __repr__(self) -> str:
        return (
            f'RegularizedLogisticRegression(penalty={self.penalty!r}, C={self.C}, '
            f'l1_ratio={self.l1_ratio}, learning_rate={self.learning_rate}, '
            f'n_iterations={self.n_iterations})'
        )

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RegularizedLogisticRegression':
        if self.penalty not in {'none', 'l1', 'l2', 'elasticnet'}:
            raise ValueError("penalty must be one of 'none', 'l1', 'l2', 'elasticnet'")
        if self.C <= 0:
            raise ValueError('C must be positive')

        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()

        self.classes_ = np.unique(y)
        rng = np.random.RandomState(self.random_state)
        self.weights_ = rng.normal(0.0, 0.01, size=(len(self.classes_), X.shape[1]))
        self.intercepts_ = np.zeros(len(self.classes_))
        self.loss_history_: list[float] = []

        for _ in range(self.n_iterations):
            total_loss = 0.0

            for idx, cls in enumerate(self.classes_):
                y_binary = (y == cls).astype(float)
                logits = X @ self.weights_[idx] + self.intercepts_[idx]
                proba = self._sigmoid(logits)
                error = proba - y_binary

                grad_w = (X.T @ error) / X.shape[0]
                grad_b = error.mean()
                grad_w += self._penalty_gradient(self.weights_[idx])

                self.weights_[idx] -= self.learning_rate * grad_w
                self.intercepts_[idx] -= self.learning_rate * grad_b

                clipped = np.clip(proba, 1e-10, 1.0 - 1e-10)
                loss = -np.mean(y_binary * np.log(clipped) + (1 - y_binary) * np.log(1 - clipped))
                total_loss += loss + self._penalty_loss(self.weights_[idx])

            self.loss_history_.append(total_loss / len(self.classes_))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        scores = self._sigmoid(X @ self.weights_.T + self.intercepts_)

        return scores / (scores.sum(axis=1, keepdims=True) + 1e-12)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def _penalty_gradient(self, weights: np.ndarray) -> np.ndarray:
        strength = 1.0 / self.C

        if self.penalty == 'none':
            return np.zeros_like(weights)
        if self.penalty == 'l1':
            return strength * np.sign(weights)
        if self.penalty == 'l2':
            return strength * weights

        return strength * (self.l1_ratio * np.sign(weights) + (1.0 - self.l1_ratio) * weights)

    def _penalty_loss(self, weights: np.ndarray) -> float:
        strength = 1.0 / self.C

        if self.penalty == 'none':
            return 0.0
        if self.penalty == 'l1':
            return strength * np.sum(np.abs(weights))
        if self.penalty == 'l2':
            return 0.5 * strength * np.sum(weights ** 2)

        return strength * (
            self.l1_ratio * np.sum(np.abs(weights))
            + 0.5 * (1.0 - self.l1_ratio) * np.sum(weights ** 2)
        )
