"""Custom support-vector-machine models."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.kernel_approximation import RBFSampler

__all__ = ["SVMClassifierCustom"]

class SVMClassifierCustom(ClassifierMixin, BaseEstimator):
    """Binary soft-margin SVM trained with subgradient descent."""

    def __init__(
        self,
        C: float = 1.0,
        learning_rate: float = 0.001,
        n_iterations: int = 1000,
        kernel: str = 'linear',
        gamma: float = 1.0,
        n_components: int = 200,
        random_state: int = 42,
    ):
        self.C = C
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.kernel = kernel
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def __repr__(self) -> str:
        return (
            f'SVMClassifierCustom(C={self.C}, learning_rate={self.learning_rate}, '
            f'n_iterations={self.n_iterations}, kernel={self.kernel!r})'
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMClassifierCustom':
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError('SVMClassifierCustom supports binary classification only')

        y_train = np.where(y == self.classes_[0], -1, 1)
        X_fit = self._transform_features(X, fit=True)

        self.w_ = np.zeros(X_fit.shape[1])
        self.b_ = 0.0
        self.loss_history_: list[float] = []

        for _ in range(self.n_iterations):
            for xi, yi in zip(X_fit, y_train):
                margin = yi * (xi @ self.w_ + self.b_)

                if margin < 1:
                    self.w_ -= self.learning_rate * ((2.0 / len(y_train)) * self.w_ - self.C * yi * xi)
                    self.b_ += self.learning_rate * self.C * yi
                else:
                    self.w_ -= self.learning_rate * (2.0 / len(y_train)) * self.w_

            scores = X_fit @ self.w_ + self.b_
            hinge = np.maximum(0.0, 1.0 - y_train * scores)
            self.loss_history_.append(float(np.mean(hinge) + np.dot(self.w_, self.w_) / len(y_train)))

        margins = y_train * (X_fit @ self.w_ + self.b_)
        self.support_vectors_ = X[np.abs(margins - 1.0) <= 0.25]

        if len(self.support_vectors_) == 0:
            self.support_vectors_ = X[margins <= 1.0]

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X_fit = self._transform_features(np.asarray(X, dtype=float), fit=False)

        return X_fit @ self.w_ + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        signs = np.where(self.decision_function(X) >= 0.0, 1, -1)

        return np.where(signs == -1, self.classes_[0], self.classes_[1])

    def _transform_features(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if self.kernel == 'linear':
            return X

        if self.kernel != 'rbf':
            raise ValueError("kernel must be 'linear' or 'rbf'")

        if fit:
            self._rbf_sampler_ = RBFSampler(
                gamma=self.gamma,
                n_components=self.n_components,
                random_state=self.random_state,
            )

            return self._rbf_sampler_.fit_transform(X)

        return self._rbf_sampler_.transform(X)
