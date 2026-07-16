"""Custom classification and regression models for EnergyTypeNet."""

import inspect
from dataclasses import dataclass
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, clone
from sklearn.kernel_approximation import RBFSampler

class PCACustom(TransformerMixin, BaseEstimator):
    """Principal Component Analysis implemented with NumPy."""

    def __init__(
        self,
        n_components: int = 2,
        whiten: bool = False,
        use_svd: bool = False,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.whiten = whiten
        self.use_svd = use_svd
        self.random_state = random_state

    def __repr__(self) -> str:
        return (
            f'PCACustom(n_components={self.n_components}, whiten={self.whiten}, '
            f'use_svd={self.use_svd})'
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> 'PCACustom':
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        n_samples, n_features = X_centered.shape
        n_components = min(self.n_components, n_features)

        if self.use_svd:
            _, singular_values, vt = np.linalg.svd(X_centered, full_matrices=False)
            eigenvalues = (singular_values ** 2) / max(n_samples - 1, 1)
            components = vt
        else:
            covariance = np.cov(X_centered, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            components = eigenvectors[:, order].T

        eigenvalues = np.maximum(eigenvalues, 0.0)
        total_variance = eigenvalues.sum()
        self.components_ = components[:n_components]
        self.explained_variance_ = eigenvalues[:n_components]
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_variance
            if total_variance > 0
            else np.zeros_like(self.explained_variance_)
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

        if self.whiten:
            return Z / np.sqrt(self.explained_variance_ + 1e-12)

        return Z

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=float)

        if self.whiten:
            Z = Z * np.sqrt(self.explained_variance_ + 1e-12)

        return Z @ self.components_ + self.mean_

class LDACustom(TransformerMixin, BaseEstimator):
    """Linear Discriminant Analysis projection implemented with NumPy."""

    def __init__(self, n_components: int = 2, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state

    def __repr__(self) -> str:
        return f'LDACustom(n_components={self.n_components})'

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LDACustom':
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.xbar_ = X.mean(axis=0)
        n_features = X.shape[1]
        within_scatter = np.zeros((n_features, n_features))
        between_scatter = np.zeros((n_features, n_features))

        for cls in self.classes_:
            X_cls = X[y == cls]
            class_mean = X_cls.mean(axis=0)
            centered = X_cls - class_mean
            within_scatter += centered.T @ centered

            mean_diff = (class_mean - self.xbar_).reshape(-1, 1)
            between_scatter += X_cls.shape[0] * (mean_diff @ mean_diff.T)

        matrix = np.linalg.pinv(within_scatter) @ between_scatter
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        n_components = min(self.n_components, len(self.classes_) - 1, n_features)

        self.scalings_ = eigenvectors[:, :n_components]
        total = eigenvalues.sum()
        self.explained_variance_ratio_ = (
            eigenvalues[:n_components] / total
            if total > 0
            else np.zeros(n_components)
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.xbar_) @ self.scalings_

class KernelPCACustom(TransformerMixin, BaseEstimator):
    """Kernel PCA implemented with NumPy."""

    def __init__(
        self,
        n_components: int = 2,
        kernel: str = 'rbf',
        gamma: float | None = None,
        degree: int = 3,
        coef0: float = 1.0,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.random_state = random_state

    def __repr__(self) -> str:
        return (
            f'KernelPCACustom(n_components={self.n_components}, kernel={self.kernel!r}, '
            f'gamma={self.gamma}, degree={self.degree}, coef0={self.coef0})'
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> 'KernelPCACustom':
        X = np.asarray(X, dtype=float)
        self.X_fit_ = X
        K = self._compute_kernel(X, X)
        self.K_fit_rows_mean_ = K.mean(axis=1)
        self.K_fit_total_mean_ = K.mean()
        K_centered = self._center_fit_kernel(K)
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        positive = eigenvalues > 1e-12
        eigenvalues = eigenvalues[positive]
        eigenvectors = eigenvectors[:, positive]
        n_components = min(self.n_components, eigenvalues.shape[0])

        self.lambdas_ = eigenvalues[:n_components]
        self.alphas_ = eigenvectors[:, :n_components] / np.sqrt(self.lambdas_ + 1e-12)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        K_new = self._compute_kernel(np.asarray(X, dtype=float), self.X_fit_)
        K_new_centered = self._center_new_kernel(K_new)

        return K_new_centered @ self.alphas_

    def _center_fit_kernel(self, K: np.ndarray) -> np.ndarray:
        return (
            K
            - self.K_fit_rows_mean_[np.newaxis, :]
            - self.K_fit_rows_mean_[:, np.newaxis]
            + self.K_fit_total_mean_
        )

    def _center_new_kernel(self, K_new: np.ndarray) -> np.ndarray:
        return (
            K_new
            - K_new.mean(axis=1, keepdims=True)
            - self.K_fit_rows_mean_[np.newaxis, :]
            + self.K_fit_total_mean_
        )

    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        gamma = self.gamma if self.gamma is not None else 1.0 / X.shape[1]

        if self.kernel == 'linear':
            return X @ Y.T
        if self.kernel == 'poly':
            return (gamma * (X @ Y.T) + self.coef0) ** self.degree
        if self.kernel == 'sigmoid':
            return np.tanh(gamma * (X @ Y.T) + self.coef0)
        if self.kernel == 'rbf':
            X_norm = np.sum(X ** 2, axis=1)[:, np.newaxis]
            Y_norm = np.sum(Y ** 2, axis=1)[np.newaxis, :]
            distances = X_norm + Y_norm - 2.0 * (X @ Y.T)

            return np.exp(-gamma * np.maximum(distances, 0.0))

        raise ValueError("kernel must be one of 'rbf', 'poly', 'linear', 'sigmoid'")
