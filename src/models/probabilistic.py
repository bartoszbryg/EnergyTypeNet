"""Custom probabilistic classification and regression models."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

__all__ = ["GaussianNaiveBayes", "MultinomialNaiveBayes", "BernoulliNaiveBayes", "BayesianLinearRegression"]

class GaussianNaiveBayes(ClassifierMixin, BaseEstimator):
    """Gaussian Naive Bayes classifier implemented with NumPy."""

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def __repr__(self) -> str:
        return f'GaussianNaiveBayes(var_smoothing={self.var_smoothing})'

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNaiveBayes':
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()

        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_priors_ = counts / counts.sum()
        self.means_ = np.vstack([X[y == c].mean(axis=0) for c in self.classes_])
        self.variances_ = np.vstack([X[y == c].var(axis=0) for c in self.classes_])
        self.variances_ += self.var_smoothing * max(float(np.var(X)), 1.0)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_proba = self._joint_log_likelihood(X)
        log_norm = self._logsumexp(log_proba, axis=1)

        return np.exp(log_proba - log_norm[:, np.newaxis])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        log_priors = np.log(self.class_priors_ + 1e-12)
        log_likelihoods = []

        for mean, var in zip(self.means_, self.variances_):
            likelihood = -0.5 * np.sum(
                np.log(2.0 * np.pi * var) + ((X - mean) ** 2) / var,
                axis=1,
            )
            log_likelihoods.append(likelihood)

        return np.column_stack(log_likelihoods) + log_priors

    @staticmethod
    def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
        max_values = np.max(values, axis=axis, keepdims=True)

        return (
            max_values
            + np.log(np.sum(np.exp(values - max_values), axis=axis, keepdims=True))
        ).squeeze(axis)

class MultinomialNaiveBayes(ClassifierMixin, BaseEstimator):
    """Multinomial Naive Bayes for non-negative count features."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __repr__(self) -> str:
        return f'MultinomialNaiveBayes(alpha={self.alpha})'

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultinomialNaiveBayes':
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()

        if np.any(X < 0):
            raise ValueError('MultinomialNaiveBayes requires non-negative count features')

        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_log_prior_ = np.log(counts / counts.sum())
        class_feature_counts = np.vstack([X[y == c].sum(axis=0) for c in self.classes_])
        smoothed = class_feature_counts + self.alpha
        denominators = smoothed.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed / denominators)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if np.any(X < 0):
            raise ValueError('MultinomialNaiveBayes requires non-negative count features')

        log_proba = X @ self.feature_log_prob_.T + self.class_log_prior_
        log_norm = GaussianNaiveBayes._logsumexp(log_proba, axis=1)

        return np.exp(log_proba - log_norm[:, np.newaxis])

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        log_proba = X @ self.feature_log_prob_.T + self.class_log_prior_

        return self.classes_[np.argmax(log_proba, axis=1)]

class BernoulliNaiveBayes(ClassifierMixin, BaseEstimator):
    """Bernoulli Naive Bayes for binary or thresholded features."""

    def __init__(self, alpha: float = 1.0, binarize: float | None = 0.0):
        self.alpha = alpha
        self.binarize = binarize

    def __repr__(self) -> str:
        return f'BernoulliNaiveBayes(alpha={self.alpha}, binarize={self.binarize})'

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BernoulliNaiveBayes':
        X = self._binarize(X)
        y = np.asarray(y).ravel()

        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_log_prior_ = np.log(counts / counts.sum())
        positives = np.vstack([X[y == c].sum(axis=0) for c in self.classes_])
        denominators = counts[:, np.newaxis] + 2.0 * self.alpha
        self.feature_prob_ = (positives + self.alpha) / denominators
        self.feature_prob_ = np.clip(self.feature_prob_, 1e-12, 1.0 - 1e-12)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._binarize(X)
        log_prob_one = np.log(self.feature_prob_)
        log_prob_zero = np.log(1.0 - self.feature_prob_)
        log_proba = X @ log_prob_one.T + (1.0 - X) @ log_prob_zero.T
        log_proba += self.class_log_prior_
        log_norm = GaussianNaiveBayes._logsumexp(log_proba, axis=1)

        return np.exp(log_proba - log_norm[:, np.newaxis])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def _binarize(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        if self.binarize is None:
            return X

        return (X > self.binarize).astype(float)

class BayesianLinearRegression(RegressorMixin, BaseEstimator):
    """Bayesian linear regression with a conjugate Gaussian prior."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        fit_intercept: bool = True,
    ):
        self.alpha = alpha
        self.beta = beta
        self.fit_intercept = fit_intercept

    def __repr__(self) -> str:
        return (
            f'BayesianLinearRegression(alpha={self.alpha}, beta={self.beta}, '
            f'fit_intercept={self.fit_intercept})'
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegression':
        X_fit = self._add_intercept(X)
        y = np.asarray(y, dtype=float).ravel()

        precision = self.alpha * np.eye(X_fit.shape[1]) + self.beta * (X_fit.T @ X_fit)
        self.S_N_ = np.linalg.pinv(precision)
        self.m_N_ = self.beta * self.S_N_ @ X_fit.T @ y

        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_fit = self._add_intercept(X)
        means = X_fit @ self.m_N_
        variances = (1.0 / self.beta) + np.sum((X_fit @ self.S_N_) * X_fit, axis=1)

        return means, variances

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        if not self.fit_intercept:
            return X

        return np.c_[np.ones(X.shape[0]), X]
