"""Custom classification and regression models for EnergyTypeNet."""

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.kernel_approximation import RBFSampler


@dataclass
class Node:
    """Tree node used by custom CART models."""

    feature_index: int | None = None
    threshold: float | None = None
    left: 'Node | None' = None
    right: 'Node | None' = None
    value: float | int | np.ndarray | None = None
    impurity: float = 0.0
    n_samples: int = 0


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


class DecisionTreeClassifierCustom(ClassifierMixin, BaseEstimator):
    """CART classifier with binary splits and Gini or entropy impurity."""

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        criterion: str = 'gini',
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion

    def __repr__(self) -> str:
        return (
            f'DecisionTreeClassifierCustom(max_depth={self.max_depth}, '
            f'min_samples_split={self.min_samples_split}, criterion={self.criterion!r})'
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifierCustom':
        if self.criterion not in {'gini', 'entropy'}:
            raise ValueError("criterion must be 'gini' or 'entropy'")

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self._feature_importances_raw_ = np.zeros(self.n_features_in_)
        self.root_ = self._build_tree(np.asarray(X, dtype=float), np.asarray(y), depth=0)

        total = self._feature_importances_raw_.sum()
        self.feature_importances_ = (
            self._feature_importances_raw_ / total
            if total > 0
            else self._feature_importances_raw_
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.root_) for x in np.asarray(X, dtype=float)])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        leaf_values = [self._traverse_tree(x, self.root_, return_distribution=True) for x in np.asarray(X, dtype=float)]

        return np.vstack(leaf_values)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        n_samples = len(y)
        impurity = self._impurity(y)
        leaf_value = self._leaf_distribution(y)

        if (
            n_samples < self.min_samples_split
            or len(np.unique(y)) == 1
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return Node(value=leaf_value, impurity=impurity, n_samples=n_samples)

        split = self._best_split(X, y)
        if split is None:
            return Node(value=leaf_value, impurity=impurity, n_samples=n_samples)

        feature_index, threshold, gain, left_mask, right_mask = split
        self._feature_importances_raw_[feature_index] += gain * n_samples

        return Node(
            feature_index=feature_index,
            threshold=threshold,
            left=self._build_tree(X[left_mask], y[left_mask], depth + 1),
            right=self._build_tree(X[right_mask], y[right_mask], depth + 1),
            impurity=impurity,
            n_samples=n_samples,
        )

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        parent_impurity = self._impurity(y)
        best_gain = 0.0
        best_split = None

        for feature_index in range(X.shape[1]):
            values = np.unique(X[:, feature_index])
            thresholds = (values[:-1] + values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if not left_mask.any() or not right_mask.any():
                    continue

                n_left = left_mask.sum()
                n_right = right_mask.sum()
                child_impurity = (
                    n_left / len(y) * self._impurity(y[left_mask])
                    + n_right / len(y) * self._impurity(y[right_mask])
                )
                gain = parent_impurity - child_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_index, threshold, gain, left_mask, right_mask)

        return best_split

    def _impurity(self, y: np.ndarray) -> float:
        if self.criterion == 'entropy':
            return self._entropy(y)

        return self._gini(y)

    def _leaf_distribution(self, y: np.ndarray) -> np.ndarray:
        counts = np.array([np.sum(y == c) for c in self.classes_], dtype=float)

        return counts / counts.sum()

    def _traverse_tree(self, x: np.ndarray, node: Node, return_distribution: bool = False):
        if node.value is not None:
            if return_distribution:
                return node.value

            return self.classes_[np.argmax(node.value)]

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left, return_distribution)

        return self._traverse_tree(x, node.right, return_distribution)

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()

        return float(1.0 - np.sum(probabilities ** 2))

    @staticmethod
    def _entropy(y: np.ndarray) -> float:
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        probabilities = probabilities[probabilities > 0]

        return float(-np.sum(probabilities * np.log2(probabilities)))


class DecisionTreeRegressorCustom(RegressorMixin, BaseEstimator):
    """CART regressor using MSE reduction for binary splits."""

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        criterion: str = 'mse',
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion

    def __repr__(self) -> str:
        return (
            f'DecisionTreeRegressorCustom(max_depth={self.max_depth}, '
            f'min_samples_split={self.min_samples_split}, criterion={self.criterion!r})'
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeRegressorCustom':
        self.n_features_in_ = X.shape[1]
        self._feature_importances_raw_ = np.zeros(self.n_features_in_)
        self.root_ = self._build_tree(np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel(), depth=0)

        total = self._feature_importances_raw_.sum()
        self.feature_importances_ = (
            self._feature_importances_raw_ / total
            if total > 0
            else self._feature_importances_raw_
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.root_) for x in np.asarray(X, dtype=float)])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        n_samples = len(y)
        impurity = self._mse(y)

        if (
            n_samples < self.min_samples_split
            or np.allclose(y, y[0])
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return Node(value=float(np.mean(y)), impurity=impurity, n_samples=n_samples)

        split = self._best_split(X, y)
        if split is None:
            return Node(value=float(np.mean(y)), impurity=impurity, n_samples=n_samples)

        feature_index, threshold, gain, left_mask, right_mask = split
        self._feature_importances_raw_[feature_index] += gain * n_samples

        return Node(
            feature_index=feature_index,
            threshold=threshold,
            left=self._build_tree(X[left_mask], y[left_mask], depth + 1),
            right=self._build_tree(X[right_mask], y[right_mask], depth + 1),
            impurity=impurity,
            n_samples=n_samples,
        )

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        parent_mse = self._mse(y)
        best_gain = 0.0
        best_split = None

        for feature_index in range(X.shape[1]):
            values = np.unique(X[:, feature_index])
            thresholds = (values[:-1] + values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if not left_mask.any() or not right_mask.any():
                    continue

                n_left = left_mask.sum()
                n_right = right_mask.sum()
                child_mse = (
                    n_left / len(y) * self._mse(y[left_mask])
                    + n_right / len(y) * self._mse(y[right_mask])
                )
                gain = parent_mse - child_mse

                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_index, threshold, gain, left_mask, right_mask)

        return best_split

    def _traverse_tree(self, x: np.ndarray, node: Node) -> float:
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)

    @staticmethod
    def _mse(y: np.ndarray) -> float:
        return float(np.mean((y - np.mean(y)) ** 2))


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
