"""Custom decision-tree models."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from ._base import Node

__all__ = ["DecisionTreeClassifierCustom", "DecisionTreeRegressorCustom"]

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
