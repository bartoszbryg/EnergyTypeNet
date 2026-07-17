"""Custom bagging and boosting models."""

import inspect
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone

from .trees import DecisionTreeClassifierCustom, DecisionTreeRegressorCustom

__all__ = ["BaggingClassifierCustom", "BaggingRegressorCustom", "AdaBoostClassifierCustom"]

class BaggingClassifierCustom(ClassifierMixin, BaseEstimator):
    """Bootstrap aggregation classifier using custom decision trees by default."""

    def __init__(
        self,
        base_estimator=None,
        n_estimators: int = 10,
        max_samples: float | int = 1.0,
        max_features: float | int = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        random_state: int | None = None,
        n_jobs: int = 1,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.random_state = random_state
        self.n_jobs = n_jobs

    def __repr__(self) -> str:
        return (
            f'BaggingClassifierCustom(n_estimators={self.n_estimators}, '
            f'max_samples={self.max_samples}, max_features={self.max_features}, '
            f'bootstrap={self.bootstrap})'
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaggingClassifierCustom':
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError('X must be a 2D array')
        if len(X) != len(y):
            raise ValueError('X and y must contain the same number of samples')
        if self.n_estimators < 1:
            raise ValueError('n_estimators must be at least 1')
        if self.oob_score and not self.bootstrap:
            raise ValueError('OOB scoring requires bootstrap=True')

        rng = np.random.RandomState(self.random_state)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.estimators_ = []
        self.estimators_features_ = []
        self.estimators_samples_ = []

        sample_size = self._resolve_size(self.max_samples, X.shape[0], 'max_samples')
        feature_size = self._resolve_size(self.max_features, X.shape[1], 'max_features')
        base = self.base_estimator or DecisionTreeClassifierCustom(max_depth=None)

        for _ in range(self.n_estimators):
            sample_idx = rng.choice(X.shape[0], size=sample_size, replace=self.bootstrap)
            feature_idx = rng.choice(
                X.shape[1],
                size=feature_size,
                replace=self.bootstrap_features,
            )
            estimator = self._fresh_estimator(base, rng)
            estimator.fit(X[sample_idx][:, feature_idx], y[sample_idx])

            self.estimators_.append(estimator)
            self.estimators_features_.append(feature_idx)
            self.estimators_samples_.append(sample_idx)

        self.oob_score_ = self._compute_oob_score(X, y) if self.oob_score else None

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        proba = np.zeros((X.shape[0], len(self.classes_)))

        for estimator, features in zip(self.estimators_, self.estimators_features_):
            proba += self._aligned_proba(estimator, X[:, features])

        return proba / len(self.estimators_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    @property
    def feature_importances_(self) -> np.ndarray:
        importances = np.zeros(self.n_features_in_)
        contribution_counts = np.zeros(self.n_features_in_)

        for estimator, features in zip(self.estimators_, self.estimators_features_):
            if not hasattr(estimator, 'feature_importances_'):
                continue

            for local_idx, feature_idx in enumerate(features):
                importances[feature_idx] += estimator.feature_importances_[local_idx]
                contribution_counts[feature_idx] += 1

        mask = contribution_counts > 0
        importances[mask] /= contribution_counts[mask]

        total = importances.sum()

        return importances / total if total > 0 else importances

    def _compute_oob_score(self, X: np.ndarray, y: np.ndarray) -> float | None:
        votes = np.zeros((X.shape[0], len(self.classes_)))
        counts = np.zeros(X.shape[0])
        all_idx = np.arange(X.shape[0])

        for estimator, features, sample_idx in zip(
            self.estimators_,
            self.estimators_features_,
            self.estimators_samples_,
        ):
            oob_idx = np.setdiff1d(all_idx, np.unique(sample_idx), assume_unique=False)
            if len(oob_idx) == 0:
                continue

            votes[oob_idx] += self._aligned_proba(estimator, X[oob_idx][:, features])
            counts[oob_idx] += 1

        mask = counts > 0
        if not mask.any():
            return None

        pred = self.classes_[np.argmax(votes[mask] / counts[mask, None], axis=1)]

        return float(np.mean(pred == y[mask]))

    def _aligned_proba(self, estimator, X_subset: np.ndarray) -> np.ndarray:
        raw = estimator.predict_proba(X_subset)
        aligned = np.zeros((X_subset.shape[0], len(self.classes_)))

        for local_idx, cls in enumerate(estimator.classes_):
            global_idx = np.where(self.classes_ == cls)[0][0]
            aligned[:, global_idx] = raw[:, local_idx]

        return aligned

    @staticmethod
    def _fresh_estimator(base, rng: np.random.RandomState):
        estimator = clone(base)
        seed = int(rng.randint(0, np.iinfo(np.int32).max))

        try:
            params = estimator.get_params(deep=False)
        except AttributeError:
            params = {}

        if 'random_state' in params:
            estimator.set_params(random_state=seed)
        elif hasattr(estimator, 'random_state'):
            estimator.random_state = seed

        return estimator

    @staticmethod
    def _resolve_size(value: float | int, upper: int, name: str) -> int:
        if isinstance(value, float):
            if not 0 < value <= 1:
                raise ValueError(f'{name} as a float must be in (0, 1]')

            return max(1, int(np.ceil(value * upper)))

        if not 1 <= value <= upper:
            raise ValueError(f'{name} as an int must be in [1, {upper}]')

        return int(value)

class BaggingRegressorCustom(RegressorMixin, BaseEstimator):
    """Bootstrap aggregation regressor using custom regression trees by default."""

    def __init__(
        self,
        base_estimator=None,
        n_estimators: int = 10,
        max_samples: float | int = 1.0,
        max_features: float | int = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        random_state: int | None = None,
        n_jobs: int = 1,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.random_state = random_state
        self.n_jobs = n_jobs

    def __repr__(self) -> str:
        return (
            f'BaggingRegressorCustom(n_estimators={self.n_estimators}, '
            f'max_samples={self.max_samples}, max_features={self.max_features}, '
            f'bootstrap={self.bootstrap})'
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaggingRegressorCustom':
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if X.ndim != 2:
            raise ValueError('X must be a 2D array')
        if len(X) != len(y):
            raise ValueError('X and y must contain the same number of samples')
        if self.n_estimators < 1:
            raise ValueError('n_estimators must be at least 1')
        if self.oob_score and not self.bootstrap:
            raise ValueError('OOB scoring requires bootstrap=True')

        rng = np.random.RandomState(self.random_state)

        self.n_features_in_ = X.shape[1]
        self.estimators_ = []
        self.estimators_features_ = []
        self.estimators_samples_ = []
        sample_size = BaggingClassifierCustom._resolve_size(
            self.max_samples,
            X.shape[0],
            'max_samples',
        )
        feature_size = BaggingClassifierCustom._resolve_size(
            self.max_features,
            X.shape[1],
            'max_features',
        )
        base = self.base_estimator or DecisionTreeRegressorCustom(max_depth=None)

        for _ in range(self.n_estimators):
            sample_idx = rng.choice(X.shape[0], size=sample_size, replace=self.bootstrap)
            feature_idx = rng.choice(
                X.shape[1],
                size=feature_size,
                replace=self.bootstrap_features,
            )
            estimator = BaggingClassifierCustom._fresh_estimator(base, rng)
            estimator.fit(X[sample_idx][:, feature_idx], y[sample_idx])

            self.estimators_.append(estimator)
            self.estimators_features_.append(feature_idx)
            self.estimators_samples_.append(sample_idx)

        self.oob_score_ = self._compute_oob_score(X, y) if self.oob_score else None

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        predictions = [
            estimator.predict(X[:, features])
            for estimator, features in zip(self.estimators_, self.estimators_features_)
        ]

        return np.mean(predictions, axis=0)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).ravel()
        pred = self.predict(X)
        total = np.sum((y - y.mean()) ** 2)

        if total == 0:
            return 0.0

        return float(1.0 - np.sum((y - pred) ** 2) / total)

    @property
    def feature_importances_(self) -> np.ndarray:
        importances = np.zeros(self.n_features_in_)
        contribution_counts = np.zeros(self.n_features_in_)

        for estimator, features in zip(self.estimators_, self.estimators_features_):
            if not hasattr(estimator, 'feature_importances_'):
                continue

            for local_idx, feature_idx in enumerate(features):
                importances[feature_idx] += estimator.feature_importances_[local_idx]
                contribution_counts[feature_idx] += 1

        mask = contribution_counts > 0
        importances[mask] /= contribution_counts[mask]
        total = importances.sum()

        return importances / total if total > 0 else importances

    def _compute_oob_score(self, X: np.ndarray, y: np.ndarray) -> float | None:
        predictions = np.zeros(X.shape[0])
        counts = np.zeros(X.shape[0])
        all_idx = np.arange(X.shape[0])

        for estimator, features, sample_idx in zip(
            self.estimators_,
            self.estimators_features_,
            self.estimators_samples_,
        ):
            oob_idx = np.setdiff1d(all_idx, np.unique(sample_idx), assume_unique=False)
            if len(oob_idx) == 0:
                continue

            predictions[oob_idx] += estimator.predict(X[oob_idx][:, features])
            counts[oob_idx] += 1

        mask = counts > 0
        if not mask.any():
            return None

        oob_pred = predictions[mask] / counts[mask]
        total = np.sum((y[mask] - y[mask].mean()) ** 2)

        if total == 0:
            return 0.0

        return float(1.0 - np.sum((y[mask] - oob_pred) ** 2) / total)

class AdaBoostClassifierCustom(ClassifierMixin, BaseEstimator):
    """Multi-class SAMME AdaBoost classifier using custom decision stumps."""

    def __init__(
        self,
        base_estimator=None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: int | None = None,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def __repr__(self) -> str:
        return (
            f'AdaBoostClassifierCustom(n_estimators={self.n_estimators}, '
            f'learning_rate={self.learning_rate}, random_state={self.random_state})'
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaBoostClassifierCustom':
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples = X.shape[0]

        if n_classes < 2:
            raise ValueError('AdaBoostClassifierCustom needs at least two classes')

        sample_weight = np.full(n_samples, 1.0 / n_samples)
        base = self.base_estimator or DecisionTreeClassifierCustom(max_depth=1)

        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        self.sample_weight_history_ = []

        for _ in range(self.n_estimators):
            estimator = BaggingClassifierCustom._fresh_estimator(base, rng)
            fit_signature = inspect.signature(estimator.fit)

            if 'sample_weight' in fit_signature.parameters:
                estimator.fit(X, y, sample_weight=sample_weight)
            else:
                draw_idx = rng.choice(n_samples, size=n_samples, replace=True, p=sample_weight)
                estimator.fit(X[draw_idx], y[draw_idx])

            pred = estimator.predict(X)
            incorrect = pred != y
            err = float(np.dot(sample_weight, incorrect) / sample_weight.sum())

            if err <= 1e-12:
                self.estimators_.append(estimator)
                self.estimator_weights_.append(float(self.learning_rate * 1e6))
                self.estimator_errors_.append(0.0)
                self.sample_weight_history_.append(sample_weight.copy())
                break

            if err >= 1 - (1 / n_classes):
                break

            err = np.clip(err, 1e-12, 1 - 1e-12)
            alpha = self.learning_rate * (np.log((1 - err) / err) + np.log(n_classes - 1))
            sample_weight *= np.exp(alpha * incorrect)
            sample_weight /= sample_weight.sum()

            self.estimators_.append(estimator)
            self.estimator_weights_.append(float(alpha))
            self.estimator_errors_.append(float(err))
            self.sample_weight_history_.append(sample_weight.copy())

        if not self.estimators_:
            estimator = clone(base)
            estimator.fit(X, y)
            self.estimators_.append(estimator)
            self.estimator_weights_.append(1.0)
            self.estimator_errors_.append(0.5)
            self.sample_weight_history_.append(sample_weight.copy())

        self.estimator_weights_ = np.array(self.estimator_weights_, dtype=float)
        self.estimator_errors_ = np.array(self.estimator_errors_, dtype=float)
        self.sample_weight_history_ = np.array(self.sample_weight_history_, dtype=float)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self._class_scores(X), axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self._class_scores(X)
        scores -= scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)

        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def staged_predict(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        scores = np.zeros((X.shape[0], len(self.classes_)))

        for estimator, alpha in zip(self.estimators_, self.estimator_weights_):
            pred = estimator.predict(X)
            for class_idx, cls in enumerate(self.classes_):
                scores[:, class_idx] += alpha * (pred == cls)

            yield self.classes_[np.argmax(scores, axis=1)]

    def _class_scores(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        scores = np.zeros((X.shape[0], len(self.classes_)))

        for estimator, alpha in zip(self.estimators_, self.estimator_weights_):
            pred = estimator.predict(X)
            for class_idx, cls in enumerate(self.classes_):
                scores[:, class_idx] += alpha * (pred == cls)

        return scores
