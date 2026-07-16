"""Custom classification and regression models for EnergyTypeNet."""

import inspect
from dataclasses import dataclass
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, clone
from sklearn.kernel_approximation import RBFSampler

class KMeansCustom(TransformerMixin, BaseEstimator):
    """K-Means clustering implemented with NumPy."""

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
        random_state: int | None = None,
        init: str = 'k-means++',
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.init = init

    def __repr__(self) -> str:
        return (
            f'KMeansCustom(n_clusters={self.n_clusters}, max_iter={self.max_iter}, '
            f'n_init={self.n_init}, init={self.init!r})'
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> 'KMeansCustom':
        X = np.asarray(X, dtype=float)
        rng = np.random.RandomState(self.random_state)
        best_inertia = np.inf
        best_centers = None
        best_labels = None
        best_iter = 0

        for _ in range(self.n_init):
            centers = self._initialize_centers(X, rng)

            for iteration in range(1, self.max_iter + 1):
                distances = self._squared_distances(X, centers)
                labels = np.argmin(distances, axis=1)
                new_centers = centers.copy()

                for cluster_idx in range(self.n_clusters):
                    mask = labels == cluster_idx
                    if np.any(mask):
                        new_centers[cluster_idx] = X[mask].mean(axis=0)

                shift = np.linalg.norm(new_centers - centers)
                centers = new_centers

                if shift <= self.tol:
                    break

            distances = self._squared_distances(X, centers)
            labels = np.argmin(distances, axis=1)
            inertia = float(np.sum(distances[np.arange(X.shape[0]), labels]))

            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()
                best_iter = iteration

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_iter

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = self._squared_distances(np.asarray(X, dtype=float), self.cluster_centers_)

        return np.argmin(distances, axis=1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.sqrt(self._squared_distances(np.asarray(X, dtype=float), self.cluster_centers_))

    def fit_predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).labels_

    def _initialize_centers(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        if self.init == 'random':
            indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)

            return X[indices].copy()

        if self.init != 'k-means++':
            raise ValueError("init must be either 'random' or 'k-means++'")

        centers = [X[rng.randint(X.shape[0])]]

        for _ in range(1, self.n_clusters):
            distances = self._squared_distances(X, np.asarray(centers))
            closest_distances = np.min(distances, axis=1)
            total = closest_distances.sum()

            if total <= 0:
                next_idx = rng.randint(X.shape[0])
            else:
                probabilities = closest_distances / total
                next_idx = rng.choice(X.shape[0], p=probabilities)

            centers.append(X[next_idx])

        return np.asarray(centers, dtype=float)

    @staticmethod
    def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        return np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)

class DBSCANCustom(BaseEstimator):
    """DBSCAN is transductive; call fit_predict on the full dataset."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5, metric: str = 'euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def __repr__(self) -> str:
        return f'DBSCANCustom(eps={self.eps}, min_samples={self.min_samples})'

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> 'DBSCANCustom':
        X = np.asarray(X, dtype=float)
        distances = self._pairwise_distances(X)
        neighbors = [np.flatnonzero(distances[i] <= self.eps) for i in range(X.shape[0])]
        core_mask = np.array([len(n) >= self.min_samples for n in neighbors])
        labels = np.full(X.shape[0], -1, dtype=int)
        cluster_id = 0

        for point_idx in range(X.shape[0]):
            if labels[point_idx] != -1 or not core_mask[point_idx]:
                continue

            labels[point_idx] = cluster_id
            stack = list(neighbors[point_idx])

            while stack:
                neighbor_idx = stack.pop()
                if labels[neighbor_idx] == -1:
                    labels[neighbor_idx] = cluster_id
                if not core_mask[neighbor_idx]:
                    continue

                for candidate in neighbors[neighbor_idx]:
                    if labels[candidate] == -1:
                        stack.append(candidate)

            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.flatnonzero(core_mask)

        return self

    def fit_predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).labels_

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        if self.metric != 'euclidean':
            raise ValueError("metric must be 'euclidean'")

        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]

        return np.sqrt(np.sum(diff ** 2, axis=2))

class GaussianMixtureModelCustom(BaseEstimator):
    """Gaussian mixture model fitted with the EM algorithm."""

    def __init__(
        self,
        n_components: int = 3,
        max_iter: int = 100,
        tol: float = 1e-3,
        random_state: int | None = None,
        reg_covar: float = 1e-6,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.reg_covar = reg_covar

    def __repr__(self) -> str:
        return (
            f'GaussianMixtureModelCustom(n_components={self.n_components}, '
            f'max_iter={self.max_iter}, tol={self.tol})'
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> 'GaussianMixtureModelCustom':
        X = np.asarray(X, dtype=float)
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        labels = KMeansCustom(
            n_clusters=self.n_components,
            n_init=3,
            random_state=self.random_state,
        ).fit_predict(X)
        responsibilities = np.zeros((n_samples, self.n_components))
        responsibilities[np.arange(n_samples), labels] = 1.0
        responsibilities += 1e-3 * rng.rand(n_samples, self.n_components)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        self.lower_bound_: list[float] = []

        for _ in range(self.max_iter):
            nk = responsibilities.sum(axis=0) + 10 * np.finfo(float).eps
            self.weights_ = nk / n_samples
            self.means_ = (responsibilities.T @ X) / nk[:, np.newaxis]
            self.covariances_ = np.zeros((self.n_components, n_features, n_features))

            for component in range(self.n_components):
                centered = X - self.means_[component]
                weighted = centered * responsibilities[:, component][:, np.newaxis]
                covariance = (weighted.T @ centered) / nk[component]
                self.covariances_[component] = covariance + self.reg_covar * np.eye(n_features)

            log_prob = self._estimate_log_prob(X) + np.log(self.weights_ + 1e-12)
            log_norm = self._logsumexp(log_prob, axis=1)
            lower_bound = float(np.mean(log_norm))
            self.lower_bound_.append(lower_bound)
            responsibilities = np.exp(log_prob - log_norm[:, np.newaxis])

            if len(self.lower_bound_) > 1 and abs(self.lower_bound_[-1] - self.lower_bound_[-2]) < self.tol:
                break

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_prob = self._estimate_log_prob(np.asarray(X, dtype=float)) + np.log(self.weights_ + 1e-12)
        log_norm = self._logsumexp(log_prob, axis=1)

        return np.exp(log_prob - log_norm[:, np.newaxis])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray) -> float:
        log_prob = self._estimate_log_prob(np.asarray(X, dtype=float)) + np.log(self.weights_ + 1e-12)

        return float(np.mean(self._logsumexp(log_prob, axis=1)))

    def _estimate_log_prob(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        log_prob = np.empty((n_samples, self.n_components))

        for component in range(self.n_components):
            covariance = self.covariances_[component]
            precision = np.linalg.pinv(covariance)
            sign, log_det = np.linalg.slogdet(covariance)

            if sign <= 0:
                log_det = np.log(np.linalg.det(covariance + self.reg_covar * np.eye(covariance.shape[0])))

            centered = X - self.means_[component]
            mahalanobis = np.sum((centered @ precision) * centered, axis=1)
            log_prob[:, component] = -0.5 * (
                X.shape[1] * np.log(2.0 * np.pi) + log_det + mahalanobis
            )

        return log_prob

    @staticmethod
    def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
        max_values = np.max(values, axis=axis, keepdims=True)
        summed = np.sum(np.exp(values - max_values), axis=axis, keepdims=True)

        return np.squeeze(max_values + np.log(summed + 1e-12), axis=axis)

class AgglomerativeCustom(BaseEstimator):
    """Agglomerative clustering with several linkage criteria."""

    def __init__(self, n_clusters: int = 3, linkage: str = 'ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def __repr__(self) -> str:
        return f'AgglomerativeCustom(n_clusters={self.n_clusters}, linkage={self.linkage!r})'

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> 'AgglomerativeCustom':
        X = np.asarray(X, dtype=float)
        if self.linkage not in {'single', 'complete', 'average', 'ward'}:
            raise ValueError("linkage must be one of 'single', 'complete', 'average', 'ward'")

        clusters = {idx: [idx] for idx in range(X.shape[0])}
        next_id = X.shape[0]
        rows = []

        while len(clusters) > 1:
            keys = list(clusters)
            best_pair = None
            best_distance = np.inf

            for i, left_key in enumerate(keys[:-1]):
                for right_key in keys[i + 1:]:
                    distance = self._linkage_distance(X, clusters[left_key], clusters[right_key])
                    if distance < best_distance:
                        best_distance = distance
                        best_pair = (left_key, right_key)

            left_key, right_key = best_pair
            merged = clusters.pop(left_key) + clusters.pop(right_key)
            rows.append([left_key, right_key, best_distance, len(merged)])
            clusters[next_id] = merged
            next_id += 1

        self.linkage_matrix_ = np.asarray(rows, dtype=float)
        self.n_samples_ = X.shape[0]
        self.labels_ = self._cut_tree()

        return self

    def fit_predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        self.fit(X, y)

        return self._cut_tree()

    def _cut_tree(self) -> np.ndarray:
        clusters = {idx: [idx] for idx in range(self.n_samples_)}
        next_id = self.n_samples_

        for row in self.linkage_matrix_[: self.n_samples_ - self.n_clusters]:
            left_key, right_key = int(row[0]), int(row[1])
            merged = clusters.pop(left_key) + clusters.pop(right_key)
            clusters[next_id] = merged
            next_id += 1

        labels = np.empty(self.n_samples_, dtype=int)
        for label, members in enumerate(clusters.values()):
            labels[members] = label

        return labels

    def _linkage_distance(self, X: np.ndarray, left: list[int], right: list[int]) -> float:
        left_points = X[left]
        right_points = X[right]

        if self.linkage == 'ward':
            left_mean = left_points.mean(axis=0)
            right_mean = right_points.mean(axis=0)
            factor = len(left) * len(right) / (len(left) + len(right))

            return float(np.sqrt(factor) * np.linalg.norm(left_mean - right_mean))

        distances = np.sqrt(np.sum((left_points[:, np.newaxis, :] - right_points[np.newaxis, :, :]) ** 2, axis=2))

        if self.linkage == 'single':
            return float(distances.min())
        if self.linkage == 'complete':
            return float(distances.max())

        return float(distances.mean())
