import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.models import (
    ActivationFunctions,
    AdalineGD,
    AgglomerativeCustom,
    AttentionClassifier,
    BayesianLinearRegression,
    BernoulliNaiveBayes,
    DBSCANCustom,
    DecisionTreeClassifierCustom,
    DecisionTreeRegressorCustom,
    ElasticNetCustom,
    GaussianMixtureModelCustom,
    GaussianNaiveBayes,
    KMeansCustom,
    LassoRegressionCustom,
    LinearRegressionGD,
    LinearRegressionNormal,
    LogisticRegressionSoftmax,
    MLPCustom,
    KernelPCACustom,
    LDACustom,
    MultinomialNaiveBayes,
    PCACustom,
    Perceptron,
    RegularizedLogisticRegression,
    RidgeRegressionCustom,
    SVMClassifierCustom,
)


def test_attention_predict_proba_shape_and_sum():
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [5.0, 5.0],
        [6.0, 6.0],
    ])
    y = np.array([0, 0, 1, 1])
    model = AttentionClassifier(w=1.0).fit(X, y)

    proba = model.predict_proba(np.array([[0.5, 0.5], [5.5, 5.5]]))

    assert proba.shape == (2, 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(2))


def test_softmax_loss_decreases():
    X = np.array([
        [0.0, 0.0],
        [0.2, 0.1],
        [2.0, 2.0],
        [2.2, 2.1],
        [4.0, 4.0],
        [4.2, 4.1],
    ])
    y = np.array([0, 0, 1, 1, 2, 2])
    model = LogisticRegressionSoftmax(eta=0.1, n_iter=80, alpha=0.0)
    model.fit(X, y)

    assert model.loss_[0] > model.loss_[-1]
    assert model.predict(X).shape == y.shape


def test_linear_regression_gd_learns_simple_line():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = 3 * X.ravel() + 2

    model = LinearRegressionGD(learning_rate=0.001, n_iterations=5000)
    model.fit(X, y)

    assert model.loss_history_[0] > model.loss_history_[-1]
    np.testing.assert_allclose(model.predict([[10.0]]), [32.0], atol=0.5)


def test_linear_regression_normal_solves_simple_line():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = 3 * X.ravel() + 2

    model = LinearRegressionNormal()
    model.fit(X, y)

    np.testing.assert_allclose(model.predict([[10.0]]), [32.0], atol=1e-8)


def test_perceptron_separates_blobs():
    X, y = make_blobs(
        n_samples=80,
        centers=[[-2, -2], [2, 2]],
        cluster_std=0.4,
        random_state=42,
    )

    model = Perceptron(learning_rate=1.0, n_iterations=20)
    model.fit(X, y)

    assert model.errors_[-1] == 0
    assert model.score(X, y) == 1.0


def test_adaline_separates_blobs():
    X, y = make_blobs(
        n_samples=80,
        centers=[[-2, -2], [2, 2]],
        cluster_std=0.4,
        random_state=42,
    )

    model = AdalineGD(learning_rate=0.05, n_iterations=100)
    model.fit(X, y)

    assert model.loss_history_[0] > model.loss_history_[-1]
    assert model.score(X, y) > 0.95


def test_custom_decision_tree_classifier_separates_blobs():
    X, y = make_blobs(
        n_samples=100,
        centers=[[-2, -2], [2, 2]],
        cluster_std=0.5,
        random_state=42,
    )

    model = DecisionTreeClassifierCustom(max_depth=3)
    model.fit(X, y)

    assert model.score(X, y) == 1.0
    assert model.predict_proba(X[:3]).shape == (3, 2)
    np.testing.assert_allclose(model.feature_importances_.sum(), 1.0)


def test_custom_decision_tree_regressor_fits_step_function():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.where(X.ravel() < 10, 1.0, 5.0)

    model = DecisionTreeRegressorCustom(max_depth=2)
    model.fit(X, y)

    np.testing.assert_allclose(model.predict([[2.0], [15.0]]), [1.0, 5.0])
    np.testing.assert_allclose(model.feature_importances_.sum(), 1.0)


def test_custom_svm_separates_blobs():
    X, y_zero_based = make_blobs(
        n_samples=80,
        centers=[[-2, -2], [2, 2]],
        cluster_std=0.4,
        random_state=42,
    )
    y = np.where(y_zero_based == 1, 1, -1)

    model = SVMClassifierCustom(C=1.0, learning_rate=0.01, n_iterations=80)
    model.fit(X, y)

    assert model.score(X, y) > 0.95
    assert len(model.support_vectors_) > 0


def test_gaussian_naive_bayes_separates_blobs():
    X, y = make_blobs(
        n_samples=100,
        centers=[[-2, -2], [2, 2]],
        cluster_std=0.5,
        random_state=42,
    )

    model = GaussianNaiveBayes().fit(X, y)
    proba = model.predict_proba(X[:4])

    assert model.score(X, y) > 0.95
    assert proba.shape == (4, 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(4))


def test_multinomial_naive_bayes_predicts_count_data():
    X = np.array([
        [4, 1, 0],
        [3, 1, 0],
        [0, 1, 4],
        [0, 1, 5],
    ])
    y = np.array([0, 0, 1, 1])

    model = MultinomialNaiveBayes(alpha=1.0).fit(X, y)

    assert model.score(X, y) == 1.0
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), np.ones(4))


def test_multinomial_naive_bayes_rejects_negative_values():
    X = np.array([[1, 0], [-1, 2]])
    y = np.array([0, 1])

    model = MultinomialNaiveBayes()

    try:
        model.fit(X, y)
    except ValueError:
        return

    raise AssertionError('negative count values should raise ValueError')


def test_bernoulli_naive_bayes_predicts_binary_data():
    X = np.array([
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
    ])
    y = np.array([0, 0, 1, 1])

    model = BernoulliNaiveBayes(alpha=1.0, binarize=None).fit(X, y)

    assert model.score(X, y) == 1.0
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), np.ones(4))


def test_bayesian_linear_regression_returns_mean_and_variance():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = 2 * X.ravel() + 1

    model = BayesianLinearRegression(alpha=1.0, beta=10.0)
    model.fit(X, y)

    mean, variance = model.predict([[5.0], [10.0]])

    np.testing.assert_allclose(mean, [11.0, 21.0], atol=0.5)
    assert np.all(variance > 0)


def test_ridge_regression_custom_learns_simple_line():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = 2.5 * X.ravel() - 3

    model = RidgeRegressionCustom(alpha=0.01)
    model.fit(X, y)

    np.testing.assert_allclose(model.predict([[10.0]]), [22.0], atol=0.1)
    assert model.score(X, y) > 0.99


def test_lasso_regression_custom_selects_sparse_signal():
    rng = np.random.RandomState(42)
    X = rng.normal(size=(120, 3))
    y = 4.0 * X[:, 0] + rng.normal(scale=0.05, size=120)

    model = LassoRegressionCustom(alpha=0.05, max_iter=3000, tol=1e-7)
    model.fit(X, y)

    assert model.converged_
    assert abs(model.coef_[0]) > 3.5
    assert np.count_nonzero(np.abs(model.coef_[1:]) > 0.1) == 0
    assert model.score(X, y) > 0.98


def test_elastic_net_custom_handles_correlated_features():
    rng = np.random.RandomState(42)
    x0 = rng.normal(size=120)
    X = np.column_stack([
        x0,
        x0 + rng.normal(scale=0.05, size=120),
        rng.normal(size=120),
    ])
    y = 2.0 * x0 + rng.normal(scale=0.05, size=120)

    model = ElasticNetCustom(alpha=0.01, l1_ratio=0.5, max_iter=3000, tol=1e-7)
    model.fit(X, y)

    assert model.converged_
    assert model.score(X, y) > 0.95
    assert np.count_nonzero(np.abs(model.coef_) > 1e-6) >= 1


def test_regularized_logistic_regression_predicts_multiclass_blobs():
    X, y = make_blobs(
        n_samples=120,
        centers=[[-3, -3], [0, 3], [3, -3]],
        cluster_std=0.4,
        random_state=42,
    )

    model = RegularizedLogisticRegression(
        penalty='l2',
        C=10.0,
        learning_rate=0.05,
        n_iterations=300,
    )
    model.fit(X, y)

    assert model.loss_history_[0] > model.loss_history_[-1]
    assert model.score(X, y) > 0.95
    np.testing.assert_allclose(model.predict_proba(X[:5]).sum(axis=1), np.ones(5))


def test_pca_custom_matches_sklearn_variance_and_reconstructs():
    rng = np.random.RandomState(42)
    X = rng.normal(size=(100, 4))
    X[:, 2] = 0.5 * X[:, 0] + rng.normal(scale=0.05, size=100)

    custom = PCACustom(n_components=2).fit(X)
    sklearn_pca = PCA(n_components=2).fit(X)

    np.testing.assert_allclose(
        custom.explained_variance_ratio_,
        sklearn_pca.explained_variance_ratio_,
        atol=1e-6,
    )

    alignment = np.abs(np.sum(custom.components_ * sklearn_pca.components_, axis=1))
    np.testing.assert_allclose(alignment, np.ones(2), atol=1e-6)

    Z = custom.transform(X)
    reconstructed = custom.inverse_transform(Z)

    assert Z.shape == (100, 2)
    assert reconstructed.shape == X.shape
    assert np.mean((X - reconstructed) ** 2) < np.var(X)


def test_lda_custom_projects_multiclass_blobs():
    X, y = make_blobs(
        n_samples=150,
        centers=[[-3, -3, 0], [0, 3, 2], [3, -3, -2]],
        cluster_std=0.5,
        random_state=42,
    )

    model = LDACustom(n_components=2)
    Z = model.fit_transform(X, y)
    class_means = np.vstack([Z[y == cls].mean(axis=0) for cls in model.classes_])

    assert Z.shape == (150, 2)
    assert model.scalings_.shape == (3, 2)
    assert model.explained_variance_ratio_.sum() > 0.99
    assert np.linalg.norm(class_means[0] - class_means[1]) > 1.0


def test_kernel_pca_custom_rbf_separates_circles():
    X, y = make_circles(n_samples=160, factor=0.35, noise=0.04, random_state=42)

    model = KernelPCACustom(n_components=3, kernel='rbf', gamma=1.0)
    Z = model.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(LogisticRegression(max_iter=1000), Z, y, cv=cv).mean()

    assert Z.shape == (160, 3)
    assert np.all(model.lambdas_ > 0)
    assert score > 0.95


def test_kernel_pca_rejects_unknown_kernel():
    model = KernelPCACustom(kernel='unknown')

    with pytest.raises(ValueError):
        model.fit(np.ones((5, 2)))


def test_kmeans_custom_matches_sklearn_on_blobs():
    X, y = make_blobs(
        n_samples=120,
        centers=[[-3, -3], [0, 3], [3, -3]],
        cluster_std=0.45,
        random_state=42,
    )

    custom = KMeansCustom(n_clusters=3, n_init=10, random_state=42).fit(X)
    reference = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)

    assert adjusted_rand_score(y, custom.labels_) > 0.95
    assert custom.transform(X[:5]).shape == (5, 3)
    assert abs(custom.inertia_ - reference.inertia_) / reference.inertia_ < 0.05


def test_dbscan_custom_matches_sklearn_on_moons():
    X, _ = make_moons(n_samples=160, noise=0.04, random_state=42)

    custom_labels = DBSCANCustom(eps=0.25, min_samples=5).fit_predict(X)
    reference_labels = DBSCAN(eps=0.25, min_samples=5).fit_predict(X)

    assert adjusted_rand_score(reference_labels, custom_labels) > 0.95


def test_gaussian_mixture_custom_clusters_blobs():
    X, y = make_blobs(
        n_samples=150,
        centers=[[-3, -3], [0, 3], [3, -3]],
        cluster_std=0.55,
        random_state=42,
    )

    custom = GaussianMixtureModelCustom(n_components=3, max_iter=100, random_state=42)
    custom.fit(X)
    reference = GaussianMixture(n_components=3, random_state=42).fit(X)

    proba = custom.predict_proba(X[:6])

    assert adjusted_rand_score(y, custom.predict(X)) > 0.95
    assert proba.shape == (6, 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(6))
    assert abs(custom.score(X) - reference.score(X)) < 0.5


def test_agglomerative_custom_matches_sklearn_linkages():
    X, _ = make_blobs(
        n_samples=45,
        centers=[[-3, -3], [0, 3], [3, -3]],
        cluster_std=0.45,
        random_state=42,
    )

    for linkage in ['single', 'complete', 'average', 'ward']:
        custom = AgglomerativeCustom(n_clusters=3, linkage=linkage)
        custom_labels = custom.fit_predict(X)

        reference = AgglomerativeClustering(n_clusters=3, linkage=linkage)
        reference_labels = reference.fit_predict(X)

        assert custom.linkage_matrix_.shape == (X.shape[0] - 1, 4)
        assert adjusted_rand_score(reference_labels, custom_labels) > 0.90



def test_activation_functions_shapes_and_softmax_rows():
    z = np.array([[-1.0, 0.0, 1.0]])

    assert ActivationFunctions.relu(z).shape == z.shape
    assert ActivationFunctions.sigmoid(z).shape == z.shape
    assert ActivationFunctions.tanh(z).shape == z.shape
    assert ActivationFunctions.leaky_relu(z).shape == z.shape

    proba = ActivationFunctions.softmax(z)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(1))


def test_mlp_custom_learns_xor():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([0, 1, 1, 0])

    model = MLPCustom(
        hidden_layer_sizes=(4,),
        activation='tanh',
        learning_rate=0.05,
        n_iterations=1200,
        batch_size=4,
        optimizer='adam',
        weight_init='xavier',
        random_state=42,
    )
    model.fit(X, y)

    assert model.score(X, y) == 1.0
    assert model.loss_history_[0] > model.loss_history_[-1]


def test_mlp_custom_predict_proba_shape_and_sum():
    X, y = make_blobs(
        n_samples=90,
        centers=[[-2, -2], [2, 2], [2, -2]],
        cluster_std=0.5,
        random_state=42,
    )

    model = MLPCustom(
        hidden_layer_sizes=(12,),
        activation='relu',
        learning_rate=0.01,
        n_iterations=120,
        batch_size=16,
        optimizer='adam',
        random_state=42,
    ).fit(X, y)
    proba = model.predict_proba(X[:5])

    assert proba.shape == (5, 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(5), atol=1e-7)
    assert model.score(X, y) > 0.9


def test_mlp_custom_regression_score_positive():
    rng = np.random.RandomState(42)
    X = rng.normal(size=(120, 2))
    y = 2.0 * X[:, 0] - 0.5 * X[:, 1] + 1.0

    model = MLPCustom(
        hidden_layer_sizes=(10,),
        activation='tanh',
        output_activation='linear',
        task='regression',
        learning_rate=0.01,
        n_iterations=250,
        batch_size=16,
        optimizer='adam',
        weight_init='xavier',
        random_state=42,
    ).fit(X, y)

    assert model.score(X, y) > 0.9
    assert model.loss_history_[0] > model.loss_history_[-1]


def test_mlp_custom_early_stopping_records_validation_loss():
    X, y = make_blobs(
        n_samples=120,
        centers=[[-2, -2], [2, 2]],
        cluster_std=0.6,
        random_state=42,
    )

    model = MLPCustom(
        hidden_layer_sizes=(8,),
        n_iterations=80,
        batch_size=16,
        optimizer='adam',
        validation_fraction=0.2,
        patience=5,
        random_state=42,
    ).fit(X, y)

    assert len(model.val_loss_history_) > 0
    assert len(model.loss_history_) == model.n_epochs_



def test_models_package_exports_public_model_api():
    import src.models as models

    expected = [
        'AttentionClassifier', 'LogisticRegressionOvR', 'LogisticRegressionSoftmax',
        'LinearRegressionGD', 'LinearRegressionNormal', 'Perceptron', 'AdalineGD',
        'Node', 'DecisionTreeClassifierCustom', 'DecisionTreeRegressorCustom',
        'SVMClassifierCustom',
        'GaussianNaiveBayes', 'MultinomialNaiveBayes', 'BernoulliNaiveBayes',
        'BayesianLinearRegression',
        'RidgeRegressionCustom', 'LassoRegressionCustom', 'ElasticNetCustom',
        'RegularizedLogisticRegression',
        'KMeansCustom', 'DBSCANCustom', 'GaussianMixtureModelCustom',
        'AgglomerativeCustom',
        'BaggingClassifierCustom', 'BaggingRegressorCustom',
        'AdaBoostClassifierCustom',
        'PCACustom', 'LDACustom', 'KernelPCACustom',
        'ActivationFunctions', 'MLPCustom',
    ]
    missing = [name for name in expected if not hasattr(models, name)]

    assert missing == []
