import importlib


MODULE_EXPORTS = {
    '_base': ['Node'],
    'linear': ['LinearRegressionGD', 'LinearRegressionNormal', 'Perceptron', 'AdalineGD', 'AttentionClassifier', 'LogisticRegressionOvR', 'LogisticRegressionSoftmax'],
    'regularized': ['RidgeRegressionCustom', 'LassoRegressionCustom', 'ElasticNetCustom', 'RegularizedLogisticRegression'],
    'trees': ['DecisionTreeClassifierCustom', 'DecisionTreeRegressorCustom'],
    'svm': ['SVMClassifierCustom'],
    'probabilistic': ['GaussianNaiveBayes', 'MultinomialNaiveBayes', 'BernoulliNaiveBayes', 'BayesianLinearRegression'],
    'clustering': ['KMeansCustom', 'DBSCANCustom', 'GaussianMixtureModelCustom', 'AgglomerativeCustom'],
    'ensemble': ['BaggingClassifierCustom', 'BaggingRegressorCustom', 'AdaBoostClassifierCustom'],
    'dimensionality': ['PCACustom', 'LDACustom', 'KernelPCACustom'],
    'neural': ['ActivationFunctions', 'MLPCustom'],
}


def test_models_package_reexports_expected_classes():
    import src.models as models

    expected = [name for names in MODULE_EXPORTS.values() for name in names]
    assert set(expected).issubset(set(models.__all__))

    for name in expected:
        assert getattr(models, name) is not None


def test_model_family_submodules_export_expected_classes():
    for module_name, class_names in MODULE_EXPORTS.items():
        module = importlib.import_module(f'src.models.{module_name}')
        for class_name in class_names:
            assert hasattr(module, class_name)
