"""Verify backward-compatible and direct model-package imports."""

import importlib
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OLD_STYLE_IMPORTS = [
    ("src.models", name)
    for name in [
        "Node",
        "AttentionClassifier",
        "LogisticRegressionOvR",
        "LogisticRegressionSoftmax",
        "LinearRegressionGD",
        "LinearRegressionNormal",
        "Perceptron",
        "AdalineGD",
        "DecisionTreeClassifierCustom",
        "DecisionTreeRegressorCustom",
        "SVMClassifierCustom",
        "GaussianNaiveBayes",
        "MultinomialNaiveBayes",
        "BernoulliNaiveBayes",
        "BayesianLinearRegression",
        "RidgeRegressionCustom",
        "LassoRegressionCustom",
        "ElasticNetCustom",
        "RegularizedLogisticRegression",
        "KMeansCustom",
        "DBSCANCustom",
        "GaussianMixtureModelCustom",
        "AgglomerativeCustom",
        "BaggingClassifierCustom",
        "BaggingRegressorCustom",
        "AdaBoostClassifierCustom",
        "PCACustom",
        "LDACustom",
        "KernelPCACustom",
        "ActivationFunctions",
        "MLPCustom",
    ]
]

NEW_STYLE_IMPORTS = [
    ("src.models._base", "Node"),
    ("src.models.linear", "LinearRegressionGD"),
    ("src.models.regularized", "RidgeRegressionCustom"),
    ("src.models.trees", "DecisionTreeClassifierCustom"),
    ("src.models.svm", "SVMClassifierCustom"),
    ("src.models.probabilistic", "GaussianNaiveBayes"),
    ("src.models.dimensionality", "PCACustom"),
    ("src.models.clustering", "KMeansCustom"),
    ("src.models.neural", "MLPCustom"),
    ("src.models.ensemble", "AdaBoostClassifierCustom"),
]


def verify_imports(imports):
    failures = []
    for module_name, item_name in imports:
        label = f"{module_name}.{item_name}"
        try:
            module = importlib.import_module(module_name)
            getattr(module, item_name)
        except Exception as exc:  # report every failure instead of stopping early
            print(f"FAIL: {label}: {exc}")
            failures.append(label)
        else:
            print(f"PASS: {label}")
    return failures


def verify_adaboost_dependency():
    label = "AdaBoostClassifierCustom uses DecisionTreeClassifierCustom"
    try:
        ensemble = importlib.import_module("src.models.ensemble")
        trees = importlib.import_module("src.models.trees")
        model = ensemble.AdaBoostClassifierCustom(n_estimators=1, random_state=0)
        X = np.array([[0.0], [0.1], [0.9], [1.0]])
        y = np.array([0, 0, 1, 1])
        model.fit(X, y)
        if not model.estimators_ or not isinstance(
            model.estimators_[0], trees.DecisionTreeClassifierCustom
        ):
            raise TypeError("default fitted estimator is not the custom decision tree")
    except Exception as exc:
        print(f"FAIL: {label}: {exc}")
        return [label]
    print(f"PASS: {label}")
    return []


def main():
    failures = verify_imports(OLD_STYLE_IMPORTS)
    failures.extend(verify_imports(NEW_STYLE_IMPORTS))
    failures.extend(verify_adaboost_dependency())
    print(f"\n{len(failures)} failure(s)" if failures else "\nAll model imports passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
