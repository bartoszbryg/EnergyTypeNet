"""Backward-compatible public API; new code may import from family submodules."""

from ._base import Node
from .linear import (
    AdalineGD,
    AttentionClassifier,
    LinearRegressionGD,
    LinearRegressionNormal,
    LogisticRegressionOvR,
    LogisticRegressionSoftmax,
    Perceptron,
)
from .regularized import (
    ElasticNetCustom,
    LassoRegressionCustom,
    RegularizedLogisticRegression,
    RidgeRegressionCustom,
)
from .trees import DecisionTreeClassifierCustom, DecisionTreeRegressorCustom
from .svm import SVMClassifierCustom
from .probabilistic import GaussianNaiveBayes, MultinomialNaiveBayes, BernoulliNaiveBayes, BayesianLinearRegression
from .dimensionality import PCACustom, LDACustom, KernelPCACustom
from .clustering import KMeansCustom, DBSCANCustom, GaussianMixtureModelCustom, AgglomerativeCustom
from .neural import ActivationFunctions, MLPCustom
from .ensemble import BaggingClassifierCustom, BaggingRegressorCustom, AdaBoostClassifierCustom

__all__ = [
    'Node',
    'LinearRegressionGD',
    'LinearRegressionNormal',
    'Perceptron',
    'AdalineGD',
    'AttentionClassifier',
    'LogisticRegressionOvR',
    'LogisticRegressionSoftmax',
    'RidgeRegressionCustom',
    'LassoRegressionCustom',
    'ElasticNetCustom',
    'RegularizedLogisticRegression',
    'DecisionTreeClassifierCustom',
    'DecisionTreeRegressorCustom',
    'SVMClassifierCustom',
    'GaussianNaiveBayes',
    'MultinomialNaiveBayes',
    'BernoulliNaiveBayes',
    'BayesianLinearRegression',
    'PCACustom',
    'LDACustom',
    'KernelPCACustom',
    'KMeansCustom',
    'DBSCANCustom',
    'GaussianMixtureModelCustom',
    'AgglomerativeCustom',
    'ActivationFunctions',
    'MLPCustom',
    'BaggingClassifierCustom',
    'BaggingRegressorCustom',
    'AdaBoostClassifierCustom',
]
