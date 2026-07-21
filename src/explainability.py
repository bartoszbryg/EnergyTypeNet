"""Model-agnostic SHAP utilities for EnergyTypeNet.

SHAP is deliberately an optional dependency.  Importing this module never
imports :mod:`shap`; only :func:`build_shap_explainer` requires it.  This keeps
the API, dashboard, and core model package usable in minimal installations.

The public functions operate on plain NumPy-compatible inputs and return
Python/pandas data structures.  No Streamlit dependency belongs in this
module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class SHAPExplainerType(str, Enum):
    """Supported SHAP explainer families."""

    TREE = "tree"
    LINEAR = "linear"
    KERNEL = "kernel"
    XGBOOST = "xgboost"


# Convenient module-level aliases preserve a constants-style API as well as
# the enum API.
TREE = SHAPExplainerType.TREE
LINEAR = SHAPExplainerType.LINEAR
KERNEL = SHAPExplainerType.KERNEL
XGBOOST = SHAPExplainerType.XGBOOST


_SKLEARN_TREE_MODELS = frozenset(
    {
        "RandomForestClassifier",
        "RandomForestRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "HistGradientBoostingClassifier",
        "HistGradientBoostingRegressor",
        "DecisionTreeClassifier",
        "DecisionTreeRegressor",
    }
)

_CUSTOM_TREE_MODELS = frozenset(
    {
        "DecisionTreeClassifierCustom",
        "DecisionTreeRegressorCustom",
        "BaggingClassifierCustom",
        "BaggingRegressorCustom",
        "AdaBoostClassifierCustom",
    }
)

_SKLEARN_LINEAR_MODELS = frozenset(
    {
        "LogisticRegression",
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "SGDClassifier",
        "SGDRegressor",
        "Perceptron",
    }
)

_CUSTOM_LINEAR_MODELS = frozenset(
    {
        "LogisticRegressionOvR",
        "LogisticRegressionSoftmax",
        "LinearRegressionGD",
        "LinearRegressionNormal",
        "RidgeRegressionCustom",
        "LassoRegressionCustom",
        "ElasticNetCustom",
        "RegularizedLogisticRegression",
    }
)


def detect_explainer_type(estimator: Any) -> SHAPExplainerType:
    """Select a SHAP explainer family for a fitted estimator.

    Routing uses class and module names rather than importing every supported
    estimator type.  Besides avoiding heavy imports, this lets the module work
    with compatible estimators loaded from optional packages.
    """

    class_name = type(estimator).__name__
    module_name = type(estimator).__module__.lower()

    if "XGB" in class_name or "xgboost" in module_name:
        return XGBOOST
    if class_name in _CUSTOM_TREE_MODELS:
        return KERNEL
    if class_name in _SKLEARN_TREE_MODELS:
        return TREE
    if class_name in _CUSTOM_LINEAR_MODELS:
        return KERNEL
    if class_name in _SKLEARN_LINEAR_MODELS:
        return LINEAR
    return KERNEL


def _identity_transform(values: Any) -> Any:
    return values


def extract_pipeline_parts(model: Any) -> tuple[Any, Callable[[Any], Any]]:
    """Return the final estimator and its original-to-model-space transform."""

    if isinstance(model, Pipeline):
        if not model.steps:
            raise ValueError("Cannot explain an empty sklearn Pipeline.")
        estimator = model.steps[-1][1]
        if len(model.steps) == 1:
            return estimator, _identity_transform
        preprocessing = model[:-1]

        def transform(values: Any) -> Any:
            return preprocessing.transform(values)

        return estimator, transform

    return model, _identity_transform


def _as_dense_2d(values: Any) -> np.ndarray:
    """Convert array-like/sparse data to a dense two-dimensional array."""

    if hasattr(values, "toarray"):
        values = values.toarray()
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim != 2:
        raise ValueError(f"Expected a 1-D or 2-D input array, got shape {array.shape}.")
    return array


def _sample_background(background: np.ndarray, maximum: int = 100) -> np.ndarray:
    """Choose at most ``maximum`` evenly distributed, deterministic rows."""

    if len(background) <= maximum:
        return background
    indices = np.linspace(0, len(background) - 1, maximum, dtype=int)
    return background[indices]


def _import_shap() -> Any:
    try:
        import shap  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "SHAP explanations require the optional 'shap' package. "
            "Install it with: pip install shap"
        ) from exc
    return shap


def _build_linear_explainer(shap: Any, estimator: Any, background: np.ndarray) -> Any:
    """Construct LinearExplainer across old and new SHAP masker APIs."""

    maskers = getattr(shap, "maskers", None)
    independent = getattr(maskers, "Independent", None) if maskers is not None else None
    if independent is not None:
        try:
            return shap.LinearExplainer(estimator, independent(background))
        except (TypeError, AttributeError, ValueError):
            # Some intermediate SHAP releases expose maskers but still expect
            # the background matrix in LinearExplainer.
            pass

    try:
        return shap.LinearExplainer(estimator, background)
    except TypeError:
        # Compatibility with older SHAP releases that used the
        # ``feature_dependence`` keyword.
        return shap.LinearExplainer(
            estimator,
            background,
            feature_dependence="independent",
        )


def build_shap_explainer(
    model: Any,
    background_data: Any,
    feature_names: Sequence[str],
) -> tuple[Any, Callable[[Any], Any]]:
    """Build the most appropriate SHAP explainer for ``model``.

    ``background_data`` must be in the original feature space.  Pipelines are
    unwrapped and their preprocessing steps are applied before the explainer
    sees the data.
    """

    shap = _import_shap()
    estimator, transform = extract_pipeline_parts(model)
    original_background = _as_dense_2d(background_data)
    transformed_background = _as_dense_2d(transform(original_background))
    explainer_type = detect_explainer_type(estimator)

    if explainer_type in {TREE, XGBOOST}:
        data = transformed_background if len(transformed_background) <= 100 else None
        explainer = shap.TreeExplainer(estimator, data=data)
    elif explainer_type is LINEAR:
        explainer = _build_linear_explainer(shap, estimator, transformed_background)
    else:
        predict_function = (
            estimator.predict_proba
            if callable(getattr(estimator, "predict_proba", None))
            else estimator.predict
        )
        explainer = shap.KernelExplainer(
            predict_function,
            _sample_background(transformed_background),
        )

    # These lightweight metadata fields make later normalization and local
    # probability lookup independent of SHAP's version-specific internals.
    metadata = {
        "_energytypenet_estimator": estimator,
        "_energytypenet_explainer_type": explainer_type.value,
        "_energytypenet_feature_names": list(feature_names),
        "_energytypenet_warning": None,
    }
    for name, value in metadata.items():
        try:
            setattr(explainer, name, value)
        except (AttributeError, TypeError):
            # Explanation computation still works for an explainer that uses
            # slots or otherwise forbids additional attributes.
            pass

    return explainer, transform


def _raw_values(shap_result: Any) -> Any:
    """Unwrap modern ``shap.Explanation`` objects without importing SHAP."""

    return getattr(shap_result, "values", shap_result)


def _normalise_shap_arrays(
    shap_result: Any,
    n_samples: int,
    n_features: int,
) -> list[np.ndarray]:
    """Normalize SHAP's version/model-specific outputs to class-first arrays."""

    raw = _raw_values(shap_result)
    if isinstance(raw, (list, tuple)):
        arrays = [_as_shap_matrix(item, n_samples, n_features) for item in raw]
        return arrays

    array = np.asarray(raw)
    if array.ndim <= 2:
        return [_as_shap_matrix(array, n_samples, n_features)]
    if array.ndim != 3:
        raise ValueError(f"Unsupported SHAP value shape: {array.shape}.")

    # Current SHAP commonly returns (samples, features, outputs).  Older
    # versions and explainers may use (outputs, samples, features) or
    # (samples, outputs, features).
    if array.shape[0] == n_samples and array.shape[1] == n_features:
        return [array[:, :, index] for index in range(array.shape[2])]
    if array.shape[1] == n_samples and array.shape[2] == n_features:
        return [array[index, :, :] for index in range(array.shape[0])]
    if array.shape[0] == n_samples and array.shape[2] == n_features:
        return [array[:, index, :] for index in range(array.shape[1])]
    raise ValueError(
        "Could not identify sample and feature axes in SHAP values with "
        f"shape {array.shape}; expected {n_samples} samples and {n_features} features."
    )


def _as_shap_matrix(values: Any, n_samples: int, n_features: int) -> np.ndarray:
    array = np.asarray(_raw_values(values))
    if array.ndim == 1:
        if n_samples != 1 or array.size != n_features:
            raise ValueError(
                f"Unexpected one-dimensional SHAP value shape {array.shape}."
            )
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"Expected a two-dimensional SHAP array, got {array.shape}.")
    if array.shape == (n_features, n_samples) and array.shape != (n_samples, n_features):
        array = array.T
    if array.shape != (n_samples, n_features):
        raise ValueError(
            f"SHAP values have shape {array.shape}; expected "
            f"({n_samples}, {n_features})."
        )
    return np.asarray(array, dtype=float)


def _class_count(explainer: Any, arrays: Sequence[np.ndarray]) -> int | None:
    estimator = getattr(explainer, "_energytypenet_estimator", None)
    classes = getattr(estimator, "classes_", None)
    if classes is not None:
        return len(classes)
    if len(arrays) > 1:
        return len(arrays)
    expected = np.asarray(getattr(explainer, "expected_value", []))
    return int(expected.size) if expected.ndim == 1 and expected.size > 1 else None


def _store_warning(explainer: Any, message: str | None) -> None:
    try:
        setattr(explainer, "_energytypenet_warning", message)
        setattr(explainer, "warning_message", message)
    except (AttributeError, TypeError):
        pass


def get_last_shap_warning(explainer: Any) -> str | None:
    """Return the last non-fatal SHAP warning stored on an explainer."""

    return getattr(
        explainer,
        "_energytypenet_warning",
        getattr(explainer, "warning_message", None),
    )


def _invoke_explainer(explainer: Any, transformed: np.ndarray) -> Any:
    shap_values = getattr(explainer, "shap_values", None)
    if callable(shap_values):
        return shap_values(transformed)
    if callable(explainer):
        return explainer(transformed)
    raise TypeError("The supplied SHAP explainer is not callable.")


def compute_shap_values(
    explainer: Any,
    transform: Callable[[Any], Any],
    input_data: Any,
    local: bool = False,
) -> list[np.ndarray] | None:
    """Compute and normalize local or global SHAP values.

    The ``local`` flag documents the caller's intent; both modes retain the
    sample axis so downstream consumers always receive arrays shaped
    ``(n_samples, n_features)``.  Failures are non-fatal: ``None`` is returned,
    a warning is emitted, and its text is stored on the explainer.
    """

    del local  # Shape normalization is intentionally identical for both modes.
    try:
        transformed = _as_dense_2d(transform(_as_dense_2d(input_data)))
        result = _invoke_explainer(explainer, transformed)
        arrays = _normalise_shap_arrays(
            result,
            n_samples=transformed.shape[0],
            n_features=transformed.shape[1],
        )
        if _class_count(explainer, arrays) == 2 and len(arrays) >= 2:
            arrays = [arrays[1]]
        _store_warning(explainer, None)
        return arrays
    except Exception as exc:  # SHAP raises many model/version-specific types.
        message = f"SHAP values could not be computed: {exc}"
        _store_warning(explainer, message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return None


def _output_names(class_names: Sequence[Any] | None, output_count: int) -> list[str]:
    supplied_names = [] if class_names is None else class_names
    names = [str(name) for name in supplied_names]
    if output_count == 1 and len(names) == 2:
        return [names[1]]  # Binary compute_shap_values keeps the positive class.
    if len(names) == output_count:
        return names
    if output_count == 1:
        return [names[0] if names else "output"]
    return [names[index] if index < len(names) else f"class_{index}" for index in range(output_count)]


def _safe_column_token(value: str) -> str:
    token = "".join(character if character.isalnum() else "_" for character in value)
    return token.strip("_") or "output"


def compute_global_importance(
    shap_values: Sequence[np.ndarray],
    feature_names: Sequence[str],
    class_names: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Aggregate per-sample SHAP values into global feature importance."""

    if not shap_values:
        raise ValueError("At least one SHAP value array is required.")
    feature_names = list(feature_names)
    matrices = []
    for values in shap_values:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
            raise ValueError(
                "Each SHAP array must have shape (n_samples, n_features) and "
                "match the supplied feature names."
            )
        matrices.append(matrix)

    per_output = np.vstack([np.mean(np.abs(matrix), axis=0) for matrix in matrices])
    frame_data: dict[str, Any] = {
        "feature": feature_names,
        "mean_abs_shap": np.mean(per_output, axis=0),
    }
    for name, values in zip(_output_names(class_names, len(matrices)), per_output):
        frame_data[f"mean_abs_shap_{_safe_column_token(name)}"] = values

    return (
        pd.DataFrame(frame_data)
        .sort_values("mean_abs_shap", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def _predicted_class_index(
    predicted_class: Any,
    class_names: Sequence[Any] | None,
    estimator: Any,
    output_count: int,
) -> int:
    candidate_groups = [
        list(getattr(estimator, "classes_", [])),
        list([] if class_names is None else class_names),
    ]
    for candidates in candidate_groups:
        for index, candidate in enumerate(candidates):
            if predicted_class == candidate or str(predicted_class) == str(candidate):
                return min(index, max(output_count - 1, 0))
    if isinstance(predicted_class, (int, np.integer)):
        return min(max(int(predicted_class), 0), max(output_count - 1, 0))
    return 0


def _select_expected_value(explainer: Any, class_index: int) -> float | None:
    expected = getattr(explainer, "expected_value", None)
    if expected is None:
        return None
    array = np.asarray(expected)
    if array.size == 0:
        return None
    flat = array.reshape(-1)
    value = flat[min(class_index, len(flat) - 1)]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _predicted_probability(
    estimator: Any,
    transformed: np.ndarray,
    predicted_class: Any,
    class_names: Sequence[Any] | None,
) -> float | None:
    predict_proba = getattr(estimator, "predict_proba", None)
    if not callable(predict_proba):
        return None
    try:
        probabilities = np.asarray(predict_proba(transformed))
        if probabilities.ndim != 2 or probabilities.shape[0] == 0:
            return None
        class_index = _predicted_class_index(
            predicted_class,
            class_names,
            estimator,
            probabilities.shape[1],
        )
        return float(probabilities[0, class_index])
    except Exception:
        return None


def compute_local_explanation(
    explainer: Any,
    transform: Callable[[Any], Any],
    sample: Any,
    feature_names: Sequence[str],
    class_names: Sequence[Any] | None,
    predicted_class: Any,
) -> dict[str, Any] | None:
    """Build a serializable SHAP explanation for one prediction."""

    try:
        original = _as_dense_2d(sample)
        if original.shape[0] != 1:
            raise ValueError("A local explanation requires exactly one sample.")
        if original.shape[1] != len(feature_names):
            raise ValueError("The sample width must match the supplied feature names.")

        transformed = _as_dense_2d(transform(original))
        raw = _invoke_explainer(explainer, transformed)
        arrays = _normalise_shap_arrays(raw, 1, transformed.shape[1])
        estimator = getattr(explainer, "_energytypenet_estimator", None)
        class_index = _predicted_class_index(
            predicted_class,
            class_names,
            estimator,
            len(arrays),
        )
        selected = arrays[class_index if len(arrays) > 1 else 0][0]
        if selected.size != len(feature_names):
            raise ValueError(
                "The transformed feature count differs from the original feature "
                "names, so a reliable local feature mapping cannot be produced."
            )

        _store_warning(explainer, None)
        return {
            "feature_names": list(feature_names),
            "feature_values": original[0].tolist(),
            "shap_values": selected.astype(float).tolist(),
            "base_value": _select_expected_value(explainer, class_index),
            "predicted_class": (
                predicted_class.item()
                if isinstance(predicted_class, np.generic)
                else predicted_class
            ),
            "predicted_probability": _predicted_probability(
                estimator,
                transformed,
                predicted_class,
                class_names,
            ),
        }
    except Exception as exc:
        message = f"A local SHAP explanation could not be computed: {exc}"
        _store_warning(explainer, message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return None


def _import_lime_tabular() -> Any:
    """Import LIME only when an explanation is explicitly requested."""

    try:
        from lime import lime_tabular  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "LIME explanations require the optional 'lime' package. "
            "Install it with: pip install lime"
        ) from exc
    return lime_tabular


def build_lime_explainer(
    training_data: Any,
    feature_names: Sequence[str],
    class_names: Sequence[Any] | None,
    task_type: str,
    categorical_features: Sequence[int] | None = None,
) -> Any:
    """Build a tabular LIME explainer in the model's original feature space."""

    if task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be 'classification' or 'regression'.")

    lime_tabular = _import_lime_tabular()
    names = list(feature_names)
    input_is_dataframe = isinstance(training_data, pd.DataFrame)
    frame = (
        training_data.loc[:, names].copy()
        if isinstance(training_data, pd.DataFrame)
        else pd.DataFrame(_as_dense_2d(training_data), columns=names)
    )
    categorical_indices = (
        [int(index) for index in categorical_features]
        if categorical_features is not None
        else [
            index
            for index, column in enumerate(frame.columns)
            if not pd.api.types.is_numeric_dtype(frame[column])
        ]
    )
    category_values: dict[int, list[Any]] = {}
    data = frame.to_numpy(copy=True)
    categorical_names: dict[int, list[str]] = {}
    for index in categorical_indices:
        values = list(pd.unique(frame.iloc[:, index].dropna()))
        if not values:
            values = [""]
        category_values[index] = values
        categorical_names[index] = [str(value) for value in values]
        mapping = {value: code for code, value in enumerate(values)}
        data[:, index] = frame.iloc[:, index].map(mapping).fillna(0).to_numpy()
    data = np.asarray(data, dtype=float)
    if data.shape[1] != len(names):
        raise ValueError("Training-data width must match the supplied feature names.")

    kwargs: dict[str, Any] = {
        "training_data": data,
        "feature_names": names,
        "mode": task_type,
    }
    if task_type == "classification":
        kwargs["class_names"] = [
            str(name) for name in ([] if class_names is None else class_names)
        ]
    if categorical_indices:
        kwargs["categorical_features"] = categorical_indices
        kwargs["categorical_names"] = categorical_names

    explainer = lime_tabular.LimeTabularExplainer(**kwargs)
    explainer._energytypenet_columns = names
    explainer._energytypenet_category_values = category_values
    explainer._energytypenet_return_dataframe = input_is_dataframe
    return explainer


def _lime_encode_sample(explainer: Any, sample: Any) -> np.ndarray:
    """Encode a model-space row into LIME's numeric perturbation space."""

    columns = list(getattr(explainer, "_energytypenet_columns", []))
    frame = (
        sample.loc[:, columns].copy()
        if columns and isinstance(sample, pd.DataFrame)
        else pd.DataFrame(_as_dense_2d(sample), columns=columns or None)
    )
    encoded = frame.to_numpy(copy=True)
    for index, values in getattr(
        explainer, "_energytypenet_category_values", {}
    ).items():
        mapping = {value: code for code, value in enumerate(values)}
        encoded[:, index] = frame.iloc[:, index].map(mapping).fillna(0).to_numpy()
    return np.asarray(encoded, dtype=float)


def _lime_decode_values(explainer: Any, values: Any) -> Any:
    """Decode LIME perturbations back to the schema expected by the pipeline."""

    # Object dtype is required before restoring string categories into LIME's
    # otherwise numeric perturbation matrix.
    array = _as_dense_2d(values).astype(object, copy=True)
    categories = getattr(explainer, "_energytypenet_category_values", {})
    for index, category_values in categories.items():
        codes = np.rint(array[:, index].astype(float)).astype(int)
        codes = np.clip(codes, 0, len(category_values) - 1)
        array[:, index] = [category_values[code] for code in codes]
    columns = getattr(explainer, "_energytypenet_columns", None)
    if columns and getattr(explainer, "_energytypenet_return_dataframe", False):
        return pd.DataFrame(array, columns=columns)
    return array


def _lime_label_value(container: Any, label: int | None, default: Any = None) -> Any:
    """Read LIME attributes that vary between mappings, arrays, and scalars."""

    if isinstance(container, dict):
        if label in container:
            return container[label]
        if container:
            return next(iter(container.values()))
        return default
    if container is None:
        return default
    array = np.asarray(container)
    if array.size == 0:
        return default
    flat = array.reshape(-1)
    index = 0 if label is None else min(max(int(label), 0), len(flat) - 1)
    return flat[index]


def compute_lime_explanation(
    explainer: Any,
    model: Any,
    sample: Any,
    class_index: int | None,
) -> dict[str, Any]:
    """Compute a local LIME explanation using the full fitted model pipeline."""

    # Deliberately validate the supported model shape through the same helper
    # used by SHAP.  LIME itself still receives the full pipeline prediction
    # function because it perturbs values in the original feature space.
    extract_pipeline_parts(model)
    original = _lime_encode_sample(explainer, sample)
    if original.shape[0] != 1:
        raise ValueError("A local LIME explanation requires exactly one sample.")

    is_classification = class_index is not None
    if is_classification:
        predict_proba = getattr(model, "predict_proba", None)
        if not callable(predict_proba):
            raise ValueError("Classification LIME explanations require predict_proba.")
        prediction_function = lambda values: np.asarray(
            predict_proba(_lime_decode_values(explainer, values))
        )
        explanation = explainer.explain_instance(
            original[0],
            prediction_function,
            labels=(int(class_index),),
            num_features=original.shape[1],
        )
        probabilities = np.asarray(prediction_function(original))
        model_prediction = float(probabilities[0, int(class_index)])
        explanation_label: int | None = int(class_index)
    else:
        predict = getattr(model, "predict", None)
        if not callable(predict):
            raise ValueError("Regression LIME explanations require predict.")
        prediction_function = lambda values: np.asarray(
            predict(_lime_decode_values(explainer, values))
        )
        explanation = explainer.explain_instance(
            original[0],
            prediction_function,
            num_features=original.shape[1],
        )
        model_prediction = float(np.asarray(prediction_function(original)).reshape(-1)[0])
        explanation_label = None

    feature_names = list(getattr(explainer, "feature_names", []))
    if len(feature_names) != original.shape[1]:
        feature_names = [f"feature_{index}" for index in range(original.shape[1])]

    indexed_weights = _lime_label_value(
        getattr(explanation, "local_exp", None), explanation_label, []
    )
    weights = np.zeros(original.shape[1], dtype=float)
    for feature_index, weight in indexed_weights:
        if 0 <= int(feature_index) < len(weights):
            weights[int(feature_index)] = float(weight)

    intercept = _lime_label_value(
        getattr(explanation, "intercept", None), explanation_label
    )
    local_prediction = _lime_label_value(
        getattr(explanation, "local_pred", None), explanation_label
    )
    return {
        "feature_names": feature_names,
        "lime_weights": weights.tolist(),
        "intercept": None if intercept is None else float(intercept),
        "local_r2": float(getattr(explanation, "score", np.nan)),
        "local_prediction": (
            None if local_prediction is None else float(local_prediction)
        ),
        "model_prediction": model_prediction,
        "explanation_label": explanation_label,
    }


def _plain_value(value: Any) -> Any:
    """Convert NumPy values recursively into JSON-serializable Python types."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_plain_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    return value


@dataclass
class ExplanationResult:
    """Unified, serializable explanation for a single model prediction."""

    sample_index: int | None
    feature_names: list[str]
    feature_values: list[Any]
    predicted_class: str | float
    predicted_probability: float | None
    class_names: list[str]
    shap_values: list[float] | None
    shap_base_value: float | None
    shap_explainer_type: str | None
    lime_weights: list[float] | None
    lime_intercept: float | None
    lime_local_r2: float | None
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Return only standard Python values suitable for JSON encoding."""

        return _plain_value(asdict(self))

    def summary_text(self) -> str:
        """Summarize the strongest and weakest local feature influences."""

        probability = (
            f" (probability {self.predicted_probability:.2f})"
            if self.predicted_probability is not None
            else ""
        )
        prefix = f"For this prediction of {self.predicted_class}{probability}"
        values = self.shap_values if self.shap_values is not None else self.lime_weights
        method = "SHAP value" if self.shap_values is not None else "LIME weight"
        if not values or not self.feature_names:
            return f"{prefix}, no SHAP or LIME feature contributions are available."

        count = min(len(values), len(self.feature_names))
        ranked = sorted(range(count), key=lambda index: abs(values[index]), reverse=True)
        strongest = ranked[: min(2, len(ranked))]
        weakest = ranked[-1]

        def describe(index: int, include_direction: bool = True) -> str:
            value = float(values[index])
            direction = ""
            if include_direction:
                direction = (
                    f", pushing {'toward' if value >= 0 else 'away from'} "
                    f"{self.predicted_class}"
                )
            return (
                f"{self.feature_names[index]} ({method} {value:+.2f}{direction})"
            )

        influential = " and ".join(describe(index) for index in strongest)
        return (
            f"{prefix}, the most influential features were {influential}. "
            f"The least influential was {describe(weakest, include_direction=False)}."
        )


def _prediction_details(
    model: Any,
    sample: np.ndarray,
    class_names: Sequence[Any] | None,
    task_type: str,
) -> tuple[str | float, float | None, int | None]:
    prediction = np.asarray(model.predict(sample)).reshape(-1)[0]
    if task_type == "regression":
        return float(prediction), None, None

    labels = list(
        getattr(model, "classes_", []) if class_names is None else class_names
    )
    model_classes = list(getattr(model, "classes_", labels))
    class_index = _predicted_class_index(prediction, labels, model, max(len(labels), 1))
    if prediction in model_classes:
        class_index = model_classes.index(prediction)
    predicted_label = (
        str(labels[class_index]) if class_index < len(labels) else str(prediction)
    )
    probability = None
    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba):
        probabilities = np.asarray(predict_proba(sample))
        if probabilities.ndim == 2 and class_index < probabilities.shape[1]:
            probability = float(probabilities[0, class_index])
    return predicted_label, probability, class_index


def explain_single_prediction(
    model: Any,
    sample: Any,
    feature_names: Sequence[str],
    class_names: Sequence[Any] | None,
    background_data: Any,
    task_type: str,
    sample_index: int | None = None,
    use_shap: bool = True,
    use_lime: bool = True,
    shap_explainer: tuple[Any, Callable[[Any], Any]] | None = None,
) -> ExplanationResult:
    """Orchestrate optional SHAP and LIME explanations for one prediction."""

    if task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be 'classification' or 'regression'.")
    original_input = sample
    original = _as_dense_2d(sample)
    if original.shape[0] != 1 or original.shape[1] != len(feature_names):
        raise ValueError("sample must contain one row matching the feature names.")
    background = _as_dense_2d(background_data)
    predicted, probability, class_index = _prediction_details(
        model, original_input, class_names, task_type
    )

    shap_values = None
    shap_base_value = None
    shap_type = None
    if use_shap:
        try:
            if shap_explainer is None:
                shap_explainer, transform = build_shap_explainer(
                    model, background, feature_names
                )
            else:
                shap_explainer, transform = shap_explainer
            local = compute_local_explanation(
                shap_explainer,
                transform,
                original,
                feature_names,
                class_names,
                predicted,
            )
            if local is not None:
                shap_values = [float(value) for value in local["shap_values"]]
                shap_base_value = local["base_value"]
                shap_type = getattr(
                    shap_explainer, "_energytypenet_explainer_type", None
                )
        except Exception as exc:
            warnings.warn(
                f"SHAP explanation unavailable: {exc}", RuntimeWarning, stacklevel=2
            )

    lime_weights = None
    lime_intercept = None
    lime_local_r2 = None
    if use_lime:
        try:
            lime_explainer = build_lime_explainer(
                background_data,
                feature_names,
                class_names,
                task_type,
            )
            lime = compute_lime_explanation(
                lime_explainer, model, original_input, class_index
            )
            lime_weights = [float(value) for value in lime["lime_weights"]]
            lime_intercept = lime["intercept"]
            lime_local_r2 = lime["local_r2"]
        except Exception as exc:
            warnings.warn(
                f"LIME explanation unavailable: {exc}", RuntimeWarning, stacklevel=2
            )

    return ExplanationResult(
        sample_index=None if sample_index is None else int(sample_index),
        feature_names=[str(name) for name in feature_names],
        feature_values=[_plain_value(value) for value in original[0]],
        predicted_class=predicted,
        predicted_probability=probability,
        class_names=[
            str(name) for name in ([] if class_names is None else class_names)
        ],
        shap_values=shap_values,
        shap_base_value=shap_base_value,
        shap_explainer_type=shap_type,
        lime_weights=lime_weights,
        lime_intercept=lime_intercept,
        lime_local_r2=lime_local_r2,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def explain_dataset_globally(
    model: Any,
    dataset: Any,
    feature_names: Sequence[str],
    class_names: Sequence[Any] | None,
    background_sample_size: int = 100,
    shap_explainer: tuple[Any, Callable[[Any], Any]] | None = None,
) -> pd.DataFrame:
    """Compute bounded, deterministic global SHAP feature importance."""

    if background_sample_size < 1:
        raise ValueError("background_sample_size must be at least 1.")
    data = _as_dense_2d(dataset)
    samples = _sample_background(data, maximum=500)
    background = _sample_background(data, maximum=background_sample_size)
    if len(samples) > 100:
        print(f"Computing global SHAP explanations for {len(samples)} rows...")
    if shap_explainer is None:
        explainer, transform = build_shap_explainer(model, background, feature_names)
    else:
        explainer, transform = shap_explainer
    values = compute_shap_values(explainer, transform, samples)
    if values is None:
        warning = get_last_shap_warning(explainer) or "SHAP computation failed."
        raise RuntimeError(warning)
    return compute_global_importance(values, feature_names, class_names)


def compare_shap_lime(result: ExplanationResult) -> pd.DataFrame:
    """Return an aligned, importance-sorted SHAP/LIME comparison table."""

    count = len(result.feature_names)
    if len(result.feature_values) != count:
        raise ValueError("feature_names and feature_values must have equal length.")
    if result.shap_values is not None and len(result.shap_values) != count:
        raise ValueError("SHAP values must align with feature names.")
    if result.lime_weights is not None and len(result.lime_weights) != count:
        raise ValueError("LIME weights must align with feature names.")

    shap_values = result.shap_values or [None] * count
    lime_weights = result.lime_weights or [None] * count
    agreement: list[bool | None] = []
    for shap_value, lime_weight in zip(shap_values, lime_weights):
        if shap_value is None or lime_weight is None:
            agreement.append(None)
        else:
            agreement.append(bool(np.sign(shap_value) == np.sign(lime_weight)))

    frame = pd.DataFrame(
        {
            "feature": result.feature_names,
            "feature_value": result.feature_values,
            "shap_value": shap_values,
            "lime_weight": lime_weights,
            "sign_agreement": agreement,
        }
    )
    sort_column = "shap_value" if result.shap_values is not None else "lime_weight"
    if result.shap_values is not None or result.lime_weights is not None:
        frame = (
            frame.assign(_importance=frame[sort_column].abs())
            .sort_values("_importance", ascending=False, kind="stable")
            .drop(columns="_importance")
        )
    return frame.reset_index(drop=True)


__all__ = [
    "SHAPExplainerType",
    "TREE",
    "LINEAR",
    "KERNEL",
    "XGBOOST",
    "detect_explainer_type",
    "extract_pipeline_parts",
    "build_shap_explainer",
    "compute_shap_values",
    "compute_global_importance",
    "compute_local_explanation",
    "get_last_shap_warning",
    "build_lime_explainer",
    "compute_lime_explanation",
    "ExplanationResult",
    "explain_single_prediction",
    "explain_dataset_globally",
    "compare_shap_lime",
]
