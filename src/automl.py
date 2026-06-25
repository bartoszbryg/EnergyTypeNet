"""Reusable dataset profiling and baseline AutoML utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor


MAX_TARGET_CLASSES = 20


@dataclass
class PreparedDataset:
    """Container for transformed metadata needed by model training."""

    X: pd.DataFrame
    y: np.ndarray
    task_type: str
    target_col: str
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    classes: list[str] | None = None


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove empty rows and columns while preserving the original values."""
    cleaned = df.dropna(axis=0, how='all').dropna(axis=1, how='all')

    return cleaned.reset_index(drop=True)


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Create a compact profile used by the assistant and dashboard."""
    rows = []

    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))
        dtype = str(series.dtype)

        rows.append({
            'column': col,
            'dtype': dtype,
            'missing': missing,
            'missing_pct': float(missing / max(len(df), 1)),
            'unique': unique,
            'example': _safe_example(series),
        })

    return {
        'n_rows': int(len(df)),
        'n_columns': int(df.shape[1]),
        'columns': rows,
        'missing_cells': int(df.isna().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
    }


def guess_task_type(series: pd.Series) -> str:
    """Infer whether a target is more likely classification or regression."""
    non_null = series.dropna()
    unique = non_null.nunique()

    if unique <= 1:
        return 'invalid'

    if pd.api.types.is_numeric_dtype(non_null):
        unique_ratio = unique / max(len(non_null), 1)

        if unique <= MAX_TARGET_CLASSES and unique_ratio <= 0.15:
            return 'classification'

        return 'regression'

    if unique <= MAX_TARGET_CLASSES:
        return 'classification'

    return 'classification'


def suggest_targets(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Rank likely target columns with a simple transparent heuristic."""
    suggestions = []

    last_col = df.columns[-1] if len(df.columns) else None

    for col in df.columns:
        series = df[col]
        unique = int(series.nunique(dropna=True))
        missing_pct = float(series.isna().mean())
        task_type = guess_task_type(series)
        col_lower = col.lower()

        if task_type == 'invalid':
            continue

        score = 0.0

        if 2 <= unique <= MAX_TARGET_CLASSES:
            score += 2.0

        if missing_pct == 0:
            score += 1.0

        if col_lower in {'target', 'label', 'class', 'type', 'category', 'outcome'}:
            score += 2.0

        if any(token in col_lower for token in ['class', 'target', 'label']):
            score += 1.0

        if any(token in col_lower for token in ['risk', 'level', 'status', 'outcome']):
            score += 2.0

        if col == last_col:
            score += 1.0

        if task_type == 'regression' and pd.api.types.is_numeric_dtype(series):
            score += 0.5

        suggestions.append({
            'column': col,
            'task_type': task_type,
            'unique': unique,
            'missing_pct': missing_pct,
            'score': score,
        })

    return sorted(suggestions, key=lambda row: row['score'], reverse=True)


def suggest_features(df: pd.DataFrame, target_col: str) -> list[dict[str, Any]]:
    """Suggest usable feature columns and explain exclusions."""
    rows = []

    for col in df.columns:
        if col == target_col:
            continue

        series = df[col]
        unique = int(series.nunique(dropna=True))
        missing_pct = float(series.isna().mean())
        usable = unique > 1 and missing_pct < 0.8

        reason = 'usable'

        if unique <= 1:
            reason = 'constant'
        elif missing_pct >= 0.8:
            reason = 'too many missing values'

        rows.append({
            'column': col,
            'dtype': str(series.dtype),
            'unique': unique,
            'missing_pct': missing_pct,
            'usable': usable,
            'reason': reason,
        })

    return rows


def prepare_dataset(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    task_type: str | None = None,
) -> PreparedDataset:
    """Select features and encode the target for supervised learning."""
    if target_col not in df.columns:
        raise ValueError(f'Unknown target column: {target_col}')

    if feature_cols is None:
        feature_cols = [
            row['column']
            for row in suggest_features(df, target_col)
            if row['usable']
        ]

    if not feature_cols:
        raise ValueError('At least one usable feature column is required.')

    data = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()

    if task_type is None:
        task_type = guess_task_type(data[target_col])

    X = data[feature_cols]
    numeric_cols = [
        col for col in feature_cols
        if pd.api.types.is_numeric_dtype(X[col])
    ]
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]

    if task_type == 'classification':
        encoder = LabelEncoder()
        y = encoder.fit_transform(data[target_col].astype(str))
        classes = [str(cls) for cls in encoder.classes_]
        _, class_counts = np.unique(y, return_counts=True)

        if len(classes) < 2:
            raise ValueError('Classification requires at least two target classes.')

        if int(class_counts.min()) < 2:
            raise ValueError(
                f'Classification target `{target_col}` has at least one class with '
                'only one row. Choose regression for continuous numeric targets, or '
                'choose a categorical target with repeated class examples.'
            )
    elif task_type == 'regression':
        if not pd.api.types.is_numeric_dtype(data[target_col]):
            raise ValueError(
                f'Regression target `{target_col}` must be numeric. '
                'Choose classification or select a numeric target column.'
            )

        y = data[target_col].astype(float).values
        classes = None
    else:
        raise ValueError(f'Unsupported task type: {task_type}')

    return PreparedDataset(
        X=X,
        y=y,
        task_type=task_type,
        target_col=target_col,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        classes=classes,
    )


def build_preprocessor(prepared: PreparedDataset) -> ColumnTransformer:
    """Build preprocessing for mixed numeric and categorical tables."""
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    return ColumnTransformer([
        ('num', numeric_pipe, prepared.numeric_cols),
        ('cat', categorical_pipe, prepared.categorical_cols),
    ])


def build_baseline_models(task_type: str, n_classes: int | None = None) -> dict[str, Any]:
    """Build a broad set of tabular baseline models."""
    if task_type == 'classification':
        num_class = n_classes or 2
        xgb_objective = 'binary:logistic' if num_class == 2 else 'multi:softprob'
        xgb_params = {
            'objective': xgb_objective,
            'eval_metric': 'logloss' if num_class == 2 else 'mlogloss',
            'max_depth': 3,
            'learning_rate': 0.05,
            'n_estimators': 120,
            'subsample': 0.9,
            'random_state': 42,
            'verbosity': 0,
        }

        if num_class > 2:
            xgb_params['num_class'] = num_class

        return {
            'Dummy baseline': DummyClassifier(strategy='most_frequent'),
            'Logistic Regression': LogisticRegression(max_iter=2000),
            'KNN': KNeighborsClassifier(n_neighbors=3),
            'SVM': SVC(),
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'MLP Neural Network': MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=600,
                early_stopping=False,
                random_state=42,
            ),
            'XGBoost': XGBClassifier(**xgb_params),
        }

    return {
        'Dummy baseline': DummyRegressor(strategy='mean'),
        'Ridge Regression': Ridge(alpha=1.0),
        'KNN Regressor': KNeighborsRegressor(n_neighbors=3),
        'SVR': SVR(),
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=200,
            random_state=42,
        ),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
        'MLP Regressor': MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=800,
            early_stopping=False,
            random_state=42,
        ),
        'XGBoost Regressor': XGBRegressor(
            max_depth=3,
            learning_rate=0.05,
            n_estimators=160,
            subsample=0.9,
            objective='reg:squarederror',
            random_state=42,
            verbosity=0,
        ),
    }


def train_baselines(prepared: PreparedDataset, cv: int = 5) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train and score classification or regression baselines."""
    n_classes = len(prepared.classes) if prepared.classes else None
    models = build_baseline_models(prepared.task_type, n_classes)
    fitted_models = {}
    rows = []

    stratify = prepared.y if prepared.task_type == 'classification' else None
    X_train, X_test, y_train, y_test = train_test_split(
        prepared.X,
        prepared.y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    for name, estimator in models.items():
        pipeline = Pipeline([
            ('preprocess', build_preprocessor(prepared)),
            ('model', estimator),
        ])

        scoring = _scoring_for_task(prepared.task_type)
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=min(cv, _safe_cv_splits(y_train, prepared.task_type)),
            scoring=scoring,
            n_jobs=1,
            error_score='raise',
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        if prepared.task_type == 'classification':
            row = {
                'model': name,
                'cv_accuracy': float(cv_results['test_accuracy'].mean()),
                'cv_f1_macro': float(cv_results['test_f1_macro'].mean()),
                'test_accuracy': float(accuracy_score(y_test, preds)),
                'test_f1_macro': float(f1_score(y_test, preds, average='macro')),
            }
        else:
            row = {
                'model': name,
                'cv_r2': float(cv_results['test_r2'].mean()),
                'cv_mae': float(-cv_results['test_neg_mean_absolute_error'].mean()),
                'test_r2': float(r2_score(y_test, preds)),
                'test_mae': float(mean_absolute_error(y_test, preds)),
            }

        rows.append(row)
        fitted_models[name] = pipeline

    results = pd.DataFrame(rows)

    if prepared.task_type == 'classification':
        results = results.sort_values('test_accuracy', ascending=False)
    else:
        results = results.sort_values('test_r2', ascending=False)

    return results.reset_index(drop=True), fitted_models


def rank_features(prepared: PreparedDataset, top_n: int | None = 12) -> pd.DataFrame:
    """Rank original feature columns with mutual information."""
    X_simple = prepared.X.copy()

    for col in prepared.categorical_cols:
        X_simple[col] = X_simple[col].astype('category').cat.codes

    for col in prepared.numeric_cols:
        X_simple[col] = X_simple[col].fillna(X_simple[col].median())

    X_simple = X_simple.fillna(0)

    if prepared.task_type == 'classification':
        scores = mutual_info_classif(X_simple, prepared.y, random_state=42)
    else:
        scores = mutual_info_regression(X_simple, prepared.y, random_state=42)

    ranked = pd.DataFrame({
        'feature': prepared.feature_cols,
        'mutual_information': scores,
    }).sort_values('mutual_information', ascending=False)

    if top_n is not None:
        ranked = ranked.head(top_n)

    return ranked.reset_index(drop=True)


def recommend_features(prepared: PreparedDataset) -> pd.DataFrame:
    """Combine feature quality signals into a clearer recommendation table."""
    ranking = rank_features(prepared, top_n=None)
    rows = []

    max_mi = max(float(ranking['mutual_information'].max()), 1e-12)
    mi_lookup = dict(zip(ranking['feature'], ranking['mutual_information']))

    for col in prepared.feature_cols:
        series = prepared.X[col]
        missing_pct = float(series.isna().mean())
        unique = int(series.nunique(dropna=True))
        mutual_info = float(mi_lookup.get(col, 0.0))
        normalized_mi = mutual_info / max_mi
        quality_score = (0.8 * normalized_mi) + (0.2 * (1 - missing_pct))

        if unique <= 1:
            recommendation = 'Remove'
            reason = 'Constant or nearly constant feature.'
        elif missing_pct >= 0.5:
            recommendation = 'Weak'
            reason = 'Too many missing values.'
        elif normalized_mi >= 0.65:
            recommendation = 'Strong'
            reason = 'High measured relationship with the target.'
        elif normalized_mi >= 0.25:
            recommendation = 'Moderate'
            reason = 'Some measured relationship with the target.'
        else:
            recommendation = 'Weak'
            reason = 'Low measured relationship with the target.'

        rows.append({
            'feature': col,
            'dtype': str(series.dtype),
            'unique': unique,
            'missing_pct': missing_pct,
            'mutual_information': mutual_info,
            'quality_score': quality_score,
            'recommendation': recommendation,
            'reason': reason,
        })

    return pd.DataFrame(rows).sort_values(
        ['quality_score', 'mutual_information'],
        ascending=False,
    ).reset_index(drop=True)


def generate_dataset_report(
    profile: dict[str, Any],
    target_suggestions: list[dict[str, Any]],
    prepared: PreparedDataset | None = None,
    results: pd.DataFrame | None = None,
    feature_ranking: pd.DataFrame | None = None,
) -> str:
    """Generate a short natural-language dataset report."""
    lines = [
        '# Dataset Report',
        '',
        f"The dataset contains {profile['n_rows']:,} rows and "
        f"{profile['n_columns']:,} columns.",
        f"It has {profile['missing_cells']:,} missing cells and "
        f"{profile['duplicate_rows']:,} duplicate rows.",
    ]

    if target_suggestions:
        best = target_suggestions[0]
        lines.extend([
            '',
            f"The strongest target candidate is `{best['column']}`, "
            f"which looks like a {best['task_type']} target.",
        ])

    if prepared is not None:
        lines.extend([
            '',
            f"Selected task: **{prepared.task_type}**.",
            f"Target column: `{prepared.target_col}`.",
            f"Selected features: {len(prepared.feature_cols)} total "
            f"({len(prepared.numeric_cols)} numeric, "
            f"{len(prepared.categorical_cols)} categorical).",
        ])

    if results is not None and not results.empty:
        best_row = results.iloc[0]

        if prepared and prepared.task_type == 'classification':
            lines.extend([
                '',
                f"Best baseline model: **{best_row['model']}**.",
                f"Test accuracy: **{best_row['test_accuracy']:.3f}**.",
                f"Macro F1: **{best_row['test_f1_macro']:.3f}**.",
            ])
        else:
            lines.extend([
                '',
                f"Best baseline model: **{best_row['model']}**.",
                f"Test R2: **{best_row['test_r2']:.3f}**.",
                f"Test MAE: **{best_row['test_mae']:.3f}**.",
            ])

    if feature_ranking is not None and not feature_ranking.empty:
        top_features = ', '.join(
            f"`{row.feature}`"
            for row in feature_ranking.head(5).itertuples()
        )
        lines.extend([
            '',
            f"The strongest feature candidates by mutual information are {top_features}.",
        ])

    lines.extend([
        '',
        'Interpretation should stay tied to these measured outputs. '
        'If the best model only slightly beats the dummy baseline, the dataset may need '
        'better features, more rows, or a different target definition.',
    ])

    return '\n'.join(lines)


def answer_dataset_question(
    question: str,
    profile: dict[str, Any],
    prepared: PreparedDataset | None,
    results: pd.DataFrame | None,
    feature_ranking: pd.DataFrame | None,
) -> str:
    """Answer common dataset questions from computed statistics only."""
    q = question.lower()

    if any(word in q for word in ['overfit', 'overfitting', 'generalize', 'generalization', 'leakage', 'leak']):
        if results is None or results.empty:
            return (
                'Train baseline models first so I can compare cross-validation and '
                'test performance. Overfitting is measured from that gap, not guessed.'
            )

        if prepared and prepared.task_type == 'classification':
            best = results.iloc[0]
            cv_score = float(best['cv_accuracy'])
            test_score = float(best['test_accuracy'])
            gap = cv_score - test_score
            score_text = (
                f"The best model is {best['model']} with CV accuracy "
                f"{cv_score:.3f} and test accuracy {test_score:.3f}."
            )
        else:
            best = results.iloc[0]
            cv_score = float(best['cv_r2'])
            test_score = float(best['test_r2'])
            gap = cv_score - test_score
            score_text = (
                f"The best model is {best['model']} with CV R2 "
                f"{cv_score:.3f} and test R2 {test_score:.3f}."
            )

        if gap > 0.10:
            risk_text = 'This suggests possible overfitting because CV is noticeably higher than the test score.'
        elif gap < -0.10:
            risk_text = 'The test score is higher than CV, so the holdout split may be small or unusually easy.'
        else:
            risk_text = 'The CV/test gap is small, so there is no strong overfitting signal from these metrics.'

        leakage_text = ''
        weak_feature_text = ''

        if feature_ranking is not None and not feature_ranking.empty:
            top = feature_ranking.iloc[0]

            if float(top['mutual_information']) > 1.0:
                leakage_text = (
                    f" `{top['feature']}` has very high mutual information "
                    f"({float(top['mutual_information']):.3f}), so inspect it for possible target leakage."
                )

            weak_features = feature_ranking[
                feature_ranking['mutual_information'] <= 0.01
            ]['feature'].head(3).astype(str).tolist()

            if weak_features:
                weak_feature_text = (
                    ' If you train with all selected features, compare it against the compact feature set; '
                    f"low-information columns such as {', '.join(weak_features)} may add noise."
                )

        return f'{score_text} {risk_text}{leakage_text}{weak_feature_text}'

    if any(word in q for word in ['missing', 'null', 'nan']):
        return (
            f"The dataset has {profile['missing_cells']:,} missing cells. "
            "Check the profile table for column-level missing percentages."
        )

    if any(word in q for word in ['best', 'model', 'accuracy', 'score', 'r2']):
        if results is None or results.empty:
            return 'Train baseline models first, then I can summarize the best model.'

        best = results.iloc[0]

        if prepared and prepared.task_type == 'classification':
            return (
                f"The best baseline is {best['model']} with test accuracy "
                f"{best['test_accuracy']:.3f} and macro F1 {best['test_f1_macro']:.3f}."
            )

        return (
            f"The best baseline is {best['model']} with test R2 "
            f"{best['test_r2']:.3f} and MAE {best['test_mae']:.3f}."
        )

    if any(word in q for word in ['feature', 'important', 'column']):
        if feature_ranking is None or feature_ranking.empty:
            return 'Run feature ranking first, then I can summarize the strongest columns.'

        names = ', '.join(feature_ranking['feature'].head(5).astype(str))
        return f"The strongest measured feature candidates are: {names}."

    if any(word in q for word in ['classification', 'regression', 'task']):
        if prepared is None:
            return 'Pick a target column first so I can infer the task type.'

        return (
            f"This is being treated as a {prepared.task_type} problem because "
            f"the selected target is `{prepared.target_col}`."
        )

    return (
        'I can answer questions about missing values, likely target columns, '
        'feature importance, task type, and baseline model results based on the '
        'computed dataset profile.'
    )


def _safe_example(series: pd.Series) -> str:
    values = series.dropna()

    if values.empty:
        return ''

    return str(values.iloc[0])


def _scoring_for_task(task_type: str) -> dict[str, str]:
    if task_type == 'classification':
        return {
            'accuracy': 'accuracy',
            'f1_macro': 'f1_macro',
        }

    return {
        'r2': 'r2',
        'neg_mean_absolute_error': 'neg_mean_absolute_error',
    }


def _safe_cv_splits(y: np.ndarray, task_type: str) -> int:
    if task_type == 'regression':
        return 5

    _, counts = np.unique(y, return_counts=True)

    return max(2, min(5, int(counts.min())))

