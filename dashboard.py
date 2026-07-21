"""EnergyTypeNet Streamlit visualization dashboard.

Two modes:
  - EnergyTypeNet: pre-loaded energy dataset with custom and sklearn models.
  - Custom Dataset: upload a CSV and run the same evaluation pipeline.

Run from repo root:
    streamlit run dashboard.py
"""

import io
import hashlib
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score,
    precision_recall_curve, average_precision_score,
    roc_curve, auc,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from xgboost import XGBClassifier

from src.automl import (
    build_model_card_data,
    clean_dataframe,
    generate_dataset_report,
    guess_task_type,
    prepare_dataset,
    profile_dataset,
    rank_features,
    recommend_features,
    suggest_features,
    suggest_targets,
    train_baselines,
)
from src.agent_tools import run_dataset_computation
from src.llm_assistant import (
    COST_PER_1K_TOKENS,
    DEFAULT_OLLAMA_MODEL,
    PROVIDERS,
    PROVIDER_MODELS,
    UsageTracker,
    load_api_key,
    stream_llm,
    stream_with_fallback,
)
from src.chat_agent import (
    ChatHistory,
    ChatMessage,
    build_contextualized_prompt,
    classify_question,
    handle_follow_up,
    suggest_next_questions,
    utc_timestamp,
)
from src.data import CLASSES as ENERGY_CLASSES, load_features, load_raw
from src.dashboard_validation import (
    recommend_classification_targets,
    validate_classification_target,
)
from src.preloaded_artifacts import (
    PreloadedArtifactError,
    load_artifact as load_preloaded_artifact,
    train_energy_models,
)
from sklearn.exceptions import ConvergenceWarning

try:
    from src.data_validation import (
        ValidationReport,
        check_column_coverage,
        run_all_leakage_checks,
        run_all_schema_checks,
        run_complete_validation,
    )
    DATA_VALIDATION_AVAILABLE = True
except ImportError:
    ValidationReport = None
    check_column_coverage = None
    run_all_leakage_checks = None
    run_all_schema_checks = None
    run_complete_validation = None
    DATA_VALIDATION_AVAILABLE = False

try:
    from src.model_card import (
        ModelCardData,
        add_chat_explanations,
        collect_from_energy_dataset,
        export_model_card_bytes,
        render_full_model_card,
        select_notable_exchanges,
    )
    MODEL_CARD_AVAILABLE = True
except ImportError:
    ModelCardData = None
    add_chat_explanations = None
    collect_from_energy_dataset = None
    export_model_card_bytes = None
    render_full_model_card = None
    select_notable_exchanges = None
    MODEL_CARD_AVAILABLE = False

try:
    from src.explainability import (
        ExplanationResult,
        build_shap_explainer,
        compare_shap_lime,
        explain_dataset_globally,
        explain_single_prediction,
    )
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    ExplanationResult = None
    build_shap_explainer = None
    compare_shap_lime = None
    explain_dataset_globally = None
    explain_single_prediction = None
    EXPLAINABILITY_AVAILABLE = False

try:
    import shap as shap_library
    SHAP_AVAILABLE = True
except ImportError:
    shap_library = None
    SHAP_AVAILABLE = False

try:
    from lime import lime_tabular as lime_tabular_library
    LIME_AVAILABLE = True
except ImportError:
    lime_tabular_library = None
    LIME_AVAILABLE = False

st.set_page_config(
    page_title='EnergyTypeNet',
    page_icon='building',
    layout='wide',
    initial_sidebar_state='expanded',
)

CMAP_LIGHT = ListedColormap(['#FFCCCC', '#CCFFCC', '#CCCCFF'])
CMAP_BOLD = ListedColormap(['#CC0000', '#006600', '#0000CC'])
CLASS_COLORS = ['#CC0000', '#006600', '#0000CC', '#FF8800', '#8800FF', '#008888']


# Session defaults

if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = 'none'
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = None
if 'llm_api_key' not in st.session_state:
    st.session_state.llm_api_key = None
if 'usage_tracker' not in st.session_state:
    st.session_state.usage_tracker = UsageTracker()
if 'shap_explainer_cache' not in st.session_state:
    st.session_state.shap_explainer_cache = {}
if 'global_explanation_cache' not in st.session_state:
    st.session_state.global_explanation_cache = {}
if 'explanation_history' not in st.session_state:
    st.session_state.explanation_history = []
if 'custom_explanation_history' not in st.session_state:
    st.session_state.custom_explanation_history = []


# Shared helpers

def render_explanation_unavailable(*, require_shap=False, require_lime=False):
    """Explain which optional dependency is needed and return availability."""

    if not EXPLAINABILITY_AVAILABLE:
        st.info(
            'Explainability is unavailable because src/explainability.py could not be '
            'imported. Reinstall the project dependencies and restart the app.'
        )
        return False
    missing = []
    if require_shap and not SHAP_AVAILABLE:
        missing.append('SHAP (`pip install shap`)')
    if require_lime and not LIME_AVAILABLE:
        missing.append('LIME (`pip install lime`)')
    if missing:
        st.info('Install ' + ' and '.join(missing) + ' to enable this explanation.')
        return False
    return True


def _clear_explanation_state(prefix):
    """Drop cached explainers/results belonging to a changed dataset or model set."""

    for name in ('shap_explainer_cache', 'global_explanation_cache'):
        cache = st.session_state.get(name, {})
        st.session_state[name] = {
            key: value for key, value in cache.items() if not str(key).startswith(prefix)
        }
    if prefix == 'custom:':
        st.session_state.custom_explanation_history = []


def _cached_shap_explainer(cache_key, model, background, feature_names):
    if not SHAP_AVAILABLE or not EXPLAINABILITY_AVAILABLE:
        return None
    cache = st.session_state.shap_explainer_cache
    if cache_key not in cache:
        cache[cache_key] = build_shap_explainer(model, background, feature_names)
    return cache[cache_key]


def render_shap_waterfall(result):
    """Render native SHAP output when possible, with a deterministic fallback."""

    if not result.shap_values:
        st.info('No SHAP explanation is available for this prediction.')
        return
    values = np.asarray(result.shap_values, dtype=float)
    names = np.asarray(result.feature_names, dtype=object)
    feature_values = np.asarray(result.feature_values, dtype=object)
    base = float(result.shap_base_value or 0.0)

    if SHAP_AVAILABLE:
        try:
            explanation = shap_library.Explanation(
                values=values,
                base_values=base,
                data=feature_values,
                feature_names=list(names),
            )
            shap_library.plots.waterfall(
                explanation, max_display=min(15, len(values)), show=False
            )
            st.pyplot(plt.gcf(), clear_figure=True)
            return
        except Exception:
            plt.close('all')

    order = np.argsort(np.abs(values))[::-1][:15]
    ordered_values = values[order]
    ordered_names = names[order]
    running = base
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.42 * len(order))))
    for position, (name, contribution) in enumerate(
        zip(ordered_names, ordered_values)
    ):
        next_value = running + contribution
        ax.barh(
            position,
            abs(contribution),
            left=min(running, next_value),
            color='#2ca02c' if contribution >= 0 else '#d62728',
            alpha=0.85,
        )
        running = next_value
    ax.axvline(base, color='#555', linestyle='--', label=f'base = {base:.3f}')
    ax.axvline(running, color='#111', linestyle=':', label=f'output = {running:.3f}')
    ax.set_yticks(range(len(order)), ordered_names)
    ax.invert_yaxis()
    ax.set_xlabel('Model output')
    ax.set_title('SHAP contribution waterfall')
    ax.legend(loc='best')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_lime_bar_chart(result):
    if not result.lime_weights:
        st.info('No LIME explanation is available for this prediction.')
        return
    values = np.asarray(result.lime_weights, dtype=float)
    names = np.asarray(result.feature_names, dtype=object)
    order = np.argsort(np.abs(values))[-15:]
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.4 * len(order))))
    ax.barh(
        names[order], values[order],
        color=np.where(values[order] >= 0, '#2ca02c', '#d62728'),
    )
    ax.axvline(0, color='#555', linewidth=1)
    ax.set_xlabel('Local LIME weight')
    ax.set_title('LIME local feature contributions')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_local_explanation(result):
    st.markdown(f'**{result.summary_text()}**')
    details = []
    if result.predicted_class is not None:
        details.append(f'prediction: `{result.predicted_class}`')
    if result.predicted_probability is not None:
        details.append(f'probability: `{result.predicted_probability:.3f}`')
    if result.shap_explainer_type:
        details.append(f'SHAP explainer: `{result.shap_explainer_type}`')
    if result.lime_local_r2 is not None:
        details.append(f'LIME local R²: `{result.lime_local_r2:.3f}`')
    if details:
        st.caption(' · '.join(details))

    shap_column, lime_column = st.columns(2)
    with shap_column:
        st.markdown('#### SHAP')
        render_shap_waterfall(result)
    with lime_column:
        st.markdown('#### LIME')
        render_lime_bar_chart(result)

    if result.shap_values is not None and result.lime_weights is not None:
        with st.expander('Compare SHAP and LIME', expanded=False):
            comparison = compare_shap_lime(result)
            st.dataframe(comparison, width="stretch", hide_index=True)
            agreement = comparison['sign_agreement'].dropna()
            if len(agreement):
                st.caption(f'Sign agreement: {agreement.mean():.1%}')


def render_global_explanation(global_importance, model=None):
    st.dataframe(global_importance, width="stretch", hide_index=True)
    ordered = global_importance.sort_values('mean_abs_shap')
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.4 * len(ordered))))
    ax.barh(ordered['feature'], ordered['mean_abs_shap'], color='#1f77b4')
    ax.set_xlabel('Mean absolute SHAP value')
    ax.set_title('Global SHAP feature importance')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    class_columns = [
        column for column in global_importance.columns
        if column.startswith('mean_abs_shap_')
    ]
    if class_columns:
        with st.expander('Per-class SHAP importance', expanded=False):
            st.dataframe(
                global_importance[['feature', *class_columns]],
                width="stretch",
                hide_index=True,
            )

    estimator = model.steps[-1][1] if hasattr(model, 'steps') else model
    standard = None
    if hasattr(estimator, 'feature_importances_'):
        standard = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, 'coef_'):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        standard = np.mean(np.abs(coefficients), axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
    if standard is not None and len(standard) == len(global_importance):
        comparison = global_importance[['feature', 'mean_abs_shap']].copy()
        comparison['standard_importance'] = standard
        st.markdown('#### SHAP vs model-native importance')
        st.dataframe(comparison, width="stretch", hide_index=True)


def _run_local_explanation(
    *, cache_key, model, sample, background, feature_names, class_names,
    sample_index=None,
):
    explainer = _cached_shap_explainer(
        cache_key, model, background, feature_names
    ) if SHAP_AVAILABLE else None
    return explain_single_prediction(
        model=model,
        sample=sample,
        feature_names=feature_names,
        class_names=class_names,
        background_data=background,
        task_type='classification',
        sample_index=sample_index,
        use_shap=SHAP_AVAILABLE,
        use_lime=LIME_AVAILABLE,
        shap_explainer=explainer,
    )

def display_validation_report(
    report,
    *,
    heading='Data Quality Analysis',
    errors_are_blocking=True,
):
    """Render structured validation findings and return whether errors are absent."""

    st.subheader(heading)
    if not report.issues:
        st.success('Validation passed: no data-quality issues were detected.')
        return True

    if report.errors():
        issue_kind = 'blocking issue' if errors_are_blocking else 'critical diagnostic issue'
        st.error(
            f'Validation found {len(report.errors())} {issue_kind}'
            f'{"s" if len(report.errors()) != 1 else ""}.'
        )
        for issue in report.errors():
            location = f' **{issue.column}:**' if issue.column else ''
            st.markdown(f'-{location} {issue.message}')
    elif report.warnings():
        st.warning(
            f'Validation passed with {len(report.warnings())} warning'
            f'{"s" if len(report.warnings()) != 1 else ""}. Review the details below.'
        )
    else:
        st.info('Validation passed with informational notes.')

    category_labels = {
        'schema': ('Schema Issues', '📋'),
        'leakage': ('Leakage Findings', '⚠️'),
        'drift': ('Drift Warnings', '📈'),
    }
    for category, (label, icon) in category_labels.items():
        findings = report.by_category(category)
        if not findings:
            continue
        with st.expander(f'{icon} {label} ({len(findings)})', expanded=False):
            for issue in findings:
                severity_icon = {'error': '🔴', 'warning': '🟠', 'info': '🔵'}[issue.severity]
                location = f' **{issue.column}:**' if issue.column else ''
                st.markdown(f'{severity_icon}{location} {issue.message}')
                if issue.suggestion:
                    st.markdown(
                        f'<span style="color:#777">Suggestion: {issue.suggestion}</span>',
                        unsafe_allow_html=True,
                    )
    return report.passed

def _safe_download_stem(name: str) -> str:
    stem = name.rsplit('.', 1)[0]
    cleaned = ''.join(
        character if character.isalnum() or character in {'-', '_'} else '_'
        for character in stem
    ).strip('_')
    return cleaned or 'dataset'


def _pdf_export_status() -> tuple[bool, str | None]:
    try:
        import markdown  # noqa: F401
        import weasyprint  # noqa: F401
    except (ImportError, OSError) as exc:
        return False, str(exc)
    return True, None


def _render_model_card_downloads(card, filename_stem: str) -> None:
    """Render Markdown and optional PDF download controls."""
    left, right = st.columns(2)
    markdown_bytes = export_model_card_bytes(card, format='markdown')
    with left:
        st.download_button(
            'Download Markdown (.md)',
            data=markdown_bytes,
            file_name=f'{filename_stem}_model_card.md',
            mime='text/markdown',
            key=f'model_card_md_{filename_stem}',
        )

    pdf_available, _ = _pdf_export_status()
    with right:
        if pdf_available:
            try:
                pdf_bytes = export_model_card_bytes(card, format='pdf')
                st.download_button(
                    'Download PDF (.pdf)',
                    data=pdf_bytes,
                    file_name=f'{filename_stem}_model_card.pdf',
                    mime='application/pdf',
                    key=f'model_card_pdf_{filename_stem}',
                )
            except (ImportError, OSError) as exc:
                st.button('Download PDF (.pdf)', disabled=True)
                st.caption(f'PDF export is unavailable: {exc}')
        else:
            st.button('Download PDF (.pdf)', disabled=True)
            st.caption('Install PDF support with `pip install markdown weasyprint`.')


@st.cache_data
def _compute_energy_comparison(_models, _X_tr, _y_tr, _X_te, _y_te, _X_tr_sc, _X_te_sc):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, (model, _, is_scaled) in _models.items():
        X_cv = _X_tr_sc if is_scaled else _X_tr
        scores = cross_val_score(model, X_cv, _y_tr, cv=skf, scoring='accuracy')
        rows.append({
            'Model': name,
            'CV Mean': scores.mean(),
            'CV Std': scores.std(),
            'Test Acc': accuracy_score(
                _y_te,
                model.predict(_X_te_sc if is_scaled else _X_te),
            ),
        })
    return rows

def build_sklearn_models(n_classes: int, random_state: int = 42) -> dict:
    """Standard sklearn + XGBoost pipelines, compatible with any feature count."""
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=10, max_iter=2000, random_state=random_state),
    )

    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(40, 20),
            activation='relu',
            alpha=0.01,
            max_iter=1200,
            # Cross-validation already estimates generalization. Disabling the
            # estimator's extra validation split keeps small uploads reliable.
            early_stopping=False,
            random_state=random_state,
        ),
    )

    xgb_options = dict(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=100,
        eval_metric='mlogloss',
        verbosity=0,
        random_state=random_state,
    )
    if n_classes == 2:
        xgb_options['objective'] = 'binary:logistic'
    else:
        xgb_options.update(objective='multi:softprob', num_class=n_classes)
    xgb = XGBClassifier(**xgb_options)

    return {'LR (sklearn)': lr, 'MLP': mlp, 'XGBoost': xgb}


def cv_score(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return cross_val_score(model, X, y, cv=skf, scoring='accuracy')


def fig_model_comparison(results: dict) -> plt.Figure:
    names = list(results)
    means = [v['cv_mean'] for v in results.values()]
    stds = [v['cv_std'] for v in results.values()]
    tests = [v.get('test_acc', 0) for v in results.values()]

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.5), 4))
    x = np.arange(len(names))

    ax.bar(
        x - 0.2,
        means,
        0.35,
        yerr=stds,
        capsize=4,
        label='CV Mean',
        color='steelblue',
        error_kw={'elinewidth': 1},
    )
    ax.bar(x + 0.2, tests, 0.35, label='Test Acc', color='darkorange')
    ax.axhline(
        1 / max(len(names), 2),
        color='red',
        ls='--',
        alpha=0.4,
        label='Random baseline',
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()

    return fig


def fig_confusion(model, X_eval, y_test, classes, title):
    preds = model.predict(X_eval)
    acc = accuracy_score(y_test, preds)

    fig, ax = plt.subplots(figsize=(4, 3.5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        preds,
        display_labels=classes,
        ax=ax,
        colorbar=False,
        cmap='Blues',
    )
    ax.set_title(f'{title}\nacc={acc:.2f}', fontsize=9)

    plt.tight_layout()

    return fig


def fig_roc(model, X_eval, y_test, classes, title):
    if not hasattr(model, 'predict_proba'):
        return None

    n_cls = len(classes)
    y_bin = label_binarize(y_test, classes=list(range(n_cls)))

    if n_cls == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    proba = model.predict_proba(X_eval)
    fig, ax = plt.subplots(figsize=(4.5, 4))

    for i, cls in enumerate(classes):
        col = CLASS_COLORS[i % len(CLASS_COLORS)]
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        ax.plot(
            fpr,
            tpr,
            color=col,
            lw=1.5,
            label=f'{cls} (AUC={auc(fpr, tpr):.2f})',
        )

    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)

    plt.tight_layout()

    return fig


def fig_pr(model, X_eval, y_test, classes, title):
    if not hasattr(model, 'predict_proba'):
        return None

    n_cls = len(classes)
    proba = model.predict_proba(X_eval)
    fig, ax = plt.subplots(figsize=(4.5, 4))

    for i, cls in enumerate(classes):
        col = CLASS_COLORS[i % len(CLASS_COLORS)]
        y_bin = (y_test == i).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, proba[:, i])
        ap = average_precision_score(y_bin, proba[:, i])

        ax.plot(rec, prec, color=col, lw=1.5, label=f'{cls} AP={ap:.2f}')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    return fig


def fig_decision_boundary_2d(model, X2, y, classes, sc2, title, h=0.08):
    """Decision boundary in a 2-D scaled space (PCA or original 2 features)."""
    x0_min, x0_max = X2[:, 0].min() - 0.6, X2[:, 0].max() + 0.6
    x1_min, x1_max = X2[:, 1].min() - 0.6, X2[:, 1].max() + 0.6
    xx, yy = np.meshgrid(
        np.arange(x0_min, x0_max, h),
        np.arange(x1_min, x1_max, h),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    n_cls = len(classes)
    cmap_light = ListedColormap(CLASS_COLORS[:n_cls])
    cmap_bold = ListedColormap(CLASS_COLORS[:n_cls])

    # Pipelines scale internally, so the grid is converted back to raw 2-D values.
    if hasattr(model, 'named_steps'):
        grid_raw = sc2.inverse_transform(grid)
        Z = model.predict(grid_raw)
    else:
        Z = model.predict(grid)

    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pcolormesh(xx, yy, Z, cmap=cmap_light, alpha=0.55, shading='auto')
    ax.scatter(
        X2[:, 0],
        X2[:, 1],
        c=y,
        cmap=cmap_bold,
        edgecolors='k',
        s=12,
        alpha=0.55,
        linewidths=0.3,
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('PC 1' if sc2 is not None else 'Feature 1')
    ax.set_ylabel('PC 2' if sc2 is not None else 'Feature 2')

    plt.tight_layout()

    return fig


def fig_learning(model, X, y, title):
    _, class_counts = np.unique(y, return_counts=True)
    if len(X) < 50 or int(class_counts.min()) < 10:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.axis('off')
        ax.text(
            0.5,
            0.5,
            'Learning curve not shown for this small dataset.\n'
            'Use at least 50 rows and 10 examples per class\n'
            'for stable incremental training subsets.',
            ha='center',
            va='center',
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.set_title(title, fontsize=9)
        return fig

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            tr_sizes, tr_sc, val_sc = learning_curve(
                model,
                X,
                y,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                train_sizes=np.linspace(0.1, 1.0, 7),
                scoring='accuracy',
                n_jobs=1,
                # Some incremental subsets can omit a class. In particular,
                # XGBoost requires locally contiguous labels, so fail cleanly
                # instead of emitting FitFailedWarning and plotting NaN scores.
                error_score='raise',
            )
    except (ValueError, RuntimeError):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.axis('off')
        ax.text(
            0.5,
            0.5,
            'Learning curve not available for these incremental subsets.\n'
            'At least one subset does not contain every target class.',
            ha='center',
            va='center',
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title(title, fontsize=9)
        return fig

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(tr_sizes, tr_sc.mean(1), 'o-', color='steelblue', label='Train')
    ax.fill_between(
        tr_sizes,
        tr_sc.mean(1) - tr_sc.std(1),
        tr_sc.mean(1) + tr_sc.std(1),
        alpha=0.15,
        color='steelblue',
    )
    ax.plot(tr_sizes, val_sc.mean(1), 's-', color='darkorange', label='CV val')
    ax.fill_between(
        tr_sizes,
        val_sc.mean(1) - val_sc.std(1),
        val_sc.mean(1) + val_sc.std(1),
        alpha=0.15,
        color='darkorange',
    )
    ax.axhline(1 / 3, color='red', ls='--', alpha=0.4, label='Baseline')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Training examples')
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=7)
    ax.set_ylim(0.1, 1.05)

    plt.tight_layout()

    return fig


def fig_precomputed_learning(curve: dict, title: str) -> plt.Figure:
    """Render a learning curve without refitting a model."""
    train_sizes = curve['train_sizes']
    train_mean = curve['train_mean']
    train_std = curve['train_std']
    validation_mean = curve['validation_mean']
    validation_std = curve['validation_std']

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(train_sizes, train_mean, 'o-', color='steelblue', label='Train')
    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color='steelblue',
    )
    ax.plot(train_sizes, validation_mean, 's-', color='darkorange', label='CV val')
    ax.fill_between(
        train_sizes,
        validation_mean - validation_std,
        validation_mean + validation_std,
        alpha=0.15,
        color='darkorange',
    )
    ax.axhline(1 / 3, color='red', ls='--', alpha=0.4, label='Baseline')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Training examples')
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=7)
    ax.set_ylim(0.1, 1.05)
    plt.tight_layout()
    return fig


# EnergyTypeNet mode

@st.cache_data
def _load_energy():
    train_df = load_raw('data/train_energy_data.csv')
    test_df = load_raw('data/test_energy_data.csv')

    X_tr, y_tr = load_features('data/train_energy_data.csv', 'core')
    X_te, y_te = load_features('data/test_energy_data.csv', 'core')
    sc = StandardScaler().fit(X_tr)

    return train_df, test_df, X_tr, y_tr, X_te, y_te, sc


@st.cache_resource
def _load_energy_resources(_sc):
    """Load versioned demo assets, with live training as a safe fallback."""
    try:
        return load_preloaded_artifact(), None
    except PreloadedArtifactError as exc:
        fallback = {
            'models': train_energy_models(_sc),
            'comparison': None,
            'learning_curves': {},
        }
        return fallback, str(exc)


def _energy_explanation_model(model, is_scaled, scaler):
    """Expose scaled demo models through a raw-feature pipeline."""

    return make_pipeline(scaler, model) if is_scaled else model


def render_energy_dashboard(page: str):
    train_df, test_df, X_tr, y_tr, X_te, y_te, sc = _load_energy()

    with st.spinner('Loading precomputed demo models...'):
        resources, artifact_warning = _load_energy_resources(sc)
    models = resources['models']

    if artifact_warning:
        st.warning(
            'Precomputed demo assets were unavailable or incompatible, so the '
            f'models were trained safely at runtime. Details: {artifact_warning}'
        )

    X_te_sc = sc.transform(X_te)
    X_tr_sc = sc.transform(X_tr)

    if page == 'Overview':
        st.title('EnergyTypeNet - Overview')
        c1, c2, c3, c4 = st.columns(4)

        c1.metric('Training buildings', len(train_df))
        c2.metric('Test buildings', len(test_df))
        c3.metric('Classes', 3)
        c4.metric('Core features', 2)

        accs = {
            n: accuracy_score(y_te, m.predict(X_te_sc if s else X_te))
            for n, (m, _, s) in models.items()
        }
        acc_df = pd.DataFrame({
            'Model': list(accs),
            'Test Accuracy': list(accs.values()),
        }).sort_values('Test Accuracy', ascending=False)

        st.subheader('Test-set accuracy')
        st.bar_chart(acc_df.set_index('Model')['Test Accuracy'])
        st.dataframe(
            acc_df.style.format({'Test Accuracy': '{:.3f}'}),
            width='stretch',
            hide_index=True,
        )
        st.info('All models use only **Energy Consumption** and **Square Footage**. '
                'The 60-67 % ceiling reflects class overlap in this 2-D space.')

    elif page == 'EDA':
        st.title('Exploratory Data Analysis')
        st.bar_chart(train_df['Building Type'].value_counts())
        feat = st.selectbox(
            'Feature distribution',
            [
                'Energy Consumption',
                'Square Footage',
                'Number of Occupants',
                'Appliances Used',
                'Average Temperature',
            ],
        )

        fig, ax = plt.subplots(figsize=(8, 3.5))

        for cls, col in zip(ENERGY_CLASSES, CLASS_COLORS):
            ax.hist(
                train_df[train_df['Building Type'] == cls][feat],
                bins=30,
                alpha=0.55,
                color=col,
                label=cls,
                edgecolor='white',
            )

        ax.set_xlabel(feat)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(train_df, width='stretch')

    elif page == 'Model Comparison':
        st.title('Model Comparison - 5-fold CV')
        rows = resources.get('comparison')
        if rows is None:
            rows = _compute_energy_comparison(
                models, X_tr, y_tr, X_te, y_te, X_tr_sc, X_te_sc
            )
        cmp_df = pd.DataFrame(rows).sort_values('CV Mean', ascending=False)
        st.dataframe(
            cmp_df.style.format({
                'CV Mean': '{:.3f}',
                'CV Std': '{:.3f}',
                'Test Acc': '{:.3f}',
            }),
            width='stretch',
            hide_index=True,
        )

    elif page == 'Decision Boundaries':
        st.title('Decision Boundaries')
        st.caption('Energy Consumption x Square Footage (scaled). Background = predicted class.')
        cols = st.columns(3)

        for idx, (name, (m, _, _s)) in enumerate(models.items()):
            fig = fig_decision_boundary_2d(m, X_tr_sc, y_tr, ENERGY_CLASSES, sc, name)
            cols[idx % 3].pyplot(fig)
            plt.close(fig)

    elif page == 'Confusion Matrices':
        st.title('Confusion Matrices - Test Set')
        cols = st.columns(3)

        for idx, (name, (m, _, is_sc)) in enumerate(models.items()):
            fig = fig_confusion(m, X_te_sc if is_sc else X_te, y_te, ENERGY_CLASSES, name)
            cols[idx % 3].pyplot(fig)
            plt.close(fig)

    elif page == 'ROC / AUC':
        st.title('ROC Curves')
        cols = st.columns(3)

        for idx, (name, (m, _, is_sc)) in enumerate(models.items()):
            fig = fig_roc(m, X_te_sc if is_sc else X_te, y_te, ENERGY_CLASSES, name)

            if fig:
                cols[idx % 3].pyplot(fig)
                plt.close(fig)

    elif page == 'Precision-Recall':
        st.title('Precision-Recall Curves + Threshold Tuning')
        cols = st.columns(3)
        idx = 0

        for name, (m, _, is_sc) in models.items():
            fig = fig_pr(m, X_te_sc if is_sc else X_te, y_te, ENERGY_CLASSES, name)

            if fig:
                cols[idx % 3].pyplot(fig)
                plt.close(fig)
                idx += 1

        st.subheader('Threshold sweep - XGBoost, Industrial class')
        xgb_m = models['XGBoost'][0]
        proba = xgb_m.predict_proba(X_te)
        y_ind = (y_te == 2).astype(int)
        prec, rec, thresholds = precision_recall_curve(y_ind, proba[:, 2])

        fig2, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(thresholds, prec[:-1], 'b-', label='Precision')
        axes[0].plot(thresholds, rec[:-1], 'r-', label='Recall')
        axes[0].axvline(0.5, color='gray', ls='--', alpha=0.6, label='Default (0.5)')
        axes[0].set_xlabel('Threshold')
        axes[0].legend()
        axes[0].set_title('Precision & Recall vs threshold')

        sc2 = axes[1].scatter(rec[:-1], prec[:-1], c=thresholds, cmap='viridis', s=8)
        plt.colorbar(sc2, ax=axes[1], label='Threshold')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('PR curve colored by threshold')

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    elif page == 'Learning Curves':
        st.title('Learning Curves')
        cols = st.columns(3)
        idx = 0
        precomputed_curves = resources.get('learning_curves', {})

        for name, (m, _, _s) in models.items():
            if not hasattr(m, 'named_steps'):
                continue

            curve = precomputed_curves.get(name)
            fig = (
                fig_precomputed_learning(curve, name)
                if curve is not None
                else fig_learning(m, X_tr, y_tr, name)
            )
            cols[idx % 3].pyplot(fig)
            plt.close(fig)
            idx += 1

    elif page == 'Explanations':
        st.title('Global Model Explanations')
        st.caption(
            'SHAP estimates how each input feature influences model output across '
            'the bundled EnergyTypeNet dataset.'
        )
        if render_explanation_unavailable(require_shap=True):
            accuracy = {
                name: accuracy_score(y_te, model.predict(X_te_sc if scaled else X_te))
                for name, (model, _, scaled) in models.items()
            }
            default_name = max(accuracy, key=accuracy.get)
            names = list(models)
            selected_name = st.selectbox(
                'Model to explain', names, index=names.index(default_name),
                key='energy_global_explanation_model',
            )
            selected_model, _, selected_scaled = models[selected_name]
            explanation_model = _energy_explanation_model(
                selected_model, selected_scaled, sc
            )
            cache_key = f'energy:global:{selected_name}'
            if st.button('Compute global SHAP explanations', type='primary'):
                with st.spinner('Computing bounded global SHAP explanations...'):
                    try:
                        explainer = _cached_shap_explainer(
                            cache_key,
                            explanation_model,
                            X_tr,
                            ['Energy Consumption', 'Square Footage'],
                        )
                        st.session_state.global_explanation_cache[cache_key] = (
                            explain_dataset_globally(
                                explanation_model,
                                X_tr,
                                ['Energy Consumption', 'Square Footage'],
                                ENERGY_CLASSES,
                                shap_explainer=explainer,
                            )
                        )
                    except Exception as exc:
                        st.warning(f'Global explanation could not be computed: {exc}')
            global_result = st.session_state.global_explanation_cache.get(cache_key)
            if global_result is not None:
                render_global_explanation(global_result, explanation_model)
            else:
                st.info('Choose a model and compute its global explanation.')

    elif page == 'Live Prediction':
        st.title('Live Prediction')

        with st.sidebar:
            st.subheader('Building features')
            energy = st.slider('Energy Consumption (kWh)', 500.0, 10000.0, 4100.0, 50.0)
            sqft = st.slider('Square Footage (ft2)', 500.0, 80000.0, 25000.0, 500.0)

        prediction_frame = pd.DataFrame([{
            'Energy Consumption': energy,
            'Square Footage': sqft,
        }])
        coverage = (
            check_column_coverage(
                ['Energy Consumption', 'Square Footage'], prediction_frame
            )
            if DATA_VALIDATION_AVAILABLE else None
        )
        if coverage is not None and not coverage.passed:
            display_validation_report(coverage, heading='Prediction Input Validation')
        else:
            row_raw = prediction_frame[
                ['Energy Consumption', 'Square Footage']
            ].to_numpy()
            row_sc = sc.transform(row_raw)
            pred_cols = st.columns(len(models))

            for col, (name, (m, _, is_sc)) in zip(pred_cols, models.items()):
                X_in = row_sc if is_sc else row_raw
                pred = int(m.predict(X_in)[0])
                proba = m.predict_proba(X_in)[0]

                col.metric(name, ENERGY_CLASSES[pred])

                fig, ax = plt.subplots(figsize=(2.8, 2))
                ax.barh(ENERGY_CLASSES, proba, color=CLASS_COLORS[:3])
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probability')
                ax.tick_params(labelsize=7)

                plt.tight_layout()
                col.pyplot(fig)
                plt.close(fig)

            st.divider()
            st.subheader('Explain this prediction')
            if render_explanation_unavailable():
                explain_name = st.selectbox(
                    'Model explanation', list(models), key='energy_local_model'
                )
                explain_model, _, explain_scaled = models[explain_name]
                raw_model = _energy_explanation_model(explain_model, explain_scaled, sc)
                if not SHAP_AVAILABLE and not LIME_AVAILABLE:
                    render_explanation_unavailable(require_shap=True)
                elif st.button('Explain prediction', type='primary'):
                    with st.spinner('Computing local SHAP and LIME explanations...'):
                        try:
                            result = _run_local_explanation(
                                cache_key=f'energy:local:{explain_name}',
                                model=raw_model,
                                sample=row_raw,
                                background=X_tr,
                                feature_names=['Energy Consumption', 'Square Footage'],
                                class_names=ENERGY_CLASSES,
                            )
                            st.session_state.energy_current_explanation = result
                            st.session_state.explanation_history.append(
                                {'model': explain_name, 'result': result}
                            )
                            st.session_state.explanation_history = (
                                st.session_state.explanation_history[-10:]
                            )
                        except Exception as exc:
                            st.warning(f'Prediction explanation failed safely: {exc}')
                current = st.session_state.get('energy_current_explanation')
                if current is not None:
                    render_local_explanation(current)
                if st.session_state.explanation_history:
                    with st.expander('Explanation history', expanded=False):
                        labels = [
                            f"{item['model']} · {item['result'].timestamp}"
                            for item in reversed(st.session_state.explanation_history)
                        ]
                        history_label = st.selectbox(
                            'Previous explanation', labels, key='energy_explanation_history'
                        )
                        history_index = labels.index(history_label)
                        render_local_explanation(
                            list(reversed(st.session_state.explanation_history))[
                                history_index
                            ]['result']
                        )

    with st.expander('Export Model Card', expanded=False):
        if not MODEL_CARD_AVAILABLE:
            st.info('Model-card export is unavailable in this installation.')
        elif not models:
            st.info('Models must be trained before a model card can be exported.')
            disabled_columns = st.columns(2)
            disabled_columns[0].button('Download Markdown (.md)', disabled=True)
            disabled_columns[1].button('Download PDF (.pdf)', disabled=True)
        else:
            try:
                comparison_rows = resources.get('comparison')
                if comparison_rows is None:
                    comparison_rows = _compute_energy_comparison(
                        models, X_tr, y_tr, X_te, y_te, X_tr_sc, X_te_sc
                    )
                comparison = pd.DataFrame(comparison_rows)
                card = collect_from_energy_dataset(
                    comparison,
                    {name: values[0] for name, values in models.items()},
                    None,
                )
                _render_model_card_downloads(card, 'energytypenet')
            except Exception as exc:
                st.info(f'Model-card export is not ready yet: {exc}')


# Custom dataset mode

def encode_labels(y_raw: np.ndarray):
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    return y, list(le.classes_)


def render_custom_dashboard():
    st.title('Custom Dataset - Universal Classifier')
    st.markdown(
        'Upload a supported **tabular classification CSV**, pick a categorical '
        'target column and features, '
        'and the full pipeline (5-fold CV, confusion matrices, ROC, PR curves, '
        'decision boundaries via PCA) runs automatically.'
    )

    uploaded = st.file_uploader('Upload a CSV file', type='csv')

    if uploaded is None:
        st.info('Upload a CSV to get started. Example: Iris, Titanic, Wine Quality, or your own data.')

        with st.expander('Supported column types'):
            st.markdown(
                '- **Numeric columns** are used as features directly.\n'
                '- **Categorical / text columns** are one-hot encoded automatically.\n'
                '- **Target column** can be any categorical or integer column.\n'
                '- Missing values are reported before incomplete selected rows are excluded.'
            )
        return

    df_uploaded = pd.read_csv(uploaded)
    dataset_widget_key = hashlib.sha256(uploaded.getvalue()).hexdigest()[:12]
    st.success(f'Loaded {len(df_uploaded):,} rows x {df_uploaded.shape[1]} columns')

    with st.expander('Preview data', expanded=False):
        st.dataframe(df_uploaded.head(20), width='stretch')

    all_cols = list(df_uploaded.columns)
    target_recommendations = recommend_classification_targets(df_uploaded)
    default_target = target_recommendations[0] if target_recommendations else all_cols[-1]

    st.subheader('Column configuration')
    col1, col2 = st.columns(2)

    with col1:
        target_col = st.selectbox(
            'Target column (what to predict)',
            all_cols,
            index=all_cols.index(default_target),
            key=f'custom_target_{dataset_widget_key}',
        )

    remaining = [c for c in all_cols if c != target_col]
    numeric_cols = [
        c for c in remaining
        if pd.api.types.is_numeric_dtype(df_uploaded[c])
    ]

    with col2:
        feature_cols = st.multiselect(
            'Feature columns',
            options=remaining,
            default=numeric_cols[:10],
            key=f'custom_features_{dataset_widget_key}_{target_col}',
        )

    if not feature_cols:
        st.warning('Select at least one feature column.')
        return

    if DATA_VALIDATION_AVAILABLE:
        schema_report = run_all_schema_checks(df_uploaded, target_col)
        if not display_validation_report(
            schema_report,
            heading='Pre-training Data Validation',
        ):
            st.info('Resolve the blocking schema issues before model training can begin.')
            return

    selected_columns = list(dict.fromkeys([*feature_cols, target_col]))
    df_raw = df_uploaded.dropna(subset=selected_columns).copy()
    excluded_rows = len(df_uploaded) - len(df_raw)
    if excluded_rows:
        st.info(
            f'Excluded {excluded_rows:,} row'
            f'{"s" if excluded_rows != 1 else ""} with missing values in the selected '
            f'target or features; {len(df_raw):,} complete rows remain.'
        )
    if df_raw.empty:
        st.error('No complete rows remain for the selected target and features.')
        return

    target_validation = validate_classification_target(df_raw[target_col])
    if not target_validation.valid:
        st.error(target_validation.reason)
        if target_recommendations:
            alternatives = [name for name in target_recommendations if name != target_col]
            if alternatives:
                st.info(
                    'Recommended classification target'
                    + ('s' if len(alternatives) > 1 else '')
                    + ': '
                    + ', '.join(f'`{name}`' for name in alternatives[:5])
                )
        return

    y_raw = df_raw[target_col].values
    y, classes = encode_labels(y_raw)
    n_classes = len(classes)

    # Numeric columns stay as-is; categorical columns become one-hot features.
    feat_parts = []
    feat_names = []

    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df_raw[c]):
            feat_parts.append(df_raw[[c]].values)
            feat_names.append(c)
        else:
            dummies = pd.get_dummies(df_raw[c], prefix=c, drop_first=False)
            feat_parts.append(dummies.values)
            feat_names.extend(dummies.columns.tolist())

    X = np.hstack(feat_parts).astype(float)

    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    st.markdown(
        f'**{len(X_tr):,}** training rows | **{len(X_te):,}** test rows | '
        f'**{X.shape[1]}** features | **{n_classes}** classes: '
        + ', '.join(f'`{c}`' for c in classes)
    )

    cache_key = f'{uploaded.name}_{target_col}_{"_".join(feature_cols)}'
    explanation_signature = (
        f'{dataset_widget_key}:{target_col}:'
        f'{hashlib.sha256(str(feature_cols).encode()).hexdigest()[:12]}'
    )
    if st.session_state.get('custom_explanation_signature') != explanation_signature:
        _clear_explanation_state('custom:')
        st.session_state.custom_current_explanation = None
        st.session_state.custom_live_explanation = None
        st.session_state.custom_explanation_signature = explanation_signature

    @st.cache_resource
    def _fit(key, _X_tr, _y_tr, _n_classes):
        mdls = build_sklearn_models(_n_classes)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            for m in mdls.values():
                m.fit(_X_tr, _y_tr)

        return mdls

    with st.spinner('Training models...'):
        fitted_models = _fit(cache_key, X_tr, y_tr, n_classes)

    if DATA_VALIDATION_AVAILABLE:
        leakage_report = run_all_leakage_checks(
            df_raw[[*feature_cols, target_col]],
            target_col,
            'classification',
        )
        display_validation_report(leakage_report, errors_are_blocking=False)

    tabs = st.tabs([
        'Model Comparison',
        'Confusion Matrices',
        'ROC / AUC',
        'Precision-Recall',
        'Decision Boundary (PCA)',
        'Learning Curves',
        'Live Prediction',
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    with tabs[0]:
        st.subheader('Model Comparison - 5-fold CV + Test Set')
        results = {}

        for name, m in fitted_models.items():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                scores = cross_val_score(
                    m, X_tr, y_tr, cv=skf, scoring='accuracy'
                )
            results[name] = {
                'cv_mean': scores.mean(),
                'cv_std': scores.std(),
                'test_acc': accuracy_score(y_te, m.predict(X_te)),
            }

        res_df = pd.DataFrame([
            {
                'Model': k,
                'CV Mean': v['cv_mean'],
                'CV Std': v['cv_std'],
                'Test Acc': v['test_acc'],
            }
            for k, v in results.items()
        ]).sort_values('CV Mean', ascending=False)

        st.dataframe(
            res_df.style.format({
                'CV Mean': '{:.3f}',
                'CV Std': '{:.3f}',
                'Test Acc': '{:.3f}',
            }),
            width='stretch',
            hide_index=True,
        )

        fig = fig_model_comparison(results)
        st.pyplot(fig)
        plt.close(fig)

    with tabs[1]:
        st.subheader('Confusion Matrices - Test Set')
        cols = st.columns(len(fitted_models))

        for col, (name, m) in zip(cols, fitted_models.items()):
            fig = fig_confusion(m, X_te, y_te, classes, name)
            col.pyplot(fig)
            plt.close(fig)

    with tabs[2]:
        st.subheader('ROC Curves - One-vs-Rest')
        cols = st.columns(len(fitted_models))

        for col, (name, m) in zip(cols, fitted_models.items()):
            fig = fig_roc(m, X_te, y_te, classes, name)

            if fig:
                col.pyplot(fig)
                plt.close(fig)

    with tabs[3]:
        st.subheader('Precision-Recall Curves')
        cols = st.columns(len(fitted_models))

        for col, (name, m) in zip(cols, fitted_models.items()):
            fig = fig_pr(m, X_te, y_te, classes, name)

            if fig:
                col.pyplot(fig)
                plt.close(fig)

        st.subheader('Threshold sweep')
        sweep_model_name = st.selectbox('Model', list(fitted_models))
        sweep_class_name = st.selectbox('Class', classes)
        sweep_class_idx = classes.index(sweep_class_name)
        m_sweep = fitted_models[sweep_model_name]
        proba_sw = m_sweep.predict_proba(X_te)
        y_sw = (y_te == sweep_class_idx).astype(int)
        prec_sw, rec_sw, thr_sw = precision_recall_curve(y_sw, proba_sw[:, sweep_class_idx])

        fig_sw, axs = plt.subplots(1, 2, figsize=(11, 4))
        axs[0].plot(thr_sw, prec_sw[:-1], 'b-', label='Precision')
        axs[0].plot(thr_sw, rec_sw[:-1], 'r-', label='Recall')
        axs[0].axvline(0.5, color='gray', ls='--', alpha=0.6)
        axs[0].set_xlabel('Threshold')
        axs[0].legend()
        axs[0].set_title(f'{sweep_model_name} - {sweep_class_name}')

        sc_sw = axs[1].scatter(rec_sw[:-1], prec_sw[:-1], c=thr_sw, cmap='viridis', s=8)
        plt.colorbar(sc_sw, ax=axs[1], label='Threshold')
        axs[1].set_xlabel('Recall')
        axs[1].set_ylabel('Precision')
        axs[1].set_title('PR curve colored by threshold')

        plt.tight_layout()
        st.pyplot(fig_sw)
        plt.close(fig_sw)

    with tabs[4]:
        st.subheader('Decision Boundary - PCA projection to 2D')
        st.caption(
            'All features are projected to 2 principal components for visualization. '
            'The model is still trained on the full feature set.'
        )

        if X_tr.shape[1] < 2:
            st.info(
                'A 2-D PCA decision boundary requires at least two encoded feature '
                'dimensions. Select another feature to enable this visualization.'
            )
        else:
            sc_full = StandardScaler().fit(X_tr)
            pca = PCA(n_components=2, random_state=42).fit(sc_full.transform(X_tr))
            X_tr_2d = pca.transform(sc_full.transform(X_tr))
            sc_2d = StandardScaler().fit(X_tr_2d)
            X_tr_2d_sc = sc_2d.transform(X_tr_2d)

            # These models are only for the 2-D boundary picture.
            @st.cache_resource
            def _fit_2d(key, _X2, _y):
                m2d = build_sklearn_models(n_classes)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', ConvergenceWarning)
                    for m in m2d.values():
                        m.fit(_X2, _y)

                return m2d

            models_2d = _fit_2d(cache_key + '_2d', X_tr_2d_sc, y_tr)
            cols = st.columns(len(models_2d))

            for col, (name, m) in zip(cols, models_2d.items()):
                fig = fig_decision_boundary_2d(m, X_tr_2d_sc, y_tr, classes, sc_2d, name)
                col.pyplot(fig)
                plt.close(fig)

            st.info(f'PCA explains {pca.explained_variance_ratio_.sum() * 100:.1f} % of variance.')

    with tabs[5]:
        st.subheader('Learning Curves')
        cols = st.columns(len(fitted_models))

        for col, (name, m) in zip(cols, fitted_models.items()):
            with st.spinner(f'{name}...'):
                fig = fig_learning(m, X_tr, y_tr, name)

            col.pyplot(fig)
            plt.close(fig)

    with tabs[6]:
        st.subheader('Live Prediction')
        st.caption('Adjust values below and all models predict in real time.')

        input_vals = {}
        col_inputs = st.columns(min(len(feature_cols), 4))

        for i, c in enumerate(feature_cols):
            with col_inputs[i % len(col_inputs)]:
                if pd.api.types.is_numeric_dtype(df_raw[c]):
                    mn, mx = float(df_raw[c].min()), float(df_raw[c].max())
                    med = float(df_raw[c].median())
                    if np.isclose(mn, mx):
                        input_vals[c] = st.number_input(
                            c,
                            value=med,
                            disabled=True,
                            help='This feature is constant in the uploaded dataset.',
                        )
                    else:
                        input_vals[c] = st.slider(c, mn, mx, med)
                else:
                    opts = sorted(df_raw[c].unique().tolist())
                    input_vals[c] = st.selectbox(c, opts)

        # Rebuild one row with the same one-hot order used during training.
        row_parts = []

        for c in feature_cols:
            if pd.api.types.is_numeric_dtype(df_raw[c]):
                row_parts.append([input_vals[c]])
            else:
                dummies = pd.get_dummies(df_raw[c], prefix=c, drop_first=False)
                one_hot = [
                    1.0 if f == f'{c}_{input_vals[c]}' else 0.0
                    for f in dummies.columns
                ]
                row_parts.extend([[v] for v in one_hot])

        row_flat = [v[0] for v in row_parts]

        if len(row_flat) < X.shape[1]:
            row_flat += [0.0] * (X.shape[1] - len(row_flat))

        row_input = np.array([row_flat[:X.shape[1]]])

        pred_cols = st.columns(len(fitted_models))

        for col, (name, m) in zip(pred_cols, fitted_models.items()):
            pred = int(m.predict(row_input)[0])
            proba = m.predict_proba(row_input)[0]

            col.metric(name, classes[pred])

            fig, ax = plt.subplots(figsize=(2.8, 2))
            ax.barh(classes, proba, color=CLASS_COLORS[:n_classes])
            ax.set_xlim(0, 1)
            ax.set_xlabel('P')
            ax.tick_params(labelsize=7)

            plt.tight_layout()
            col.pyplot(fig)
            plt.close(fig)

        pred_data = {
            name: {
                'class': classes[int(m.predict(row_input)[0])],
                'probabilities': {
                    c: float(p)
                    for c, p in zip(classes, m.predict_proba(row_input)[0])
                },
            }
            for name, m in fitted_models.items()
        }

        import json

        st.download_button(
            'Download prediction JSON',
            json.dumps(pred_data, indent=2),
            file_name='prediction.json',
            mime='application/json',
        )

        st.markdown('#### Explain this prediction')
        if render_explanation_unavailable(require_shap=True, require_lime=True):
            live_explanation_model = st.selectbox(
                'Explanation model',
                list(fitted_models),
                key=f'custom_live_explanation_model_{explanation_signature}',
            )
            if st.button(
                'Generate local SHAP + LIME explanation',
                key=f'custom_live_explanation_button_{explanation_signature}',
            ):
                with st.spinner('Explaining this prediction...'):
                    explanation = _run_local_explanation(
                        cache_key=(
                            f'custom:live:{explanation_signature}:'
                            f'{live_explanation_model}'
                        ),
                        model=fitted_models[live_explanation_model],
                        sample=row_input,
                        background=X_tr,
                        feature_names=feat_names,
                        class_names=classes,
                    )
                st.session_state.custom_live_explanation = explanation
                history = st.session_state.custom_explanation_history
                history.append((live_explanation_model, explanation))
                st.session_state.custom_explanation_history = history[-10:]

            live_explanation = st.session_state.get('custom_live_explanation')
            if live_explanation is not None:
                if (
                    live_explanation.shap_explainer_type
                    and 'kernel' in str(live_explanation.shap_explainer_type).lower()
                ):
                    st.warning(
                        'This model uses Kernel SHAP, so explanation generation can '
                        'take longer than prediction.'
                    )
                render_local_explanation(live_explanation)

    with st.expander('Model Explanations', expanded=False):
        if render_explanation_unavailable(require_shap=True, require_lime=True):
            explanation_model_name = st.selectbox(
                'Model to explain',
                list(fitted_models),
                index=list(fitted_models).index(res_df.iloc[0]['Model']),
                key=f'custom_explanation_model_{explanation_signature}',
            )
            explanation_model = fitted_models[explanation_model_name]
            global_tab, local_tab = st.tabs(['Global SHAP', 'Test-row explanation'])

            with global_tab:
                global_key = (
                    f'custom:global:{explanation_signature}:{explanation_model_name}'
                )
                if st.button(
                    'Generate global SHAP explanation',
                    key=f'custom_global_button_{explanation_signature}',
                ):
                    with st.spinner('Computing global SHAP importance...'):
                        try:
                            explainer = _cached_shap_explainer(
                                global_key, explanation_model, X_tr, feat_names
                            )
                            st.session_state.global_explanation_cache[global_key] = (
                                explain_dataset_globally(
                                    explanation_model,
                                    X_te,
                                    feat_names,
                                    class_names=classes,
                                    shap_explainer=explainer,
                                )
                            )
                        except Exception as exc:
                            st.warning(
                                f'Global explanation could not be computed: {exc}'
                            )
                global_result = st.session_state.global_explanation_cache.get(global_key)
                if global_result is not None:
                    render_global_explanation(global_result, explanation_model)

            with local_tab:
                sample_index = st.selectbox(
                    'Test-row index',
                    range(len(X_te)),
                    key=f'custom_test_row_{explanation_signature}',
                )
                if st.button(
                    'Explain selected test row',
                    key=f'custom_local_button_{explanation_signature}',
                ):
                    with st.spinner('Computing SHAP and LIME explanations...'):
                        st.session_state.custom_current_explanation = (
                            _run_local_explanation(
                                cache_key=(
                                    f'custom:local:{explanation_signature}:'
                                    f'{explanation_model_name}'
                                ),
                                model=explanation_model,
                                sample=X_te[[sample_index]],
                                background=X_tr,
                                feature_names=feat_names,
                                class_names=classes,
                                sample_index=int(sample_index),
                            )
                        )
                current = st.session_state.get('custom_current_explanation')
                if current is not None:
                    if (
                        current.shap_explainer_type
                        and 'kernel' in str(current.shap_explainer_type).lower()
                    ):
                        st.warning(
                            'This model uses Kernel SHAP, which may take longer than '
                            'tree or linear explainers.'
                        )
                    render_local_explanation(current)

            history = st.session_state.custom_explanation_history
            if history:
                with st.expander('Recent local explanations', expanded=False):
                    for position, (model_name, explanation) in enumerate(
                        reversed(history), start=1
                    ):
                        st.markdown(
                            f'**{position}. {model_name}:** '
                            f'{explanation.summary_text()}'
                        )

    with st.expander('Export Model Card', expanded=False):
        if not MODEL_CARD_AVAILABLE:
            st.info('Model-card export is unavailable in this installation.')
        else:
            try:
                card_profile = profile_dataset(df_raw)
                card_prepared = prepare_dataset(
                    df_raw,
                    target_col,
                    feature_cols,
                    task_type='classification',
                )
                card_ranking = rank_features(card_prepared, top_n=None)
                dataset_name = _safe_download_stem(uploaded.name)
                card = build_model_card_data(
                    card_profile,
                    card_prepared,
                    res_df,
                    fitted_models,
                    card_ranking,
                    dataset_name=dataset_name,
                )
                card.dataset_description = (
                    f'Uploaded tabular dataset `{uploaded.name}` analyzed with the '
                    'EnergyTypeNet interactive classification workflow.'
                )
                limitations_key = (
                    f'model_card_limitations_{dataset_widget_key}_{target_col}_'
                    f'{hashlib.sha256(str(feature_cols).encode()).hexdigest()[:8]}'
                )
                if limitations_key not in st.session_state:
                    st.session_state[limitations_key] = card.limitations
                card.limitations = st.text_area(
                    'Limitations and intended use',
                    key=limitations_key,
                    height=150,
                    help='Customize this text for the intended deployment and data source.',
                )

                render_full_model_card(card)
                _render_model_card_downloads(card, dataset_name)
            except Exception as exc:
                st.info(f'Model-card export is not ready yet: {exc}')


def render_dataset_assistant():
    st.title('AI Dataset Assistant')
    st.markdown(
        'Upload a CSV, inspect the dataset, infer possible targets, select features, '
        'train classification or regression baselines, and generate a short report '
        'grounded in computed results.'
    )

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = ChatHistory(max_turns=20)
    if 'last_dataset_signature' not in st.session_state:
        st.session_state.last_dataset_signature = None
    if 'last_profile' not in st.session_state:
        st.session_state.last_profile = None
    if 'last_prepared' not in st.session_state:
        st.session_state.last_prepared = None
    if 'last_results' not in st.session_state:
        st.session_state.last_results = None
    if 'last_fitted_models' not in st.session_state:
        st.session_state.last_fitted_models = None
    if 'last_compact_results' not in st.session_state:
        st.session_state.last_compact_results = None
    if 'last_feature_ranking' not in st.session_state:
        st.session_state.last_feature_ranking = None
    if 'last_training_signature' not in st.session_state:
        st.session_state.last_training_signature = None
    if 'last_dataset_report' not in st.session_state:
        st.session_state.last_dataset_report = None
    if 'last_tool_tables' not in st.session_state:
        st.session_state.last_tool_tables = {}
    if 'pending_question' not in st.session_state:
        st.session_state.pending_question = None
    if 'pending_response' not in st.session_state:
        st.session_state.pending_response = None

    uploaded = st.file_uploader(
        'Upload a CSV file',
        type='csv',
        key='assistant_csv_upload',
    )

    if uploaded is None:
        st.info('Upload any tabular CSV dataset to start the assistant workflow.')
        return

    df = clean_dataframe(pd.read_csv(uploaded))

    if df.empty:
        st.error('The uploaded CSV has no usable rows or columns.')
        return

    dataset_signature = f"{uploaded.name}:{df.shape}:{','.join(df.columns.astype(str))}"

    if st.session_state.last_dataset_signature != dataset_signature:
        st.session_state.chat_history.clear()
        st.session_state.last_profile = None
        st.session_state.last_prepared = None
        st.session_state.last_results = None
        st.session_state.last_fitted_models = None
        st.session_state.last_compact_results = None
        st.session_state.last_feature_ranking = None
        st.session_state.last_training_signature = None
        st.session_state.last_dataset_report = None
        st.session_state.last_tool_tables = {}
        st.session_state.last_dataset_signature = dataset_signature

    profile = profile_dataset(df)
    target_suggestions = suggest_targets(df)
    st.session_state.last_profile = profile

    st.subheader('1. Dataset Profile')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Rows', f"{profile['n_rows']:,}")
    c2.metric('Columns', f"{profile['n_columns']:,}")
    c3.metric('Missing cells', f"{profile['missing_cells']:,}")
    c4.metric('Duplicate rows', f"{profile['duplicate_rows']:,}")

    profile_df = pd.DataFrame(profile['columns'])
    st.dataframe(
        profile_df.style.format({'missing_pct': '{:.1%}'}),
        width='stretch',
        hide_index=True,
    )

    with st.expander('Preview uploaded data', expanded=False):
        st.dataframe(df.head(30), width='stretch')

    st.subheader('2. Target and Feature Suggestions')

    if not target_suggestions:
        st.error('No valid target columns found. Try a dataset with at least one non-constant column.')
        return

    target_df = pd.DataFrame(target_suggestions)
    st.dataframe(
        target_df.style.format({'missing_pct': '{:.1%}', 'score': '{:.1f}'}),
        width='stretch',
        hide_index=True,
    )

    default_target = target_suggestions[0]['column']
    target_col = st.selectbox(
        'Target column',
        list(df.columns),
        index=list(df.columns).index(default_target),
    )

    inferred_task = guess_task_type(df[target_col])
    task_type = st.radio(
        'Prediction task',
        ['classification', 'regression'],
        index=0 if inferred_task != 'regression' else 1,
        horizontal=True,
        key=f'assistant_task_{target_col}',
    )

    if task_type == 'regression' and not pd.api.types.is_numeric_dtype(df[target_col]):
        st.warning(
            f'`{target_col}` is not numeric, so it cannot be used as a regression target. '
            'Switch to classification or choose a numeric target.'
        )
        return

    feature_suggestions = suggest_features(df, target_col)
    usable_features = [
        row['column']
        for row in feature_suggestions
        if row['usable']
    ]

    feature_cols = st.multiselect(
        'Selected candidate feature columns',
        [col for col in df.columns if col != target_col],
        default=usable_features[: min(12, len(usable_features))],
        key=f'assistant_features_{target_col}',
    )
    st.caption(
        'These are usable input columns selected for training. '
        'The assistant ranks which ones look strongest in the next section.'
    )

    with st.expander('Feature suggestion details', expanded=False):
        feature_df = pd.DataFrame(feature_suggestions)
        st.dataframe(
            feature_df.style.format({'missing_pct': '{:.1%}'}),
            width='stretch',
            hide_index=True,
        )

    if not feature_cols:
        st.warning('Select at least one feature column to continue.')
        return

    try:
        prepared = prepare_dataset(df, target_col, feature_cols, task_type)
    except Exception as exc:
        st.error(f'Could not prepare this dataset: {exc}')
        return

    st.markdown(
        f'Prepared **{prepared.task_type}** dataset with '
        f'**{len(prepared.feature_cols)}** selected features '
        f'(**{len(prepared.numeric_cols)}** numeric, '
        f'**{len(prepared.categorical_cols)}** categorical).'
    )
    st.session_state.last_prepared = prepared

    training_signature = (
        dataset_signature,
        target_col,
        task_type,
        tuple(feature_cols),
    )
    if st.session_state.last_training_signature != training_signature:
        if st.session_state.last_training_signature is not None:
            st.session_state.chat_history.clear()
            st.session_state.last_tool_tables = {}
            st.session_state.pending_question = None
            st.session_state.pending_response = None
        st.session_state.last_results = None
        st.session_state.last_fitted_models = None
        st.session_state.last_compact_results = None
        st.session_state.last_dataset_report = None

    if prepared.classes:
        st.caption('Classes: ' + ', '.join(f'`{cls}`' for cls in prepared.classes))

    st.subheader('3. Feature Selection')

    feature_ranking = rank_features(prepared)
    feature_recommendations = recommend_features(prepared)
    st.session_state.last_feature_ranking = feature_ranking

    st.markdown('**Recommended strongest features**')
    st.caption(
        'Recommendations combine mutual information with missingness and uniqueness. '
        'Very high scores can be useful, but they can also indicate target leakage if a '
        'feature directly encodes the label.'
    )
    st.dataframe(
        feature_recommendations.style.format({
            'missing_pct': '{:.1%}',
            'mutual_information': '{:.4f}',
            'quality_score': '{:.3f}',
        }),
        width='stretch',
        hide_index=True,
    )

    strong_features = feature_recommendations[
        feature_recommendations['recommendation'].isin(['Strong', 'Moderate'])
    ]['feature'].tolist()

    if strong_features:
        st.info(
            'Suggested compact feature set: '
            + ', '.join(f'`{feature}`' for feature in strong_features)
        )
    else:
        st.warning(
            'No strong feature set was detected. You can still train baselines, '
            'but the dataset may need better predictors.'
        )

    st.markdown('**Mutual information ranking**')
    st.dataframe(
        feature_ranking.style.format({'mutual_information': '{:.4f}'}),
        width='stretch',
        hide_index=True,
    )

    st.subheader('4. Baseline Model Training')
    st.caption(
        'The assistant trains a broad baseline suite: dummy baseline, linear model, '
        'KNN, SVM/SVR, random forest, gradient boosting, neural network, and XGBoost.'
    )

    compare_compact = bool(strong_features and set(strong_features) != set(feature_cols))

    if compare_compact:
        st.caption(
            'Training will compare the full selected feature set against the suggested '
            'compact feature set, so you can see whether removing weak columns helps.'
        )

    train_clicked = st.button('Train baseline models and compare feature sets', type='primary')

    results = None
    compact_results = None
    compact_prepared = None

    if train_clicked:
        with st.spinner('Training baseline models and computing cross-validation scores...'):
            try:
                results, fitted_models = train_baselines(prepared)
                st.session_state.last_results = results
                st.session_state.last_fitted_models = fitted_models
                st.session_state.last_compact_results = None
                st.session_state.last_training_signature = training_signature

                if compare_compact:
                    compact_prepared = prepare_dataset(
                        df,
                        target_col,
                        strong_features,
                        task_type,
                    )
                    compact_results, _ = train_baselines(compact_prepared)
                    st.session_state.last_compact_results = compact_results
            except Exception as exc:
                st.error(f'Model training failed: {exc}')
                return

        if compact_results is not None:
            st.subheader('Feature Set Comparison')
            full_best = results.iloc[0]
            compact_best = compact_results.iloc[0]

            if prepared.task_type == 'classification':
                comparison_df = pd.DataFrame([
                    {
                        'Feature Set': 'All selected features',
                        'Feature Count': len(prepared.feature_cols),
                        'Best Model': full_best['model'],
                        'Test Accuracy': full_best['test_accuracy'],
                        'Macro F1': full_best['test_f1_macro'],
                    },
                    {
                        'Feature Set': 'Suggested compact features',
                        'Feature Count': len(compact_prepared.feature_cols),
                        'Best Model': compact_best['model'],
                        'Test Accuracy': compact_best['test_accuracy'],
                        'Macro F1': compact_best['test_f1_macro'],
                    },
                ])
                metric_name = 'Test Accuracy'
            else:
                comparison_df = pd.DataFrame([
                    {
                        'Feature Set': 'All selected features',
                        'Feature Count': len(prepared.feature_cols),
                        'Best Model': full_best['model'],
                        'Test R2': full_best['test_r2'],
                        'Test MAE': full_best['test_mae'],
                    },
                    {
                        'Feature Set': 'Suggested compact features',
                        'Feature Count': len(compact_prepared.feature_cols),
                        'Best Model': compact_best['model'],
                        'Test R2': compact_best['test_r2'],
                        'Test MAE': compact_best['test_mae'],
                    },
                ])
                metric_name = 'Test R2'

            numeric_formats = {
                col: '{:.3f}'
                for col in comparison_df.columns
                if col.startswith('Test') or col == 'Macro F1'
            }
            st.dataframe(
                comparison_df.style.format(numeric_formats),
                width='stretch',
                hide_index=True,
            )

            full_score = float(comparison_df.loc[0, metric_name])
            compact_score = float(comparison_df.loc[1, metric_name])
            feature_gap = full_score - compact_score

            if compact_score > full_score + 0.01:
                st.success(
                    'The compact feature set performed better. Removing weak columns '
                    'likely reduced noise.'
                )
            elif full_score >= 0.98 and feature_gap >= 0.15:
                st.warning(
                    'The all-selected feature set is near-perfect while the compact '
                    'feature set is much weaker. Treat this as a diagnostic warning, '
                    'not automatic proof of real-world generalization. Some selected '
                    'columns may reveal the target too directly, or this holdout split '
                    'may be unusually easy.'
                )
            elif full_score > compact_score + 0.01:
                st.warning(
                    'The full feature set performed better. Some weak-looking columns '
                    'may still help when combined with other features.'
                )
            else:
                st.info(
                    'Both feature sets performed similarly. The compact set is simpler, '
                    'while the full set keeps more information.'
                )

        st.subheader('Model Results - All Selected Features')

        if prepared.task_type == 'classification':
            st.dataframe(
                results.style.format({
                    'cv_accuracy': '{:.3f}',
                    'cv_f1_macro': '{:.3f}',
                    'test_accuracy': '{:.3f}',
                    'test_f1_macro': '{:.3f}',
                }),
                width='stretch',
                hide_index=True,
            )
            st.bar_chart(results.set_index('model')['test_accuracy'])
        else:
            st.dataframe(
                results.style.format({
                    'cv_r2': '{:.3f}',
                    'cv_mae': '{:.3f}',
                    'test_r2': '{:.3f}',
                    'test_mae': '{:.3f}',
                }),
                width='stretch',
                hide_index=True,
            )
            st.bar_chart(results.set_index('model')['test_r2'])

        if compact_results is not None:
            st.subheader('Model Results - Suggested Compact Features')

            if prepared.task_type == 'classification':
                st.dataframe(
                    compact_results.style.format({
                        'cv_accuracy': '{:.3f}',
                        'cv_f1_macro': '{:.3f}',
                        'test_accuracy': '{:.3f}',
                        'test_f1_macro': '{:.3f}',
                    }),
                    width='stretch',
                    hide_index=True,
                )
                st.bar_chart(compact_results.set_index('model')['test_accuracy'])
            else:
                st.dataframe(
                    compact_results.style.format({
                        'cv_r2': '{:.3f}',
                        'cv_mae': '{:.3f}',
                        'test_r2': '{:.3f}',
                        'test_mae': '{:.3f}',
                    }),
                    width='stretch',
                    hide_index=True,
                )
                st.bar_chart(compact_results.set_index('model')['test_r2'])

        st.subheader('5. Natural-Language Dataset Report')
        report = generate_dataset_report(
            profile,
            target_suggestions,
            prepared,
            results,
            feature_ranking,
            compact_results,
        )
        st.session_state.last_dataset_report = report
        st.markdown(report)
        st.download_button(
            'Download dataset report',
            report,
            file_name='dataset_report.md',
            mime='text/markdown',
        )

    assistant_ready = (
        st.session_state.last_training_signature == training_signature
        and st.session_state.last_results is not None
        and st.session_state.last_dataset_report is not None
    )

    if not assistant_ready:
        st.subheader('6. Dataset Assistant')
        st.info(
            'Train the baseline models and wait for the dataset report to finish. '
            'The chat will appear when those results are ready.'
        )
        return

    if task_type == 'classification':
        target_validation = validate_classification_target(df[target_col])
        if not target_validation.valid:
            st.warning(target_validation.reason)
            alternatives = recommend_classification_targets(df)
            alternatives = [name for name in alternatives if name != target_col]
            if alternatives:
                st.info(
                    'Recommended classification target'
                    + ('s' if len(alternatives) > 1 else '')
                    + ': '
                    + ', '.join(f'`{name}`' for name in alternatives[:5])
                )
            return

    st.subheader('6. Dataset Assistant')

    provider = st.session_state.get('llm_provider', 'none')

    if provider == 'none':
        st.info(
            'Deterministic answers only. Select an LLM provider in the sidebar '
            'for richer explanations.'
        )
    elif provider == 'ollama':
        st.info(
            'Using Ollama locally. It requires this dashboard to run on the same '
            'computer as Ollama; Streamlit Cloud cannot connect to your localhost. '
            'Answers fall back safely to deterministic dataset statistics if Ollama '
            'is unavailable.'
        )
    elif provider == 'openai':
        st.info(
            f"Using OpenAI ({st.session_state.llm_model}). "
            'Estimated cost is shown in the sidebar.'
        )
    elif provider == 'anthropic':
        st.info(
            f"Using Anthropic ({st.session_state.llm_model}). "
            'Estimated cost is shown in the sidebar.'
        )

    history = st.session_state.chat_history

    with st.container():
        for msg in history.messages:
            with st.chat_message(msg.role):
                st.write(msg.content)

                if msg.role == 'assistant':
                    st.caption(
                        f"Source: {msg.source} | "
                        f"{msg.question_type.replace('_', ' ')} | "
                        f"{msg.timestamp[:10]}"
                    )

    if st.session_state.last_tool_tables:
        with st.expander('Latest computed diagnostic tables', expanded=False):
            for table_name, table in st.session_state.last_tool_tables.items():
                st.markdown(f'**{table_name.replace("_", " ").title()}**')
                st.dataframe(table, width='stretch', hide_index=True)

    if history.messages and st.session_state.pending_response is None:
        last_type = history.messages[-1].question_type
        suggestions = suggest_next_questions(
            last_type,
            st.session_state.last_profile,
            st.session_state.last_prepared,
            st.session_state.last_results,
        )
        st.caption('Suggested follow-ups:')
        cols = st.columns(len(suggestions))

        for i, suggestion in enumerate(suggestions):
            if cols[i].button(suggestion, key=f'suggest_{i}_{len(history.messages)}'):
                st.session_state.pending_question = suggestion
                st.rerun()

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button('Clear chat', type='secondary'):
            st.session_state.chat_history.clear()
            st.session_state.pending_question = None
            st.session_state.pending_dataset_question = None
            st.rerun()

    with col2:
        if history.messages:
            history_json = json.dumps(history.to_dict_list(), indent=2)
            st.download_button(
                label='Export chat',
                data=history_json,
                file_name='dataset_chat.json',
                mime='application/json',
            )

    with col3:
        st.caption(
            f'{len(history.messages)} messages | '
            f'{len(history.user_questions())} questions'
        )

    with st.expander('Export Model Card', expanded=False):
        if not MODEL_CARD_AVAILABLE:
            st.info('Model-card export is unavailable in this installation.')
        else:
            try:
                dataset_name = _safe_download_stem(uploaded.name)
                card = build_model_card_data(
                    st.session_state.last_profile,
                    st.session_state.last_prepared,
                    st.session_state.last_results,
                    st.session_state.last_fitted_models,
                    st.session_state.last_feature_ranking,
                    dataset_name=dataset_name,
                )
                card.dataset_description = (
                    f'Uploaded tabular dataset `{uploaded.name}` analyzed with the '
                    f'EnergyTypeNet AI Dataset Assistant as a {task_type} task.'
                )

                signature_hash = hashlib.sha256(
                    repr(training_signature).encode('utf-8')
                ).hexdigest()[:12]
                limitations_key = f'assistant_model_card_limitations_{signature_hash}'
                if limitations_key not in st.session_state:
                    st.session_state[limitations_key] = card.limitations
                card.limitations = st.text_area(
                    'Limitations and intended use',
                    key=limitations_key,
                    height=150,
                    help='Customize this text for the intended deployment and data source.',
                )

                include_chat = st.checkbox(
                    'Include chat explanations',
                    key=f'assistant_model_card_chat_{signature_hash}',
                )
                if include_chat:
                    notable = select_notable_exchanges(history)
                    card = add_chat_explanations(card, notable)
                    if not card.chat_explanations:
                        st.caption(
                            'No grounded chat explanations are available to include. '
                            'Ask the Dataset Assistant a question first.'
                        )

                render_full_model_card(card)
                _render_model_card_downloads(card, f'{dataset_name}_assistant')
            except Exception as exc:
                st.info(f'Model-card export is not ready yet: {exc}')

    user_input = st.chat_input('Ask something about your dataset...')

    question = None
    if user_input:
        question = user_input.strip()
    elif st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question:
        profile_saved = st.session_state.last_profile
        prepared_saved = st.session_state.last_prepared

        if profile_saved is None:
            st.warning('Please upload a dataset and run analysis first.')
            return

        q_type = classify_question(question, profile_saved, prepared_saved)
        st.session_state.chat_history.add(ChatMessage(
            role='user',
            content=question,
            source='user',
            timestamp=utc_timestamp(),
            question_type=q_type,
            grounded=False,
        ))
        st.session_state.pending_response = {
            'question': question,
            'question_type': q_type,
        }
        st.rerun()

    if st.session_state.pending_response:
        pending = st.session_state.pending_response
        question = pending['question']
        q_type = pending['question_type']

        profile_saved = st.session_state.last_profile
        prepared_saved = st.session_state.last_prepared
        results_saved = st.session_state.last_results
        compact_results_saved = st.session_state.last_compact_results
        ranking_saved = st.session_state.last_feature_ranking

        with st.chat_message('assistant'):
            response_placeholder = st.empty()
            response_placeholder.markdown('_Thinking through the dataset..._')

            previous_assistant = st.session_state.chat_history.last_assistant_message()
            tool_result = run_dataset_computation(
                question,
                profile_saved,
                prepared_saved,
                results_saved,
                ranking_saved,
                compact_results_saved,
                previous_answer=previous_assistant.content if previous_assistant else None,
            )

            if tool_result is not None:
                deterministic = tool_result.answer
                source = tool_result.source
                st.session_state.last_tool_tables = tool_result.tables
            else:
                deterministic = handle_follow_up(
                    question,
                    st.session_state.chat_history,
                    profile_saved,
                    prepared_saved,
                    results_saved,
                    ranking_saved,
                )
                source = 'deterministic'

            answer = deterministic
            provider = st.session_state.get('llm_provider', 'none')

            if provider != 'none':
                prompt = build_contextualized_prompt(
                    question,
                    st.session_state.chat_history,
                    profile_saved,
                    prepared_saved,
                    results_saved,
                    ranking_saved,
                    q_type,
                    tool_result=tool_result.answer if tool_result else None,
                )
                time.sleep(0.25)
                full_response = ''

                try:
                    gen, actual_source = stream_with_fallback(
                        prompt,
                        provider=provider,
                        model=st.session_state.get('llm_model') or DEFAULT_OLLAMA_MODEL,
                        fallback_answer=deterministic,
                        api_key=st.session_state.get('llm_api_key'),
                    )

                    for token in gen:
                        full_response += token
                        response_placeholder.markdown(full_response + '...')

                    response_placeholder.markdown(full_response)
                    answer = full_response
                    source = actual_source

                    if actual_source != 'deterministic':
                        st.session_state.usage_tracker.record(
                            actual_source,
                            st.session_state.get('llm_model') or DEFAULT_OLLAMA_MODEL,
                            prompt,
                            full_response,
                        )
                except Exception as exc:
                    source = 'deterministic'
                    st.caption(
                        f'LLM streaming failed ({exc}), so the deterministic '
                        'assistant answered instead.'
                    )
                    response_placeholder.markdown(answer)
            else:
                response_placeholder.markdown(answer)

            st.caption(
                f"Source: {source} | "
                f"{q_type.replace('_', ' ')} | "
                f"{utc_timestamp()[:10]}"
            )

            if tool_result is not None and tool_result.tables:
                with st.expander('Computed diagnostic tables', expanded=False):
                    for table_name, table in tool_result.tables.items():
                        st.markdown(f'**{table_name.replace("_", " ").title()}**')
                        st.dataframe(table, width='stretch', hide_index=True)

        st.session_state.chat_history.add(ChatMessage(
            role='assistant',
            content=answer,
            source=source,
            timestamp=utc_timestamp(),
            question_type=q_type,
            grounded=True,
        ))
        st.session_state.pending_response = None
        st.rerun()


# Sidebar and routing

st.sidebar.title('EnergyTypeNet')
mode = st.sidebar.radio(
    'Mode',
    ['EnergyTypeNet (pre-loaded)', 'Custom Dataset', 'AI Dataset Assistant'],
)

if mode == 'AI Dataset Assistant':
    st.sidebar.markdown('---')
    st.sidebar.subheader('LLM Provider')

    provider_labels = {
        'none': 'None (deterministic only)',
        'ollama': 'Ollama (local)',
        'openai': 'OpenAI (API key required)',
        'anthropic': 'Anthropic (API key required)',
    }
    provider_options = ['none'] + [provider for provider in PROVIDERS if provider != 'none']
    current_provider = st.session_state.get('llm_provider', 'none')

    if current_provider not in provider_options:
        current_provider = 'none'

    provider = st.sidebar.selectbox(
        'Provider',
        options=provider_options,
        format_func=lambda item: provider_labels[item],
        index=provider_options.index(current_provider),
    )
    st.session_state.llm_provider = provider

    if provider in PROVIDER_MODELS:
        model_options = PROVIDER_MODELS[provider]
        current_model = st.session_state.get('llm_model')
        model_index = model_options.index(current_model) if current_model in model_options else 0
        st.session_state.llm_model = st.sidebar.selectbox(
            'Model',
            model_options,
            index=model_index,
        )
    else:
        st.session_state.llm_model = None

    st.session_state.llm_api_key = None

    if provider in ('openai', 'anthropic'):
        auto_key = load_api_key(provider)

        if auto_key:
            st.sidebar.success(f'{provider.title()} key loaded from environment/secrets')
            st.session_state.llm_api_key = auto_key
        else:
            manual_key = st.sidebar.text_input(
                f'{provider.title()} API key',
                type='password',
                placeholder='sk-...' if provider == 'openai' else 'sk-ant-...',
            )

            if manual_key:
                st.session_state.llm_api_key = manual_key

    if provider == 'ollama':
        st.sidebar.caption(
            'Local use only: Ollama must run on localhost:11434 on the same computer '
            'as this dashboard. Streamlit Cloud cannot access your local Ollama. '
            'Start it with `ollama run llama3.1`.'
        )

    premium_models = {'gpt-4o', 'claude-sonnet-4-6'}
    selected_model = st.session_state.get('llm_model')

    if selected_model in premium_models:
        output_rate = COST_PER_1K_TOKENS[selected_model]['output']
        st.sidebar.warning(
            f'{selected_model} is a premium model. Each question costs approximately '
            f'${output_rate * 0.5:.4f}-${output_rate * 2:.4f}. '
            'Consider using the smaller model first.'
        )

    tracker = st.session_state.usage_tracker

    if tracker.total_tokens() > 0:
        st.sidebar.caption(f'Session usage: {tracker.summary()}')

if mode == 'EnergyTypeNet (pre-loaded)':
    st.sidebar.markdown('---')
    page = st.sidebar.radio(
        'Navigate',
        ['Overview', 'EDA', 'Model Comparison', 'Decision Boundaries',
         'Confusion Matrices', 'ROC / AUC', 'Precision-Recall',
         'Learning Curves', 'Explanations', 'Live Prediction'],
    )
    render_energy_dashboard(page)
else:
    if mode == 'Custom Dataset':
        render_custom_dashboard()
    else:
        render_dataset_assistant()
