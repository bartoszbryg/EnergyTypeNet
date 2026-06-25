"""Reusable evaluation and plotting utilities for EnergyTypeNet."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler


CMAP_LIGHT = ListedColormap(['#FFCCCC', '#CCFFCC', '#CCCCFF'])
CMAP_BOLD = ListedColormap(['#CC0000', '#006600', '#0000CC'])

CLASSES = [
    'Residential',
    'Commercial',
    'Industrial',
]


def make_skf(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    """Create the stratified cross-validation splitter used across notebooks."""
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def cross_validate_custom(
    model_cls,
    kwargs: dict,
    X: np.ndarray,
    y: np.ndarray,
    skf=None,
    needs_scaling: bool = True,
) -> np.ndarray:
    """Run k-fold cross-validation for a custom model class.

    Parameters
    ----------
    model_cls : class to instantiate each fold
    kwargs : constructor arguments
    X, y : full dataset
    skf : StratifiedKFold instance
    needs_scaling : whether to fit a StandardScaler inside each fold
    """
    if skf is None:
        skf = make_skf()

    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if needs_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        model = model_cls(**kwargs)
        model.fit(X_train, y_train)

        scores.append(accuracy_score(y_val, model.predict(X_val)))

    return np.array(scores)


def plot_decision_boundaries(
    named_models: list,
    X_sc: np.ndarray,
    y: np.ndarray,
    h: float = 0.06,
    figsize=(16, 10),
) -> plt.Figure:
    """Plot 2-D decision boundaries for a list of fitted models."""
    x0_min, x0_max = X_sc[:, 0].min() - 0.5, X_sc[:, 0].max() + 0.5
    x1_min, x1_max = X_sc[:, 1].min() - 0.5, X_sc[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x0_min, x0_max, h),
        np.arange(x1_min, x1_max, h),
    )

    n_models = len(named_models)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten() if n_models > 1 else [axes]

    for ax, (title, model, grid) in zip(axes_flat, named_models):
        Z = model.predict(grid).reshape(xx.shape)

        ax.pcolormesh(xx, yy, Z, cmap=CMAP_LIGHT, alpha=0.65, shading='auto')
        ax.scatter(
            X_sc[:, 0],
            X_sc[:, 1],
            c=y,
            cmap=CMAP_BOLD,
            edgecolors='k',
            s=15,
            alpha=0.6,
            linewidths=0.3,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Energy Consumption (scaled)')
        ax.set_ylabel('Square Footage (scaled)')

    for ax in axes_flat[n_models:]:
        ax.set_visible(False)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            markerfacecolor=color,
            markersize=9,
            label=label,
        )
        for color, label in zip(['#CC0000', '#006600', '#0000CC'], CLASSES)
    ]

    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=3,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(
        'Decision Boundaries - Energy Consumption x Square Footage (scaled)',
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()

    return fig


def plot_confusion_matrices(
    named_models: list,
    y_test: np.ndarray,
    figsize=(15, 9),
) -> plt.Figure:
    """Plot a grid of confusion matrices."""
    n_models = len(named_models)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten() if n_models > 1 else [axes]

    for ax, (title, model, X) in zip(axes_flat, named_models):
        y_pred = model.predict(X)
        acc = accuracy_score(y_test, y_pred)

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=CLASSES,
            ax=ax,
            colorbar=False,
            cmap='Blues',
        )
        ax.set_title(f'{title}\nacc={acc:.2f}', fontsize=10)

    for ax in axes_flat[n_models:]:
        ax.set_visible(False)

    fig.suptitle('Confusion Matrices - All Models (Test Set)', fontsize=12, y=1.01)
    plt.tight_layout()

    return fig


def plot_learning_curves(
    named_estimators: list,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    train_sizes=None,
    figsize=(16, 4),
) -> plt.Figure:
    """Plot training and cross-validation accuracy as training size grows."""
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    n_models = len(named_estimators)
    fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=True)

    if n_models == 1:
        axes = [axes]

    for ax, (title, estimator) in zip(axes, named_estimators):
        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator,
            X,
            y,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            train_sizes=train_sizes,
            scoring='accuracy',
            n_jobs=1,
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        ax.plot(train_sizes_abs, train_mean, 'o-', color='steelblue', label='Train')
        ax.fill_between(
            train_sizes_abs,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.15,
            color='steelblue',
        )

        ax.plot(train_sizes_abs, val_mean, 's-', color='darkorange', label='CV val')
        ax.fill_between(
            train_sizes_abs,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.15,
            color='darkorange',
        )

        ax.axhline(1 / 3, color='red', linestyle='--', alpha=0.5, label='Random baseline')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Training examples')
        ax.set_ylabel('Accuracy')
        ax.legend(fontsize=8)
        ax.set_ylim(0.2, 1.05)

    fig.suptitle('Learning Curves', fontsize=13, y=1.02)
    plt.tight_layout()

    return fig
