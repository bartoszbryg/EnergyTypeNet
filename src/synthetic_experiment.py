"""Synthetic experiment to test sample-size vs. feature-overlap limits."""

import argparse
import json
from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_synthetic_models(random_state: int = 42) -> dict:
    """Create the models used in the synthetic separability experiment."""
    logistic_regression = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )

    xgboost = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        max_depth=4,
        learning_rate=0.05,
        n_estimators=100,
        random_state=random_state,
        verbosity=0,
    )

    return {
        'logistic_regression': logistic_regression,
        'xgboost': xgboost,
    }


def run_experiment(random_state: int = 42) -> list:
    """Run synthetic datasets across sample sizes and class separability levels."""
    rows = []
    sample_sizes = [300, 1000, 3000]
    class_separations = [0.4, 0.7, 1.0, 1.5, 2.0]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for n_samples in sample_sizes:
        for class_sep in class_separations:
            X, y = make_classification(
                n_samples=n_samples,
                n_features=6,
                n_informative=2,
                n_redundant=2,
                n_classes=3,
                n_clusters_per_class=1,
                class_sep=class_sep,
                random_state=random_state,
            )

            models = build_synthetic_models(random_state=random_state)

            for name, model in models.items():
                scores = cross_val_score(
                    model,
                    X,
                    y,
                    cv=cv,
                    scoring='accuracy',
                    n_jobs=1,
                )

                rows.append({
                    'model': name,
                    'n_samples': n_samples,
                    'class_sep': class_sep,
                    'cv_mean': float(scores.mean()),
                    'cv_std': float(scores.std()),
                })

    return rows


def main():
    parser = argparse.ArgumentParser(description='Run synthetic separability experiment.')
    parser.add_argument('--output', default='results/synthetic_experiment.json')

    args = parser.parse_args()

    rows = run_experiment()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding='utf-8')

    best = max(rows, key=lambda row: row['cv_mean'])

    print(f'Saved synthetic experiment results: {output_path}')
    print(
        f"Best run: {best['model']} "
        f"n={best['n_samples']} "
        f"sep={best['class_sep']} "
        f"acc={best['cv_mean']:.3f}"
    )


if __name__ == '__main__':
    main()
