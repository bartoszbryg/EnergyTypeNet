"""Train and save the production EnergyTypeNet model."""

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data import CLASSES, load_features


def build_models(random_state: int = 42) -> dict:
    """Build all candidate models used by the training script."""
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=10, max_iter=1000, random_state=random_state),
    )

    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(40, 20),
            activation='tanh',
            alpha=1e-5,
            max_iter=3000,
            early_stopping=True,
            random_state=random_state,
        ),
    )

    xgb = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        max_depth=5,
        learning_rate=0.05,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=1.0,
        gamma=0,
        random_state=random_state,
        verbosity=0,
    )

    voting = VotingClassifier(
        estimators=[('lr', lr), ('mlp', mlp), ('xgb', xgb)],
        voting='soft',
    )

    stacking = StackingClassifier(
        estimators=[('lr', lr), ('mlp', mlp), ('xgb', xgb)],
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method='predict_proba',
        cv=5,
        n_jobs=1,
    )

    return {
        'logistic_regression': lr,
        'mlp': mlp,
        'xgboost': xgb,
        'soft_voting': voting,
        'stacking': stacking,
    }


def evaluate_models(models: dict, X: np.ndarray, y: np.ndarray) -> dict:
    """Evaluate candidate models with stratified cross-validation."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=1)

        results[name] = {
            'cv_mean': float(scores.mean()),
            'cv_std': float(scores.std()),
            'cv_scores': [float(score) for score in scores],
        }

    return results


def train_best_model(train_path: str, test_path: str, feature_set: str) -> dict:
    """Train all candidates and return the best fitted model artifact."""
    X_train, y_train = load_features(train_path, feature_set)
    X_test, y_test = load_features(test_path, feature_set)

    models = build_models()
    results = evaluate_models(models, X_train, y_train)

    best_name = max(results, key=lambda name: results[name]['cv_mean'])
    best_model = models[best_name]
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    results[best_name]['test_accuracy'] = float(accuracy_score(y_test, y_pred))
    results[best_name]['classification_report'] = classification_report(
        y_test,
        y_pred,
        target_names=CLASSES,
        output_dict=True,
    )

    return {
        'best_name': best_name,
        'best_model': best_model,
        'results': results,
        'feature_set': feature_set,
        'classes': CLASSES,
    }


def log_to_mlflow(output: dict, model_path: Path, feature_set: str) -> None:
    """Log every candidate's CV scores plus the best model artifact to MLflow."""
    mlflow.set_experiment('EnergyTypeNet')
    with mlflow.start_run(run_name=f'train-{feature_set}'):
        mlflow.log_param('feature_set', feature_set)
        mlflow.log_param('best_model', output['best_name'])

        for model_name, metrics in output['results'].items():
            mlflow.log_metric(f'{model_name}_cv_mean', metrics['cv_mean'])
            mlflow.log_metric(f'{model_name}_cv_std', metrics['cv_std'])

        if 'test_accuracy' in output['results'][output['best_name']]:
            mlflow.log_metric(
                'test_accuracy',
                output['results'][output['best_name']]['test_accuracy'],
            )

        mlflow.sklearn.log_model(
            output['best_model'],
            artifact_path='model',
            registered_model_name='EnergyTypeNet',
        )
        mlflow.log_artifact(str(model_path), artifact_path='joblib')


def main():
    parser = argparse.ArgumentParser(description='Train and save EnergyTypeNet.')
    parser.add_argument('--train-path', default='data/train_energy_data.csv')
    parser.add_argument('--test-path', default='data/test_energy_data.csv')
    parser.add_argument('--feature-set', default='core', choices=['core', 'extended', 'all'])
    parser.add_argument('--model-out', default='artifacts/model.joblib')
    parser.add_argument('--metrics-out', default='artifacts/metrics.json')
    parser.add_argument('--no-mlflow', action='store_true',
                        help='Skip MLflow logging (useful for CI / offline runs)')

    args = parser.parse_args()

    output = train_best_model(args.train_path, args.test_path, args.feature_set)

    model_path = Path(args.model_out)
    metrics_path = Path(args.metrics_out)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        'model': output['best_model'],
        'feature_set': output['feature_set'],
        'classes': output['classes'],
        'best_name': output['best_name'],
    }

    joblib.dump(artifact, model_path)
    metrics_path.write_text(json.dumps(output['results'], indent=2), encoding='utf-8')

    if not args.no_mlflow:
        log_to_mlflow(output, model_path, args.feature_set)
        print('MLflow run logged. View with: mlflow ui')

    print(f'Saved model:   {model_path}')
    print(f'Saved metrics: {metrics_path}')
    print(f"Best model:    {output['best_name']}")


if __name__ == '__main__':
    main()
