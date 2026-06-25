"""Command-line inference for EnergyTypeNet."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.data import FEATURE_COLS


def load_artifact(model_path: str = 'artifacts/model.joblib') -> dict:
    """Load a saved model artifact from disk."""
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f'Model artifact not found at {path}. Run: python -m src.train'
        )

    return joblib.load(path)


def predict_dataframe(df: pd.DataFrame, artifact: dict) -> list:
    """Predict class labels and probabilities for a dataframe."""
    feature_set = artifact['feature_set']
    model = artifact['model']
    classes = artifact['classes']

    X = df[FEATURE_COLS[feature_set]].values.astype(float)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    rows = []

    for pred, proba in zip(predictions, probabilities):
        rows.append({
            'class': classes[int(pred)],
            'probabilities': {
                classes[i]: float(probability)
                for i, probability in enumerate(proba)
            },
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description='Predict building type from a CSV file.')
    parser.add_argument('--input', required=True, help='CSV file containing one or more rows.')
    parser.add_argument('--model', default='artifacts/model.joblib')

    args = parser.parse_args()

    artifact = load_artifact(args.model)
    df = pd.read_csv(args.input)
    predictions = predict_dataframe(df, artifact)

    print(json.dumps(predictions, indent=2))


if __name__ == '__main__':
    main()
