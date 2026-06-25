"""Data loading and feature engineering utilities."""

import numpy as np
import pandas as pd


LABEL_MAP = {
    'Residential': 0,
    'Commercial': 1,
    'Industrial': 2,
}

CLASSES = [
    'Residential',
    'Commercial',
    'Industrial',
]

FEATURE_COLS = {
    'core': [
        'Energy Consumption',
        'Square Footage',
    ],
    'extended': [
        'Energy Consumption',
        'Square Footage',
        'Number of Occupants',
        'Appliances Used',
    ],
    'all': [
        'Energy Consumption',
        'Square Footage',
        'Number of Occupants',
        'Appliances Used',
        'Average Temperature',
    ],
}


def load_raw(filepath: str) -> pd.DataFrame:
    """Load the raw CSV data and remove empty rows."""
    return pd.read_csv(filepath).dropna()


def load_features(filepath: str, feature_set: str = 'core'):
    """Return feature and target arrays for the selected feature set.

    Parameters
    ----------
    filepath : path to CSV
    feature_set : one of 'core', 'extended', 'all'
    """
    df = load_raw(filepath)

    y = df['Building Type'].map(LABEL_MAP).values
    X = df[FEATURE_COLS[feature_set]].values.astype(float)

    return X, y


def make_engineered_features(df: pd.DataFrame):
    """Create original and derived numeric features from a raw dataframe.

    Returns
    -------
    X : ndarray of shape (n, n_features)
    feat_names : list of feature name strings
    """
    feat = pd.DataFrame()

    feat['energy_consumption'] = df['Energy Consumption']
    feat['square_footage'] = df['Square Footage']
    feat['num_occupants'] = df['Number of Occupants']
    feat['appliances_used'] = df['Appliances Used']
    feat['avg_temperature'] = df['Average Temperature']
    feat['is_weekend'] = (df['Day of Week'] == 'Weekend').astype(float)

    sqft_safe = df['Square Footage'].clip(lower=1)
    occ_safe = df['Number of Occupants'].clip(lower=1)

    feat['energy_per_sqft'] = df['Energy Consumption'] / sqft_safe
    feat['occupancy_density'] = df['Number of Occupants'] / sqft_safe
    feat['appliance_per_occ'] = df['Appliances Used'] / occ_safe

    return feat.values.astype(float), list(feat.columns)
