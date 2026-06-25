from src.data import FEATURE_COLS, load_features, make_engineered_features, load_raw


def test_load_features_core_shape():
    X, y = load_features('data/train_energy_data.csv', 'core')

    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == len(FEATURE_COLS['core'])


def test_engineered_features_have_expected_columns():
    df = load_raw('data/train_energy_data.csv').head(5)
    X, names = make_engineered_features(df)

    assert X.shape == (5, len(names))
    assert 'energy_per_sqft' in names
    assert 'occupancy_density' in names
    assert 'appliance_per_occ' in names
