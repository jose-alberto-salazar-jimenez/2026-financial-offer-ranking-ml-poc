"""Test feature pipeline: preprocessor shape and columns."""
import numpy as np
import pandas as pd
import pytest

from src.pipelines.features import build_preprocessor, transform, get_feature_columns


@pytest.fixture
def tiny_df():
    """Minimal DataFrame with required columns."""
    num_cols, cat_cols = get_feature_columns()
    n = 5
    data = {}
    for c in num_cols:
        data[c] = np.random.RandomState(42).randint(0, 100, n) if c != "balance" else np.random.RandomState(42).randint(-100, 1000, n)
    for c in cat_cols:
        data[c] = ["unknown"] * n
    return pd.DataFrame(data)


def test_build_preprocessor(tiny_df):
    preprocessor, feature_names = build_preprocessor(tiny_df)
    assert len(feature_names) >= 1
    X = transform(preprocessor, tiny_df)
    assert X.shape[0] == len(tiny_df)
    assert X.shape[1] == len(feature_names)


def test_preprocessor_no_error(tiny_df):
    preprocessor, _ = build_preprocessor(tiny_df)
    out = transform(preprocessor, tiny_df)
    assert out is not None
    assert not np.any(np.isnan(out))
