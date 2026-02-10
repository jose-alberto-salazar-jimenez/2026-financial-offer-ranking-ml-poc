"""Test prediction API: shape and score in [0,1]."""
import numpy as np
import pandas as pd
import pytest

from src.serving.predict import predict, predict_batch


@pytest.fixture
def one_row():
    return {
        "age": 40,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "balance": 1000,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "duration": 300,
        "campaign": 2,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown",
    }


def test_predict_requires_model(one_row):
    """Without a trained model, predict raises FileNotFoundError or similar."""
    # Clear cache so we load from disk
    import src.serving.predict as mod
    mod._model_cache = None
    mod._preprocessor_cache = None
    mod._feature_names_cache = None
    try:
        score = predict(one_row)
        assert isinstance(score, (float, list))
        if isinstance(score, float):
            assert 0 <= score <= 1
        else:
            for s in score:
                assert 0 <= s <= 1
    except FileNotFoundError:
        pytest.skip("Model not found (run make train first)")


def test_predict_batch_requires_model(one_row):
    df = pd.DataFrame([one_row, one_row])
    import src.serving.predict as mod
    mod._model_cache = None
    mod._preprocessor_cache = None
    mod._feature_names_cache = None
    try:
        scores = predict_batch(df)
        assert scores.shape == (2,)
        for s in scores:
            assert 0 <= s <= 1
    except FileNotFoundError:
        pytest.skip("Model not found (run make train first)")
