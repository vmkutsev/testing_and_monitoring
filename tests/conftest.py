"""
Shared fixtures for the test suite.
"""
import os
import pytest
from dotenv import load_dotenv

load_dotenv()
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd


MOCK_RUN_ID = 'test-run-id-123'
MOCK_FEATURES = ['race', 'sex', 'native.country', 'education', 'occupation', 'capital.gain']


def make_mock_model(probability: float = 0.8) -> MagicMock:
    """Return a mock sklearn pipeline that mimics predict_proba."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[1 - probability, probability]])
    model.feature_names_in_ = MOCK_FEATURES
    return model


@pytest.fixture()
def mock_model():
    return make_mock_model()


@pytest.fixture()
def valid_predict_payload() -> dict:
    return {
        'race': 'White',
        'sex': 'Male',
        'native.country': 'United-States',
        'education': 'Bachelors',
        'occupation': 'Prof-specialty',
        'capital.gain': 5000,
    }


@pytest.fixture()
def app_client(mock_model):
    """TestClient with a mocked model — no MLflow calls."""
    from ml_service.model import ModelData

    with patch('ml_service.app.MODEL') as mock_global_model, \
         patch('ml_service.app.init_monitoring'), \
         patch('ml_service.app.drift_monitoring_cron'):

        mock_global_model.get.return_value = ModelData(
            model=mock_model, run_id=MOCK_RUN_ID
        )
        mock_global_model.features = MOCK_FEATURES

        from ml_service.app import app
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client
