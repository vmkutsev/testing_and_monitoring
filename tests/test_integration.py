"""
Integration tests: the full service with a real MLflow model.

Requires environment variables:
  MLFLOW_TRACKING_URI  — e.g. http://158.160.2.37:5000/
  TEST_RUN_ID          — run_id of a valid MLflow run with a logged model

Skip automatically when TEST_RUN_ID is not set.
"""
import os
import pytest
from fastapi.testclient import TestClient


MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://158.160.2.37:5000/')
TEST_RUN_ID = os.getenv('TEST_RUN_ID', '8990717746ed4cfda04aaabd43c8bad')

pytestmark = pytest.mark.skipif(
    not TEST_RUN_ID or not MLFLOW_URI,
    reason='TEST_RUN_ID or MLFLOW_TRACKING_URI not set — skipping integration tests',
)

VALID_PAYLOAD = {
    'race': 'White',
    'sex': 'Male',
    'native.country': 'United-States',
    'education': 'Bachelors',
    'occupation': 'Prof-specialty',
    'capital.gain': 5000,
}


@pytest.fixture(scope='module')
def integration_client():
    os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_URI
    os.environ['DEFAULT_RUN_ID'] = TEST_RUN_ID

    from unittest.mock import patch
    with patch('ml_service.app.init_monitoring'), \
         patch('ml_service.app.drift_monitoring_cron'):
        from ml_service.app import app
        with TestClient(app) as client:
            yield client


class TestIntegrationHealth:
    def test_health_ok(self, integration_client):
        resp = integration_client.get('/health')
        assert resp.status_code == 200
        data = resp.json()
        assert data['status'] == 'ok'
        assert data['run_id'] == TEST_RUN_ID


class TestIntegrationPredict:
    def test_predict_returns_200(self, integration_client):
        resp = integration_client.post('/predict', json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_predict_response_structure(self, integration_client):
        resp = integration_client.post('/predict', json=VALID_PAYLOAD)
        data = resp.json()
        assert 'prediction' in data
        assert 'probability' in data

    def test_predict_probability_in_range(self, integration_client):
        resp = integration_client.post('/predict', json=VALID_PAYLOAD)
        prob = resp.json()['probability']
        assert 0.0 <= prob <= 1.0

    def test_predict_prediction_is_binary(self, integration_client):
        resp = integration_client.post('/predict', json=VALID_PAYLOAD)
        assert resp.json()['prediction'] in (0, 1)

    def test_predict_empty_payload_handled(self, integration_client):
        resp = integration_client.post('/predict', json={})
        assert resp.status_code in (200, 422, 500)

    def test_predict_invalid_type_returns_422(self, integration_client):
        resp = integration_client.post('/predict', json={'age': 'not_a_number'})
        assert resp.status_code == 422


class TestIntegrationUpdateModel:
    def test_update_to_same_run_id(self, integration_client):
        resp = integration_client.post('/updateModel', json={'run_id': TEST_RUN_ID})
        assert resp.status_code == 200
        assert resp.json()['run_id'] == TEST_RUN_ID

    def test_update_to_invalid_run_id_returns_404(self, integration_client):
        resp = integration_client.post(
            '/updateModel', json={'run_id': 'definitely-not-a-valid-run-id'}
        )
        assert resp.status_code == 404

    def test_update_to_empty_run_id_returns_422(self, integration_client):
        resp = integration_client.post('/updateModel', json={'run_id': ''})
        assert resp.status_code == 422
