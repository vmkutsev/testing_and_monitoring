"""
Tests for FastAPI handlers using a mocked model (no MLflow calls).
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from tests.conftest import MOCK_RUN_ID, MOCK_FEATURES, make_mock_model
from ml_service.model import ModelData
from ml_service.app import app


_UNSET = object()


def patched_client(model=_UNSET, run_id=MOCK_RUN_ID, features=None):
    """Context manager: returns a TestClient with MODEL fully mocked.

    Pass model=None explicitly to simulate an unloaded model (503).
    Omit model to get a default mock model.
    """
    if model is _UNSET:
        model = make_mock_model()
    if features is None:
        features = MOCK_FEATURES

    model_data = ModelData(model=model, run_id=run_id)

    p_model = patch('ml_service.app.MODEL')
    p_monitor = patch('ml_service.app.init_monitoring')
    p_cron = patch('ml_service.app.drift_monitoring_cron')
    p_record = patch('ml_service.app.record_request')

    class _Ctx:
        def __enter__(self):
            self.m = p_model.start()
            p_monitor.start()
            p_cron.start()
            p_record.start()
            self.m.get.return_value = model_data
            self.m.features = features
            self.client = TestClient(app, raise_server_exceptions=False)
            return self.client

        def __exit__(self, *args):
            p_model.stop()
            p_monitor.stop()
            p_cron.stop()
            p_record.stop()

    return _Ctx()


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_ok(self):
        with patched_client() as client:
            resp = client.get('/health')
            assert resp.status_code == 200
            assert resp.json()['status'] == 'ok'

    def test_returns_run_id(self):
        with patched_client(run_id='some-run-id') as client:
            resp = client.get('/health')
            assert resp.json()['run_id'] == 'some-run-id'


# ── /predict ──────────────────────────────────────────────────────────────────

class TestPredict:
    def test_valid_request_returns_200(self, valid_predict_payload):
        with patched_client() as client:
            resp = client.post('/predict', json=valid_predict_payload)
            assert resp.status_code == 200

    def test_response_has_prediction_and_probability(self, valid_predict_payload):
        with patched_client() as client:
            resp = client.post('/predict', json=valid_predict_payload)
            data = resp.json()
            assert 'prediction' in data
            assert 'probability' in data

    def test_high_probability_predicts_1(self, valid_predict_payload):
        with patched_client(model=make_mock_model(probability=0.9)) as client:
            resp = client.post('/predict', json=valid_predict_payload)
            assert resp.json()['prediction'] == 1

    def test_low_probability_predicts_0(self, valid_predict_payload):
        with patched_client(model=make_mock_model(probability=0.1)) as client:
            resp = client.post('/predict', json=valid_predict_payload)
            assert resp.json()['prediction'] == 0

    def test_probability_in_range(self, valid_predict_payload):
        with patched_client() as client:
            resp = client.post('/predict', json=valid_predict_payload)
            prob = resp.json()['probability']
            assert 0.0 <= prob <= 1.0

    def test_empty_body_returns_200(self):
        """Empty body is valid: all fields optional, None values are passed to model."""
        with patched_client() as client:
            resp = client.post('/predict', json={})
            assert resp.status_code == 200

    def test_invalid_field_type_returns_422(self):
        with patched_client() as client:
            resp = client.post('/predict', json={'age': 'not_an_int'})
            assert resp.status_code == 422

    def test_model_not_loaded_returns_503(self):
        with patched_client(model=None) as client:
            resp = client.post('/predict', json={'race': 'White'})
            assert resp.status_code == 503

    def test_inference_error_returns_500(self, valid_predict_payload):
        bad_model = make_mock_model()
        bad_model.predict_proba.side_effect = RuntimeError('inference blew up')
        with patched_client(model=bad_model) as client:
            resp = client.post('/predict', json=valid_predict_payload)
            assert resp.status_code == 500


# ── /updateModel ──────────────────────────────────────────────────────────────

class TestUpdateModel:
    def test_valid_run_id_returns_200(self):
        with patched_client() as client:
            with patch('ml_service.app.MODEL.set'), \
                 patch('ml_service.app.update_model_info'):
                resp = client.post('/updateModel', json={'run_id': 'new-run-id'})
                assert resp.status_code == 200
                assert resp.json()['run_id'] == 'new-run-id'

    def test_empty_run_id_returns_422(self):
        with patched_client() as client:
            resp = client.post('/updateModel', json={'run_id': ''})
            assert resp.status_code == 422

    def test_missing_run_id_returns_422(self):
        with patched_client() as client:
            resp = client.post('/updateModel', json={})
            assert resp.status_code == 422

    def test_invalid_run_id_returns_404(self):
        import mlflow.exceptions
        with patched_client() as client:
            with patch('ml_service.app.MODEL.set',
                       side_effect=mlflow.exceptions.MlflowException('not found')):
                resp = client.post('/updateModel', json={'run_id': 'bad-id'})
                assert resp.status_code == 404
