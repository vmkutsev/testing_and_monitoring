"""
Unit tests for ml_service/schemas.py — Pydantic request/response validation.
"""
import pytest
from pydantic import ValidationError

from ml_service.schemas import (
    PredictRequest,
    PredictResponse,
    UpdateModelRequest,
    UpdateModelResponse,
)


class TestPredictRequest:
    def test_all_fields_optional(self):
        """PredictRequest should be constructable with no fields."""
        req = PredictRequest()
        assert req.race is None

    def test_dotted_aliases_accepted(self):
        req = PredictRequest(**{'capital.gain': 1000, 'native.country': 'Canada'})
        assert req.capital_gain == 1000
        assert req.native_country == 'Canada'

    def test_python_field_names_accepted(self):
        req = PredictRequest(capital_gain=500, education='Bachelors')
        assert req.capital_gain == 500

    def test_wrong_type_raises(self):
        with pytest.raises(ValidationError):
            PredictRequest(age='not_an_int')

    def test_valid_full_payload(self):
        payload = {
            'race': 'White',
            'sex': 'Male',
            'native.country': 'United-States',
            'education': 'Bachelors',
            'occupation': 'Prof-specialty',
            'capital.gain': 5000,
            'age': 35,
        }
        req = PredictRequest(**payload)
        assert req.race == 'White'
        assert req.age == 35


class TestPredictResponse:
    def test_valid(self):
        resp = PredictResponse(prediction=1, probability=0.85)
        assert resp.prediction == 1
        assert resp.probability == pytest.approx(0.85)

    def test_prediction_must_be_int(self):
        with pytest.raises(ValidationError):
            PredictResponse(prediction='yes', probability=0.5)


class TestUpdateModelRequest:
    def test_valid(self):
        req = UpdateModelRequest(run_id='abc-123')
        assert req.run_id == 'abc-123'

    def test_missing_run_id_raises(self):
        with pytest.raises(ValidationError):
            UpdateModelRequest()


class TestUpdateModelResponse:
    def test_valid(self):
        resp = UpdateModelResponse(run_id='abc-123')
        assert resp.run_id == 'abc-123'
