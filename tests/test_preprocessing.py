"""
Unit tests for ml_service/features.py — to_dataframe preprocessing.
"""
import pytest
import pandas as pd

from ml_service.features import to_dataframe, FEATURE_COLUMNS
from ml_service.schemas import PredictRequest


def make_request(**kwargs) -> PredictRequest:
    return PredictRequest(**kwargs)


class TestToDataframe:
    def test_returns_dataframe(self):
        req = make_request(race='White', sex='Male')
        df = to_dataframe(req)
        assert isinstance(df, pd.DataFrame)

    def test_single_row(self):
        req = make_request(race='White', sex='Male')
        df = to_dataframe(req)
        assert len(df) == 1

    def test_all_features_present_when_no_filter(self):
        req = make_request(race='White', sex='Male')
        df = to_dataframe(req)
        assert list(df.columns) == FEATURE_COLUMNS

    def test_needed_columns_filters_correctly(self):
        req = make_request(race='White', sex='Male', **{'capital.gain': 5000})
        needed = ['race', 'sex', 'capital.gain']
        df = to_dataframe(req, needed_columns=needed)
        assert set(df.columns) == set(needed)

    def test_column_values_match_request(self):
        req = make_request(**{
            'race': 'White',
            'sex': 'Male',
            'capital.gain': 12345,
            'education': 'Bachelors',
        })
        df = to_dataframe(req, needed_columns=['race', 'sex', 'capital.gain', 'education'])
        assert df['race'].iloc[0] == 'White'
        assert df['sex'].iloc[0] == 'Male'
        assert df['capital.gain'].iloc[0] == 12345
        assert df['education'].iloc[0] == 'Bachelors'

    def test_dotted_alias_fields_work(self):
        """Fields with dots use pydantic aliases — verify they reach the dataframe."""
        req = PredictRequest(**{'capital.gain': 99999, 'native.country': 'Germany'})
        df = to_dataframe(req, needed_columns=['capital.gain', 'native.country'])
        assert df['capital.gain'].iloc[0] == 99999
        assert df['native.country'].iloc[0] == 'Germany'

    def test_none_values_pass_through(self):
        req = make_request()
        df = to_dataframe(req, needed_columns=['race', 'sex'])
        assert df['race'].iloc[0] is None
        assert df['sex'].iloc[0] is None

    def test_needed_columns_ignores_unknown(self):
        """Columns not in FEATURE_COLUMNS should be silently ignored."""
        req = make_request(race='White')
        df = to_dataframe(req, needed_columns=['race', 'nonexistent_column'])
        assert 'race' in df.columns
        assert 'nonexistent_column' not in df.columns
