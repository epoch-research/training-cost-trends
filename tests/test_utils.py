"""Tests for utility functions."""

import numpy as np
import pandas as pd
import pytest

from training_cost_trends.utils import (
    datetime_to_float_year,
    float_year_to_datetime,
    geomean,
    get_top_models,
    wgeomean,
)


class TestDatetimeConversions:
    def test_datetime_to_float_year_jan_1(self):
        dates = pd.Series(pd.to_datetime(["2020-01-01"]))
        result = datetime_to_float_year(dates)
        assert result.iloc[0] == pytest.approx(2020.0, abs=0.01)

    def test_datetime_to_float_year_july(self):
        dates = pd.Series(pd.to_datetime(["2020-07-01"]))
        result = datetime_to_float_year(dates)
        assert 2020.4 < result.iloc[0] < 2020.6

    def test_float_year_to_datetime_roundtrip(self):
        original = pd.Timestamp("2021-06-15")
        dates = pd.Series([original])
        float_year = datetime_to_float_year(dates).iloc[0]
        recovered = float_year_to_datetime(float_year)
        # Allow tolerance of a few days due to float precision
        assert abs((recovered - original).days) < 5


class TestGeomean:
    def test_geomean_equal_values(self):
        assert geomean(np.array([5.0, 5.0, 5.0])) == pytest.approx(5.0)

    def test_geomean_known_values(self):
        # geomean(1, 4) = 2
        assert geomean(np.array([1.0, 4.0])) == pytest.approx(2.0)

    def test_wgeomean_equal_weights(self):
        arr = np.array([1.0, 4.0])
        weights = np.array([1.0, 1.0])
        assert wgeomean(arr, weights) == pytest.approx(2.0)

    def test_wgeomean_skewed_weights(self):
        arr = np.array([1.0, 100.0])
        weights = np.array([1.0, 0.0])
        assert wgeomean(arr, weights) == pytest.approx(1.0)


class TestGetTopModels:
    def test_returns_top_n_models(self):
        df = pd.DataFrame({
            "Model": ["A", "B", "C", "D"],
            "Compute rank when published": [1, 3, 5, 10],
        })
        result = get_top_models(df, n=5)
        assert len(result) == 3
        assert list(result["Model"]) == ["A", "B", "C"]

    def test_returns_copy(self):
        df = pd.DataFrame({
            "Model": ["A"],
            "Compute rank when published": [1],
        })
        result = get_top_models(df, n=5)
        result["Model"] = "X"
        assert df["Model"].iloc[0] == "A"
