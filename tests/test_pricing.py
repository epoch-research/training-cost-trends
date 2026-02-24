"""Tests for pricing functions."""

import numpy as np
import pandas as pd
import pytest

from training_cost_trends.constants import ML_GPU_PRICE_PERFORMANCE_OOMS_PER_YEAR
from training_cost_trends.pricing import (
    apply_cud,
    depreciate_by_hardware_progress,
    get_acquisition_date,
    get_training_start_date,
)


class TestDepreciateByHardwareProgress:
    def test_no_depreciation_same_date(self):
        date = pd.Timestamp("2022-01-01")
        result = depreciate_by_hardware_progress(10000, date, date)
        assert result == pytest.approx(10000.0)

    def test_depreciation_over_one_year(self):
        start = pd.Timestamp("2022-01-01")
        end = pd.Timestamp("2023-01-01")
        result = depreciate_by_hardware_progress(10000, start, end)
        # After 1 year: 10^(log10(10000) - 0.14) = 10^(4 - 0.14) = 10^3.86
        expected = 10 ** (np.log10(10000) - ML_GPU_PRICE_PERFORMANCE_OOMS_PER_YEAR)
        assert result == pytest.approx(expected, rel=0.01)

    def test_depreciation_monotonically_decreasing(self):
        start = pd.Timestamp("2022-01-01")
        values = []
        for years in range(5):
            end = start + pd.Timedelta(days=365 * years)
            values.append(depreciate_by_hardware_progress(10000, start, end))
        # Each value should be less than the previous
        for i in range(1, len(values)):
            assert values[i] < values[i - 1]


class TestApplyCud:
    def test_same_price_type_no_change(self):
        result = apply_cud(10.0, "Amazon Web Services", "Price per chip-hour (3-year CUD)", "Price per chip-hour (3-year CUD)")
        assert result == 10.0

    def test_cud_adjustment_aws(self):
        # From on-demand (discount=0) to 3-year CUD (discount=0.64)
        result = apply_cud(
            10.0, "Amazon Web Services",
            "Price per chip-hour (on-demand)", "Price per chip-hour (3-year CUD)"
        )
        # Expected: 10 * (1-0.64)/(1-0) = 10 * 0.36 = 3.6
        assert result == pytest.approx(3.6)


class TestGetTrainingStartDate:
    def test_gpt4_special_case(self):
        row = pd.Series({
            "Model": "GPT-4",
            "Training time (hours)": 2160,
            "Publication date": pd.Timestamp("2023-03-14"),
        })
        result = get_training_start_date(row)
        expected = pd.Timestamp("2022-08-15") - pd.Timedelta(hours=2160)
        assert result == expected

    def test_gpt3_special_case(self):
        row = pd.Series({
            "Model": "GPT-3",
            "Training time (hours)": 1000,
            "Publication date": pd.Timestamp("2020-05-28"),
        })
        result = get_training_start_date(row)
        assert result == pd.Timestamp("2019-10-01")

    def test_generic_model_with_training_time(self):
        row = pd.Series({
            "Model": "Some Model",
            "Training time (hours)": 720,
            "Publication date": pd.Timestamp("2023-06-01"),
        })
        result = get_training_start_date(row)
        expected = pd.Timestamp("2023-06-01") - pd.Timedelta(hours=720) - pd.Timedelta(days=60)
        assert result == expected


class TestGetAcquisitionDate:
    def test_adds_buffer(self):
        row = pd.Series({
            "Model": "Some Model",
            "Training time (hours)": 720,
            "Publication date": pd.Timestamp("2023-06-01"),
        })
        training_start = get_training_start_date(row)
        acquisition = get_acquisition_date(row)
        assert acquisition == training_start - pd.Timedelta(days=60)
