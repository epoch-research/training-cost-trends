"""Tests for cost estimation functions."""

import numpy as np
import pandas as pd
import pytest

from training_cost_trends.constants import (
    CLUSTER_INTERCONNECT_COST_FRACTION,
    HOURS_PER_YEAR,
    MEDIAN_UTILIZATION,
    ML_GPU_PRICE_PERFORMANCE_OOMS_PER_YEAR,
    PRIORITY_VENDORS,
)
from training_cost_trends.cost_estimation import estimate_chip_hours


class TestEstimateChipHours:
    def test_from_quantity_and_time(self):
        row = pd.Series({
            "Training chip-hours": np.nan,
            "Hardware quantity": 256,
            "Training time (hours)": 720,
            "Training compute (FLOP)": np.nan,
            "Training hardware": np.nan,
            "Hardware utilization (MFU)": np.nan,
        })
        hardware_df = pd.DataFrame()
        result = estimate_chip_hours(row, hardware_df)
        assert result == 256 * 720

    def test_from_existing_chip_hours(self):
        row = pd.Series({
            "Training chip-hours": 50000,
            "Hardware quantity": 256,
            "Training time (hours)": 720,
        })
        hardware_df = pd.DataFrame()
        result = estimate_chip_hours(row, hardware_df)
        assert result == 50000


class TestPriorityVendorsImmutability:
    """Ensure the PRIORITY_VENDORS global is not mutated by find_price calls."""

    def test_priority_vendors_is_not_mutated(self):
        original = list(PRIORITY_VENDORS)
        # Importing find_price should not have changed the global
        from training_cost_trends.pricing import find_price  # noqa: F401
        assert PRIORITY_VENDORS == original


class TestCostConstants:
    def test_interconnect_fraction_range(self):
        assert 0 < CLUSTER_INTERCONNECT_COST_FRACTION < 1

    def test_hardware_progress_rate_positive(self):
        assert ML_GPU_PRICE_PERFORMANCE_OOMS_PER_YEAR > 0

    def test_hours_per_year(self):
        assert HOURS_PER_YEAR == pytest.approx(24 * 365.25)

    def test_median_utilization_range(self):
        assert 0 < MEDIAN_UTILIZATION < 1
