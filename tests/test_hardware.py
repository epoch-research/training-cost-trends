"""Tests for hardware specification functions."""

import pytest

from training_cost_trends.constants import HOURS_PER_YEAR
from training_cost_trends.hardware import (
    chip_to_server_power,
    energy_price,
    get_server_cost_overhead,
    get_server_lifetime,
    get_simplified_hardware_model,
    power_usage_effectiveness,
    server_TDP_fraction,
)


class TestGetServerLifetime:
    def test_early_era(self):
        assert get_server_lifetime(2018) == 3 * HOURS_PER_YEAR
        assert get_server_lifetime(2020) == 3 * HOURS_PER_YEAR

    def test_middle_era(self):
        assert get_server_lifetime(2021) == 4 * HOURS_PER_YEAR
        assert get_server_lifetime(2022) == 4 * HOURS_PER_YEAR

    def test_late_era(self):
        assert get_server_lifetime(2023) == 5 * HOURS_PER_YEAR
        assert get_server_lifetime(2025) == 5 * HOURS_PER_YEAR


class TestGetServerCostOverhead:
    def test_a100(self):
        assert get_server_cost_overhead("NVIDIA A100 SXM4 80 GB") == 1.66

    def test_v100(self):
        assert get_server_cost_overhead("NVIDIA V100") == 1.69

    def test_p100(self):
        assert get_server_cost_overhead("NVIDIA P100") == 1.54

    def test_unknown_returns_average(self):
        assert get_server_cost_overhead("some-unknown-chip") == 1.64


class TestPowerUsageEffectiveness:
    def test_google_hyperscaler(self):
        assert power_usage_effectiveness("Google DeepMind") == 1.1

    def test_meta_hyperscaler(self):
        assert power_usage_effectiveness("Meta AI") == 1.1

    def test_microsoft_hyperscaler(self):
        assert power_usage_effectiveness("Microsoft Research") == 1.1

    def test_non_hyperscaler(self):
        assert power_usage_effectiveness("Stanford University") == 1.25

    def test_case_insensitive(self):
        assert power_usage_effectiveness("GOOGLE") == 1.1


class TestServerTDPFraction:
    def test_tpu(self):
        assert server_TDP_fraction("Google TPU v3") == 0.43

    def test_gpu(self):
        assert server_TDP_fraction("NVIDIA A100") == 0.75


class TestChipToServerPower:
    def test_tpu_v1(self):
        assert chip_to_server_power("Google TPU v1") == 2.93

    def test_tpu_v2(self):
        assert chip_to_server_power("Google TPU v2") == 1.64

    def test_h100(self):
        expected = (10.2 / 8) / 0.7
        assert chip_to_server_power("NVIDIA H100 SXM5") == pytest.approx(expected)

    def test_unknown_gpu(self):
        assert chip_to_server_power("NVIDIA GTX 580") == 1.8


class TestEnergyPrice:
    def test_known_years(self):
        assert energy_price(2020) == 0.0667
        assert energy_price(2023) == 0.0806
        assert energy_price(2025) == 0.0826

    def test_unknown_year_raises(self):
        with pytest.raises(KeyError):
            energy_price(2005)


class TestGetSimplifiedHardwareModel:
    def test_known_mapping(self):
        assert get_simplified_hardware_model("NVIDIA A100 SXM4 80 GB") == "NVIDIA A100"
        assert get_simplified_hardware_model("NVIDIA Tesla V100 DGXS 16 GB") == "NVIDIA V100"

    def test_unknown_returns_none(self):
        assert get_simplified_hardware_model("NVIDIA RTX 4090") is None
