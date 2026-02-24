"""Hardware specification functions: FLOP/s lookup, server economics, energy, and power."""

import logging

import pandas as pd

from .constants import HOURS_PER_YEAR, SIMPLIFIED_HARDWARE_NAMES

logger = logging.getLogger(__name__)


# ==============================================================================
# HARDWARE PERFORMANCE
# ==============================================================================


def get_flop_per_second(hardware_model: str, hardware_df: pd.DataFrame) -> float | None:
    """Look up hardware FLOP/s performance with a priority order of precision formats.

    Priority: Tensor-FP16/BF16 > TF32 > FP16 (TPUs only) > FP32 (fallback).
    """
    flop_per_second_columns = [
        "Tensor-FP16/BF16 performance (FLOP/s)",
        "TF32 (TensorFloat-32) performance (FLOP/s)",
        "FP16 (half precision) performance (FLOP/s)",
        "FP32 (single precision) performance (FLOP/s)",
    ]
    hardware_df_match = hardware_df[hardware_df["Hardware name"] == hardware_model]

    if "TPU v1" in hardware_model:
        flop_per_second = hardware_df_match["INT8 performance (OP/s)"].values[0]
        return flop_per_second

    flop_per_second = None
    for col in flop_per_second_columns:
        if col == "FP16 (half precision) performance (FLOP/s)":
            if "TPU" not in hardware_model:
                # FP16 performance for older GPUs can be misleading; only use for TPUs
                continue
        value = hardware_df_match[col].values[0]
        if not pd.isna(value):
            flop_per_second = value
            logger.debug("Found %s at %s FLOP/s", hardware_model, flop_per_second)
            break

    if flop_per_second is None:
        logger.warning("Could not find FLOP/s for %s", hardware_model)

    return flop_per_second


def get_simplified_hardware_model(hardware_model: str) -> str | None:
    """Map a specific hardware variant to its simplified name for soft matching."""
    return SIMPLIFIED_HARDWARE_NAMES.get(hardware_model)


def get_release_date(hardware_model: str, hardware_df: pd.DataFrame) -> pd.Timestamp:
    """Get the release date of a hardware model."""
    hardware_df_match = hardware_df[hardware_df["Hardware name"] == hardware_model]
    release_date = hardware_df_match["Release date"].values[0]
    return pd.to_datetime(release_date)


# ==============================================================================
# SERVER ECONOMICS
# ==============================================================================


def get_server_lifetime(year: int) -> float:
    """Return the estimated AI server lifetime in hours, varying by era."""
    if year <= 2020:
        return 3 * HOURS_PER_YEAR
    elif year >= 2023:
        return 5 * HOURS_PER_YEAR
    else:
        return 4 * HOURS_PER_YEAR


def get_server_cost_overhead(hardware_model: str) -> float:
    """Return server-to-chip cost multiplier (accounts for CPU, memory, etc.)."""
    if "A100" in hardware_model:
        return 1.66
    elif "V100" in hardware_model:
        return 1.69
    elif "P100" in hardware_model:
        return 1.54
    else:
        return 1.64  # average


# ==============================================================================
# ENERGY FUNCTIONS
# ==============================================================================


def power_usage_effectiveness(organization: str) -> float:
    """Return PUE (Power Usage Effectiveness) based on organization type.

    Hyperscalers (Google, Meta, Microsoft, Amazon) get 1.1; others get 1.25.
    """
    org = organization.lower()
    hyperscalers = ["google", "deepmind", "microsoft", "amazon", "meta", "facebook"]
    if any(hs in org for hs in hyperscalers):
        return 1.1
    return 1.25


def server_TDP_fraction(hardware_model: str) -> float:
    """Return the fraction of server TDP used during training (average power draw)."""
    if "TPU" in hardware_model:
        return 0.43
    else:
        return 0.75


def chip_to_server_power(hardware_model: str) -> float:
    """Return a multiplier to estimate server power from chip power."""
    if "TPU" in hardware_model:
        if "v1" in hardware_model:
            return 2.93
        elif "v2" in hardware_model:
            return 1.64
        elif "v3" in hardware_model:
            return 1.47
        else:
            return 1.56
    else:
        if "H100" in hardware_model:
            return (10.2 / 8) / 0.7
        elif "A100" in hardware_model:
            return (6.5 / 8) / 0.4
        elif "V100" in hardware_model:
            return ((10 / 16) / 0.3 + (1.5 / 4) / 0.3) / 2
        else:
            return 1.8


def energy_price(year: int) -> float:
    """Return average US industrial electricity price ($/kWh) for a given year."""
    prices = {
        2010: 0.0677,
        2011: 0.0682,
        2012: 0.0667,
        2013: 0.0689,
        2014: 0.0710,
        2015: 0.0691,
        2016: 0.0676,
        2017: 0.0688,
        2018: 0.0692,
        2019: 0.0681,
        2020: 0.0667,
        2021: 0.0718,
        2022: 0.0832,
        2023: 0.0806,
        2024: 0.0771,
        2025: 0.0826,
    }
    return prices[year]


def cluster_power_capacity(
    hardware_model: str,
    hardware_quantity: float,
    hardware_df: pd.DataFrame,
    organization: str,
) -> float | None:
    """Return the power capacity in kilowatts required for a training run."""
    matching_hardware = hardware_df[hardware_df["Hardware name"] == hardware_model]
    chip_TDP_kw = matching_hardware["TDP (W)"].squeeze() / 1000
    if pd.isna(chip_TDP_kw):
        if "TPU v4" in hardware_model:
            chip_TDP_kw = 373 / 1000
        else:
            logger.warning("Unable to estimate chip TDP")
            return None
    server_TDP_kw = chip_TDP_kw * chip_to_server_power(hardware_model)
    adj_server_power_kw = server_TDP_kw * power_usage_effectiveness(organization)
    cluster_kw = adj_server_power_kw * hardware_quantity
    return cluster_kw
