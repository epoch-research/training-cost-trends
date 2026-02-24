"""Cost estimation methods: hardware CapEx+energy, hardware acquisition, and cloud costs."""

import logging
from typing import Callable

import numpy as np
import pandas as pd

from .constants import (
    CLUSTER_INTERCONNECT_COST_FRACTION,
    HOURS_PER_YEAR,
    MEDIAN_UTILIZATION,
    ML_GPU_PRICE_PERFORMANCE_OOMS_PER_YEAR,
    ORG_TO_CLOUD_VENDOR,
    SECONDS_PER_HOUR,
)
from .hardware import (
    energy_price,
    get_flop_per_second,
    get_server_cost_overhead,
    chip_to_server_power,
    cluster_power_capacity,
    power_usage_effectiveness,
    server_TDP_fraction,
)
from .pricing import (
    find_price,
    get_hardware_acquisition_price,
    get_hardware_value_at_training_start,
)
from .utils import print_safely

logger = logging.getLogger(__name__)


# ==============================================================================
# SHARED HELPERS
# ==============================================================================


def estimate_chip_hours(row: pd.Series, hardware_df: pd.DataFrame) -> float | None:
    """Estimate total chip-hours for a training run from available data."""
    if not pd.isna(row.get("Training chip-hours", np.nan)):
        return row["Training chip-hours"]

    hardware_quantity = row["Hardware quantity"]
    training_time = row["Training time (hours)"]

    if any(np.isnan(x) for x in [hardware_quantity, training_time] if not pd.isna(x) is False):
        # Try to compute from FLOP and hardware specs
        try:
            if np.isnan(hardware_quantity) or np.isnan(training_time):
                raise ValueError
        except (TypeError, ValueError):
            flop = row["Training compute (FLOP)"]
            hardware_model = row["Training hardware"]
            if not any(pd.isna(x) for x in [flop, hardware_model]):
                logger.debug("Imputing training time from compute and hardware")
                flop_per_second = get_flop_per_second(hardware_model, hardware_df)
                flop_utilization = row["Hardware utilization (MFU)"]
                if pd.isna(flop_utilization):
                    flop_utilization = MEDIAN_UTILIZATION
                training_chip_seconds = flop / (flop_per_second * flop_utilization)
                return training_chip_seconds / SECONDS_PER_HOUR
            else:
                return None

    return training_time * hardware_quantity


def cluster_energy_cost(
    hardware_model: str,
    total_chip_hours: float,
    hardware_df: pd.DataFrame,
    organization: str,
    year: int,
) -> float | None:
    """Calculate the total energy cost for a training run."""
    matching_hardware = hardware_df[hardware_df["Hardware name"] == hardware_model]
    chip_TDP_kw = matching_hardware["TDP (W)"].squeeze() / 1000
    if pd.isna(chip_TDP_kw):
        if "TPU v4" in hardware_model:
            chip_TDP_kw = 373 / 1000
        else:
            logger.warning("Unable to estimate chip TDP")
            return None
    server_TDP_kw = chip_TDP_kw * chip_to_server_power(hardware_model)
    server_power_kw = server_TDP_kw * server_TDP_fraction(hardware_model)
    adj_server_power_kw = server_power_kw * power_usage_effectiveness(organization)
    cluster_kwh = adj_server_power_kw * total_chip_hours
    return cluster_kwh * energy_price(year)


def _run_estimation(
    frontier_pcd_df: pd.DataFrame,
    price_fn: Callable,
    cost_fn: Callable,
    impute_pcd_fn: Callable | None = None,
    **impute_kwargs,
) -> pd.DataFrame:
    """Shared pipeline: optionally impute, look up prices, then estimate costs.

    Args:
        frontier_pcd_df: DataFrame of frontier models.
        price_fn: Function(row) -> (price, price_id) to look up per-model price.
        cost_fn: Function(row, system_to_price) -> cost to estimate per-model cost.
        impute_pcd_fn: Optional imputation function.
        **impute_kwargs: Keyword arguments for the imputation function.
    """
    if impute_pcd_fn is not None:
        frontier_pcd_df = impute_pcd_fn(frontier_pcd_df, **impute_kwargs)

    # Phase 1: Look up prices
    system_to_price: dict[str, float] = {}
    for _, row in frontier_pcd_df.iterrows():
        print_safely(f"==== System: {row['Model']} ====")
        price, _ = price_fn(row)
        if price is not None:
            system_to_price[row["Model"]] = price

    # Phase 2: Estimate costs
    system_to_cost: dict[str, float | dict] = {}
    for _, row in frontier_pcd_df.iterrows():
        print_safely(f"==== System: {row['Model']} ====")
        cost = cost_fn(row, system_to_price)
        if cost is None:
            logger.debug("Unable to estimate cost")
            continue
        else:
            logger.debug("Estimated cost: %s", cost)
        system_to_cost[row["Model"]] = cost

    logger.debug("All costs: %s", system_to_cost)
    return frontier_pcd_df, system_to_cost


# ==============================================================================
# METHOD 1: HARDWARE CAPEX + ENERGY
# ==============================================================================


def _capex_energy_cost_fn(
    row: pd.Series,
    system_to_price: dict[str, float],
    hardware_df: pd.DataFrame,
    separate_components: bool = False,
) -> float | dict | None:
    """Per-model cost function for the amortized hardware CapEx + energy method."""
    system = row["Model"]
    price = system_to_price.get(system)
    if price is None:
        return None

    hardware_model = row["Training hardware"]
    training_chip_hours = estimate_chip_hours(row, hardware_df)
    if training_chip_hours is None:
        logger.debug("Unable to estimate training chip hours")
        return None

    hardware_replacement_per_year = ML_GPU_PRICE_PERFORMANCE_OOMS_PER_YEAR * np.log(10)
    price_per_chip_hour = price * hardware_replacement_per_year / HOURS_PER_YEAR
    amortized_hardware_cost = price_per_chip_hour * training_chip_hours

    interconnect_cost = (
        amortized_hardware_cost * CLUSTER_INTERCONNECT_COST_FRACTION
        / (1 - CLUSTER_INTERCONNECT_COST_FRACTION)
    )

    org = row["Organization"]
    pub_year = row["Publication date"].year
    energy = cluster_energy_cost(hardware_model, training_chip_hours, hardware_df, org, pub_year)

    if separate_components:
        ai_chip_cost = amortized_hardware_cost / get_server_cost_overhead(hardware_model)
        extra_server_cost = amortized_hardware_cost - ai_chip_cost
        cost = {
            "AI accelerator chip cost": ai_chip_cost,
            "Other server components cost": extra_server_cost,
            "Cluster-level interconnect cost": interconnect_cost,
            "Energy cost": energy,
        }
    else:
        cost = amortized_hardware_cost + interconnect_cost + energy
        overall_cost_per_chip_hour = cost / training_chip_hours
        logger.debug("Overall cost per chip-hour for %s: %s", hardware_model, overall_cost_per_chip_hour)

    # Skip fine-tuned models
    if not pd.isna(row["Base model"]):
        logger.debug("Skipping %s because it is a fine-tuned version of a base model", system)
        return None

    return cost


def estimate_hardware_capex_energy(
    frontier_pcd_df: pd.DataFrame,
    hardware_df: pd.DataFrame,
    price_df: pd.DataFrame,
    separate_components: bool = False,
    impute_pcd_fn: Callable | None = None,
    **impute_kwargs,
) -> pd.DataFrame:
    """Full pipeline for estimating the sum of amortized hardware CapEx and energy cost."""
    pcd_hardware_model_colname = "Training hardware"
    price_colname = "Price (hardware purchase)"

    def price_fn(row):
        return get_hardware_value_at_training_start(
            row, price_df, hardware_df, pcd_hardware_model_colname, price_colname
        )

    def cost_fn(row, system_to_price):
        return _capex_energy_cost_fn(row, system_to_price, hardware_df, separate_components)

    frontier_pcd_df, system_to_cost = _run_estimation(
        frontier_pcd_df, price_fn, cost_fn, impute_pcd_fn, **impute_kwargs
    )

    if separate_components:
        cost_component_names = [
            "AI accelerator chip cost",
            "Other server components cost",
            "Cluster-level interconnect cost",
            "Energy cost",
        ]
        for k in cost_component_names:
            system_to_component_cost = {
                system: system_to_cost[system][k] for system in system_to_cost
            }
            frontier_pcd_df[k] = frontier_pcd_df["Model"].map(system_to_component_cost)
        frontier_pcd_df["Cost"] = frontier_pcd_df[cost_component_names].sum(axis=1)
    else:
        frontier_pcd_df["Cost"] = frontier_pcd_df["Model"].map(system_to_cost)

    return frontier_pcd_df


# ==============================================================================
# METHOD 2: HARDWARE ACQUISITION
# ==============================================================================


def estimate_hardware_acquisition_cost(
    frontier_pcd_df: pd.DataFrame,
    hardware_df: pd.DataFrame,
    price_df: pd.DataFrame,
    impute_pcd_fn: Callable | None = None,
    **impute_kwargs,
) -> pd.DataFrame:
    """Full pipeline for estimating up-front server capex."""
    logger.info("Estimating up-front server capex")
    pcd_hardware_model_colname = "Training hardware"
    price_colname = "Price (hardware purchase)"

    def price_fn(row):
        return get_hardware_acquisition_price(
            row, price_df, hardware_df, pcd_hardware_model_colname, price_colname
        )

    def cost_fn(row, system_to_price):
        system = row["Model"]
        price = system_to_price.get(system)
        if price is None:
            logger.debug("No hardware price found")
            return None
        hardware_quantity = row["Hardware quantity"]
        if pd.isna(hardware_quantity):
            logger.debug("No hardware quantity found")
            return None
        cost = hardware_quantity * price
        cost *= 1 / (1 - CLUSTER_INTERCONNECT_COST_FRACTION)
        if not pd.isna(row["Base model"]):
            logger.debug("Skipping %s because it is a fine-tuned version of a base model", system)
            return None
        return cost

    frontier_pcd_df, system_to_cost = _run_estimation(
        frontier_pcd_df, price_fn, cost_fn, impute_pcd_fn, **impute_kwargs
    )
    frontier_pcd_df["Cost"] = frontier_pcd_df["Model"].map(system_to_cost)
    return frontier_pcd_df


# ==============================================================================
# METHOD 3: CLOUD COSTS
# ==============================================================================


def estimate_cloud_costs(
    frontier_pcd_df: pd.DataFrame,
    hardware_df: pd.DataFrame,
    price_df: pd.DataFrame,
    impute_pcd_fn: Callable | None = None,
    **impute_kwargs,
) -> pd.DataFrame:
    """Full pipeline for estimating cloud rental costs."""
    pcd_hardware_model_colname = "Training hardware"
    price_colname = "Price per chip-hour (3-year CUD)"

    def price_fn(row):
        return find_price(
            row, price_df, hardware_df, pcd_hardware_model_colname, price_colname, ORG_TO_CLOUD_VENDOR
        )

    def cost_fn(row, system_to_price):
        system = row["Model"]
        price = system_to_price.get(system)
        if price is None:
            return None

        if "Training time (chip hours)" in row.index:
            chip_hours = row["Training time (chip hours)"]
        else:
            chip_hours = estimate_chip_hours(row, hardware_df)
        if np.isnan(chip_hours):
            logger.debug("Unable to estimate chip hours")
            return None

        cost = price * chip_hours
        if not pd.isna(row["Base model"]):
            logger.debug("Skipping %s because it is a fine-tuned version of a base model", system)
            return None
        return cost

    frontier_pcd_df, system_to_cost = _run_estimation(
        frontier_pcd_df, price_fn, cost_fn, impute_pcd_fn, **impute_kwargs
    )
    frontier_pcd_df["Cost"] = frontier_pcd_df["Model"].map(system_to_cost)
    return frontier_pcd_df
