"""Price discovery, matching, CUD application, depreciation, and inflation adjustment."""

import logging

import numpy as np
import pandas as pd

from .constants import (
    DAYS_PER_YEAR,
    DEFAULT_CUD,
    GPU_HARDWARE_ALIASES,
    MEDIAN_TRAINING_TIME_DAYS,
    ML_GPU_PRICE_PERFORMANCE_OOMS_PER_YEAR,
    PRICE_INDEX_SERIES,
    PRIORITY_VENDORS,
    SIMPLIFIED_HARDWARE_NAMES,
    TPU_EQUIVALENT_RELEASE_PRICES,
)
from .hardware import get_release_date, get_server_cost_overhead, get_simplified_hardware_model

logger = logging.getLogger(__name__)


# ==============================================================================
# TEMPORAL PRICE MATCHING
# ==============================================================================


def find_closest_price_dates(
    hardware_model: str,
    date: pd.Timestamp,
    df: pd.DataFrame,
    vendor: str | None = None,
    price_colname: str | None = None,
) -> pd.DataFrame:
    """Find rows with the closest price date to the given date, matching hardware and vendor."""

    def filter_df(_hardware_model: str) -> pd.DataFrame:
        filtered_df = df
        if vendor is not None:
            filtered_df = df[df["Vendor"] == vendor]
        if price_colname is not None:
            filtered_df = filtered_df.dropna(subset=[price_colname])
        filtered_df = filtered_df[filtered_df["Hardware model"] == _hardware_model]
        return filtered_df

    filtered_df = filter_df(hardware_model)
    if len(filtered_df) == 0:
        simplified_hardware_model = get_simplified_hardware_model(hardware_model)
        if simplified_hardware_model is not None:
            logger.debug("Soft matching %s to %s", hardware_model, simplified_hardware_model)
            filtered_df = filter_df(simplified_hardware_model)
            if len(filtered_df) == 0:
                for full_hardware_model in SIMPLIFIED_HARDWARE_NAMES.keys():
                    terms = simplified_hardware_model.split()
                    if all(term in full_hardware_model for term in terms):
                        logger.debug("Soft matching %s to %s", hardware_model, full_hardware_model)
                        filtered_df = filter_df(full_hardware_model)
                        if len(filtered_df) > 0:
                            break

    target_date = pd.to_datetime(date)
    closest_row_df = filtered_df.iloc[(filtered_df["Price date"] - target_date).abs().argsort()]
    return closest_row_df


# ==============================================================================
# TRAINING TIMELINE
# ==============================================================================


def get_training_start_date(row: pd.Series, backup_training_time: bool = True) -> pd.Timestamp | None:
    """Estimate when training started, accounting for publication-to-training-end gap."""
    if "Gemini 1.0 Ultra" in row["Model"]:
        return pd.to_datetime("2023-05-10")
    elif row["Model"] == "GPT-4":
        training_time = pd.Timedelta(hours=int(row["Training time (hours)"]))
        return pd.to_datetime("2022-08-15") - training_time
    elif "GPT-3.5" in row["Model"]:
        return pd.to_datetime("2022-03-14")
    elif "GPT-3" in row["Model"]:
        return pd.to_datetime("2019-10-01")
    else:
        if pd.isna(row["Training time (hours)"]):
            if backup_training_time:
                logger.debug("No training time found, assuming %.1f days", MEDIAN_TRAINING_TIME_DAYS)
                training_time = pd.Timedelta(days=MEDIAN_TRAINING_TIME_DAYS)
            else:
                logger.debug("No training time found")
                return None
        else:
            training_time = pd.Timedelta(hours=int(row["Training time (hours)"]))
        buffer_time = pd.Timedelta(days=60)
        return row["Publication date"] - (training_time + buffer_time)


def get_acquisition_date(row: pd.Series, backup_training_time: bool = True) -> pd.Timestamp | None:
    """Estimate hardware acquisition date (training start minus buffer)."""
    training_start_date = get_training_start_date(row, backup_training_time)
    if training_start_date is None:
        return None
    buffer_time = pd.Timedelta(days=60)
    return training_start_date - buffer_time


# ==============================================================================
# VENDOR SELECTION
# ==============================================================================


def select_vendor(
    row: pd.Series,
    org_to_cloud_vendor: dict[str, str],
    default_vendor: bool = True,
) -> str | None:
    """Select the cloud vendor for a model based on its organization."""
    orgs = row["Organization"].split(",")
    vendor = None
    for org in orgs:
        for key in org_to_cloud_vendor:
            if key in org.lower():
                vendor = org_to_cloud_vendor[key]
                break
    if default_vendor and vendor is None:
        vendor = "Amazon Web Services"
    return vendor


# ==============================================================================
# PRICE LOOKUP
# ==============================================================================


def find_price_for_vendor_and_hardware_model(
    closest_price_dates_df: pd.DataFrame,
    acquisition_date: pd.Timestamp,
    price_colname: str,
    price_date_after: bool = True,
) -> tuple[float | None, str | None, pd.Timestamp | None]:
    """Find a price from the closest-price-dates DataFrame."""
    price_per_chip_hour = None
    price_id = None
    price_date = None

    for _, price_row in closest_price_dates_df.iterrows():
        if price_row["Price date"] <= acquisition_date:
            price_per_chip_hour = price_row[price_colname]
            price_id = price_row["Price source"]
            price_date = price_row["Price date"]
            break

    if price_per_chip_hour is None:
        for _, price_row in closest_price_dates_df.iterrows():
            if price_date_after and price_row["Price date"] > acquisition_date:
                price_per_chip_hour = price_row[price_colname]
                price_id = price_row["Price source"]
                price_date = price_row["Price date"]
                break

    if price_per_chip_hour is not None and not pd.isna(price_per_chip_hour):
        return float(price_per_chip_hour), price_id, price_date
    else:
        logger.debug("Could not find price")
        return None, None, None


def apply_cud(
    price_per_chip_hour: float,
    vendor: str,
    price_type: str,
    default_price_type: str,
) -> float:
    """Apply committed-use discount adjustment between price types."""
    if price_type == default_price_type:
        return price_per_chip_hour

    default_cud = DEFAULT_CUD[vendor][default_price_type]
    current_cud = DEFAULT_CUD[vendor][price_type]
    cud_ratio = (1 - default_cud) / (1 - current_cud)
    adjusted = price_per_chip_hour * cud_ratio
    logger.debug("Applying CUD: %s * %s = %s", price_per_chip_hour, cud_ratio, adjusted)
    return adjusted


def find_price(
    row: pd.Series,
    price_df: pd.DataFrame,
    hardware_df: pd.DataFrame,
    pcd_hardware_model_colname: str,
    price_colname: str,
    org_to_cloud_vendor: dict[str, str],
    price_date_after: bool = True,
    default_vendor: bool = True,
    backup_vendor: bool = True,
    backup_price_type: bool = True,
    backup_training_time: bool = True,
) -> tuple[float | None, str | None]:
    """Multi-tier fallback price lookup across vendors and CUD levels."""
    acquisition_date = get_acquisition_date(row, backup_training_time)
    if acquisition_date is None:
        return None, None

    hardware_model = row[pcd_hardware_model_colname]
    if pd.isna(hardware_model):
        logger.debug("Could not find hardware model for %s", row["Model"])
        return None, None

    vendor = select_vendor(row, org_to_cloud_vendor, default_vendor=default_vendor)
    if vendor is None:
        logger.debug("Could not find vendor for %s", row["Model"])
        return None, None

    logger.debug("Trying %s at %s", hardware_model, acquisition_date)

    # Use a local copy to avoid mutating the global PRIORITY_VENDORS list
    possible_vendors = list(PRIORITY_VENDORS)
    possible_vendors.extend(
        v for v in price_df["Vendor"].dropna().unique() if v not in possible_vendors
    )

    vendors = [vendor]
    if "TPU" not in hardware_model:
        for possible_vendor in possible_vendors:
            if possible_vendor != vendor:
                vendors.append(possible_vendor)
    elif vendor != "Google Cloud":
        vendor = "Google Cloud"
        vendors = [vendor]

    price_types = [price_colname]
    for possible_price_type in [
        "Price per chip-hour (3-year CUD)",
        "Price per chip-hour (1-year CUD)",
        "Price per chip-hour (on-demand)",
    ]:
        if possible_price_type != price_colname:
            price_types.append(possible_price_type)

    for price_type in price_types:
        for v in vendors:
            logger.debug("Trying %s, %s", v, price_type)
            closest_price_dates_df = find_closest_price_dates(
                hardware_model, acquisition_date, price_df, v,
            )
            price_value, price_id, price_date = find_price_for_vendor_and_hardware_model(
                closest_price_dates_df,
                acquisition_date,
                price_type,
                price_date_after=price_date_after,
            )
            if price_value is not None:
                if backup_price_type:
                    price_value = apply_cud(price_value, v, price_type, price_colname)
                logger.debug("Found price: %s at %s", price_value, price_date)
                logger.debug(
                    "Difference between acquisition date and price date: %s",
                    acquisition_date - price_date,
                )
                break
            if not backup_vendor:
                break
        if not backup_price_type:
            break
        if price_value is not None:
            break

    return price_value, price_id


# ==============================================================================
# GPU/TPU ACQUISITION PRICES
# ==============================================================================


def find_gpu_acquisition_price(
    price_df: pd.DataFrame, hardware_model: str, price_colname: str
) -> pd.Series | None:
    """Find the GPU acquisition price using hardware alias matching."""
    gpu_hardware_alias = None
    for alias in GPU_HARDWARE_ALIASES:
        if alias in hardware_model:
            gpu_hardware_alias = alias
            break
    if gpu_hardware_alias is None:
        logger.debug("Could not find alias for %s", hardware_model)
        return None

    price_df = price_df.sort_values(by="Price date").dropna(subset=[price_colname])
    matching_prices = price_df[price_df["Hardware model"].str.contains(gpu_hardware_alias)]

    if matching_prices.empty:
        raise ValueError(f"Could not find any prices for '{hardware_model}' in the price data.")

    chosen_price_row = None
    for _, price_row in matching_prices.iterrows():
        if "DGX" in price_row["Notes"]:
            chosen_price_row = price_row
            break
    if chosen_price_row is None:
        chosen_price_row = matching_prices.iloc[0]

    if not pd.isna(chosen_price_row[price_colname]):
        return chosen_price_row
    else:
        logger.debug("Could not find price for %s", hardware_model)
        return None


def find_TPU_equivalent_acquisition_price(hardware_model: str) -> float | None:
    """Look up the equivalent acquisition price for a TPU model."""
    price_value = TPU_EQUIVALENT_RELEASE_PRICES.get(hardware_model)
    if price_value is None:
        logger.debug("Could not find price for %s", hardware_model)
    return price_value


# ==============================================================================
# DEPRECIATION
# ==============================================================================


def depreciate_by_hardware_progress(
    initial_price: float, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> float:
    """Depreciate hardware price by the price-performance trend over time."""
    years_since = (end_date - pd.to_datetime(start_date)).days / DAYS_PER_YEAR
    depreciation = ML_GPU_PRICE_PERFORMANCE_OOMS_PER_YEAR * years_since
    price_value_ooms = np.log10(initial_price) - depreciation
    return 10**price_value_ooms


def find_hardware_acquisition_price(
    row: pd.Series,
    price_df: pd.DataFrame,
    hardware_df: pd.DataFrame,
    pcd_hardware_model_colname: str,
    price_colname: str,
) -> tuple[float | None, str | None, pd.Timestamp | None, str | None]:
    """Find the hardware acquisition price with server overhead adjustments."""
    hardware_model = row[pcd_hardware_model_colname]
    if pd.isna(hardware_model):
        logger.debug("Could not find hardware model for %s", row["Model"])
        return None, None, None, None
    if "," in hardware_model:
        logger.debug("Skipping %s (multiple hardware types)", hardware_model)
        return None, None, None, None

    if "TPU" in hardware_model:
        price_id = None
        price_value = find_TPU_equivalent_acquisition_price(hardware_model)
        if price_value is None:
            return None, None, None, None
        price_value *= get_server_cost_overhead(hardware_model)
        acquisition_date = get_release_date(hardware_model, hardware_df)
    else:
        chosen_price_row = find_gpu_acquisition_price(price_df, hardware_model, price_colname)
        if chosen_price_row is None:
            return None, None, None, None
        price_id = chosen_price_row["Price source"]
        price_value = chosen_price_row[price_colname]
        price_date = chosen_price_row["Price date"]
        release_date = get_release_date(hardware_model, hardware_df)
        buffer_time = pd.Timedelta(days=90)
        if price_date < release_date + buffer_time:
            acquisition_date = release_date + buffer_time
        else:
            acquisition_date = price_date
        if "single-unit" in chosen_price_row["Notes"].lower():
            price_value *= get_server_cost_overhead(hardware_model)

    return price_value, price_id, acquisition_date, hardware_model


def get_hardware_acquisition_price(
    row: pd.Series,
    price_df: pd.DataFrame,
    hardware_df: pd.DataFrame,
    pcd_hardware_model_colname: str,
    price_colname: str,
) -> tuple[float | None, str | None]:
    """Get the hardware acquisition price (wrapper around find_hardware_acquisition_price)."""
    price_value, price_id, acquisition_date, hardware_model = find_hardware_acquisition_price(
        row, price_df, hardware_df, pcd_hardware_model_colname, price_colname
    )
    if price_value is None:
        return None, None
    logger.debug(
        "Estimated the value of %s server, available from %s: %s per chip",
        hardware_model, acquisition_date, price_value,
    )
    return price_value, price_id


def get_hardware_value_at_training_start(
    row: pd.Series,
    price_df: pd.DataFrame,
    hardware_df: pd.DataFrame,
    pcd_hardware_model_colname: str,
    price_colname: str,
    backup_training_time: bool = True,
) -> tuple[float | None, str | None]:
    """Get hardware value depreciated to the training start date."""
    price_value, price_id, acquisition_date, hardware_model = find_hardware_acquisition_price(
        row, price_df, hardware_df, pcd_hardware_model_colname, price_colname
    )
    if price_value is None:
        return None, None

    training_start_date = get_training_start_date(row, backup_training_time)
    if "TPU" not in hardware_model and training_start_date < acquisition_date:
        training_start_date = acquisition_date

    price_value = depreciate_by_hardware_progress(price_value, acquisition_date, training_start_date)
    logger.debug(
        "Estimated the value of %s server, available from %s and used from %s: %s per chip",
        hardware_model, acquisition_date, training_start_date, price_value,
    )
    return price_value, price_id


# ==============================================================================
# INFLATION ADJUSTMENT
# ==============================================================================


def adjust_value_for_inflation(
    row: pd.Series, cost_colname: str, price_index: pd.DataFrame, to_year_month: str
) -> float:
    """Adjust a single cost value for inflation using the producer price index."""
    if pd.isna(row[cost_colname]):
        return row[cost_colname]

    from_date = str(row["Publication date"])
    from_year_month = from_date.rsplit("-", maxsplit=1)[0] + "-01"

    from_matches = price_index[price_index["observation_date"] == from_year_month]
    if len(from_matches) == 0:
        logger.warning("No price index found for %s, skipping inflation adjustment", from_year_month)
        return row[cost_colname]
    from_price_index = from_matches[PRICE_INDEX_SERIES].values[0]

    to_matches = price_index[price_index["observation_date"] == to_year_month]
    if len(to_matches) == 0:
        price_index_sorted = price_index.sort_values("observation_date", ascending=False)
        to_price_index = price_index_sorted[PRICE_INDEX_SERIES].iloc[0]
        actual_to_date = price_index_sorted["observation_date"].iloc[0]
        logger.warning(
            "No price index found for %s, using most recent date: %s", to_year_month, actual_to_date
        )
    else:
        to_price_index = to_matches[PRICE_INDEX_SERIES].values[0]

    adjust_factor = to_price_index / from_price_index
    return row[cost_colname] * adjust_factor


def adjust_column_for_inflation(
    df: pd.DataFrame, cost_colname: str, price_index: pd.DataFrame, to_year_month: str
) -> pd.DataFrame:
    """Add an inflation-adjusted column to the DataFrame."""
    df[cost_colname + " (inflation-adjusted)"] = df.apply(
        adjust_value_for_inflation, axis=1, args=(cost_colname, price_index, to_year_month)
    )
    return df
