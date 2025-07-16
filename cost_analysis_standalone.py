#!/usr/bin/env python3
"""
Standalone cost analysis script that updates model costs.
Contains all necessary functions from the project dependencies.
"""

import json
import numpy as np
import os
import pandas as pd
from contextlib import redirect_stdout

# ==============================================================================
# CONSTANTS AND PARAMETERS
# ==============================================================================

SECONDS_PER_HOUR = 60 * 60
DAYS_PER_YEAR = 365.25
HOURS_PER_YEAR = 24 * DAYS_PER_YEAR
SECONDS_PER_YEAR = SECONDS_PER_HOUR * 24 * DAYS_PER_YEAR
CLUSTER_INTERCONNECT_COST_FRACTION = 0.19  # fraction of total cluster cost
MEDIAN_UTILIZATION = 0.375  # median of 33 known values
MEDIAN_TRAINING_TIME_DAYS = 793.5 / 24  # median of the running top-10 models
ML_GPU_PRICE_PERFORMANCE_OOMS_PER_YEAR = 0.14  # https://epochai.org/blog/trends-in-machine-learning-hardware

DEFAULT_RNG = np.random.default_rng(20240531)

# ==============================================================================
# HARDWARE MAPPINGS AND CONSTANTS
# ==============================================================================

# Mapping of simplified hardware names, for soft matching
SIMPLIFIED_HARDWARE_NAMES = {
    'NVIDIA Tesla V100 DGXS 16 GB': 'NVIDIA V100',
    'NVIDIA Tesla V100 DGXS 32 GB': 'NVIDIA V100',
    'NVIDIA Tesla V100S PCIe 32 GB': 'NVIDIA V100',  # similar in specs
    'NVIDIA V100': 'NVIDIA V100',
    'NVIDIA A100 PCIe': 'NVIDIA A100',
    'NVIDIA A100 SXM4 40 GB': 'NVIDIA A100',
    'NVIDIA A100 SXM4 80 GB': 'NVIDIA A100',
    'NVIDIA A100': 'NVIDIA A100',
    'NVIDIA H100 PCIe': 'NVIDIA H100',
    'NVIDIA H100 SXM5': 'NVIDIA H100',
    'NVIDIA H100': 'NVIDIA H100',
}

GPU_HARWARE_ALIASES = [
    'A100',
    'V100',
    'H100',
    'P100',
    'K80',
    'K40',
    'Titan X',
    'GTX 580',
    'GTX TITAN',
    'Titan Black',
]

# Default committed-use discounts (CUD) for each cloud provider
# See cud_estimate.ipynb
DEFAULT_CUD = {
    'Amazon Web Services': {
        'Price per chip-hour (on-demand)': 0,
        'Price per chip-hour (1-year CUD)': 0.41,
        'Price per chip-hour (3-year CUD)': 0.64,
    },
    'Google Cloud': {
        'Price per chip-hour (on-demand)': 0,
        'Price per chip-hour (1-year CUD)': 0.37,
        'Price per chip-hour (3-year CUD)': 0.56,
    },
    'Microsoft Azure': {
        'Price per chip-hour (on-demand)': 0,
        'Price per chip-hour (1-year CUD)': 0.25,
        'Price per chip-hour (3-year CUD)': 0.49,
    },
    # Assume Lambda Labs has average of above CUDs
    'Lambda Labs': {
        'Price per chip-hour (on-demand)': 0,
        'Price per chip-hour (1-year CUD)': 0.34,
        'Price per chip-hour (3-year CUD)': 0.56,
    },
}

# These numbers are kept precise to match our calculations
# Even though they have high uncertainty
TPU_EQUIVALENT_RELEASE_PRICES = {
    "Google TPU v1": 5463,
    "Google TPU v2": 5054,
    "Google TPU v3": 5276,
    "Google TPU v4": 5167,
}

PRIORITY_VENDORS = ['Amazon Web Services', 'Microsoft Azure', 'Google Cloud']

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def datetime_to_float_year(datetimes):
    date_floats = datetimes.dt.year + (datetimes.dt.month-1) / 12 + (datetimes.dt.day-1) / 365
    return date_floats

def float_year_to_datetime(float_year):
    year = int(float_year)
    remainder = float_year - year
    days_in_year = 365 + int(pd.Timestamp(year=year, month=12, day=31).is_leap_year)
    day_of_year = int(remainder * days_in_year)
    return pd.Timestamp(year=year, month=1, day=1) + pd.to_timedelta(day_of_year, unit='D')

def print_safely(input_string):
    """Prints strings, and can handle unicode encoding errors."""
    try:
        print(input_string)
    except UnicodeEncodeError:
        encoded_string = ''.join([char if char.encode('ascii', 'ignore') else f'\\u{ord(char):04x}' for char in input_string])
        print(f"Could not print original string due to encoding error. Printing modified version: {encoded_string}")

def geomean(arr):
    return np.exp(np.mean(np.log(arr)))

def wgeomean(arr, weights):
    return np.exp(np.average(np.log(arr), weights=weights))

# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================

def load_frontier_systems(compute_threshold_method='top_n', compute_threshold=10):
    """
    Load the frontier systems from the file

    Returns a list of the frontier systems
    """
    frontier_systems = []
    with open(f'data/frontier_systems_by_{compute_threshold_method}.json', 'r') as f:
        # Load JSON data
        frontier_systems_index = json.load(f)
        if compute_threshold_method == 'top_n':
            indices = range(1, compute_threshold+1)
        elif compute_threshold_method == 'window_percentile':
            indices = range(compute_threshold, 100, 5)
        elif compute_threshold_method == 'backward_window_percentile':
            indices = range(compute_threshold, 100, 5)
        elif compute_threshold_method == 'residual_from_trend':
            indices = range(compute_threshold, 100, 5)
        else:
            raise ValueError(f"Invalid compute_threshold_method: {compute_threshold_method}")
        for i in indices:
            frontier_systems.extend(frontier_systems_index[str(i)])

    return frontier_systems

def load_pcd_df():
    dtype = {'Training compute (FLOP)': 'float64'}
    return pd.read_csv('data/All ML Systems - full view.csv', dtype=dtype)

def load_hardware_df():
    return pd.read_csv('data/Chip dataset-Grid view.csv')

def load_price_df():
    return pd.read_csv('data/Hardware prices.csv')

def load_data_for_cost_estimation(compute_threshold_method='top_n', compute_threshold=10):
    """
    Load the data needed for cost estimation

    Returns a tuple of the frontier systems PCD dataframe, hardware dataframe, and price dataframe
    """
    pcd_df = load_pcd_df()

    # Publication date in datetime format
    pcd_df.dropna(subset=['Publication date'], inplace=True)
    pcd_df['Publication date'] = pd.to_datetime(pcd_df['Publication date'])

    frontier_systems = load_frontier_systems(
        compute_threshold_method=compute_threshold_method,
        compute_threshold=compute_threshold,
    )
    frontier_systems = [_.replace('Î£', 'Σ') for _ in frontier_systems]
    frontier_pcd_df = pcd_df[pcd_df['Model'].isin(frontier_systems)]

    ## Prices
    price_df = load_price_df()

    # Price date in datetime format
    price_df.dropna(subset=['Price date'], inplace=True)
    price_df['Price date'] = pd.to_datetime(price_df['Price date'])

    ## Hardware data
    hardware_df = load_hardware_df()

    return frontier_pcd_df, hardware_df, price_df

# ==============================================================================
# HARDWARE FUNCTIONS
# ==============================================================================

def get_flop_per_second(hardware_model, hardware_df):
    # Get FLOP/second from the hardware database
    flop_per_second_columns = [  # ordered by preference
        'Tensor-FP16/BF16 performance (FLOP/s)',
        'TF32 (TensorFloat-32) performance (FLOP/s)',
        'FP16 (half precision) performance (FLOP/s)',
        'FP32 (single precision) performance (FLOP/s)',
    ]
    hardware_df_match = hardware_df[hardware_df['Hardware name'] == hardware_model]
    if 'TPU v1' in hardware_model:
        # Special case
        flop_per_second = hardware_df_match['INT8 performance (OP/s)'].values[0]
        return flop_per_second
    for col in flop_per_second_columns:
        if col == 'FP16 (half precision) performance (FLOP/s)':
            if 'TPU' in hardware_model:
                # FP16 performance for older GPUs can be misleading
                # So only use it for TPUs
                flop_per_second = hardware_df_match[col].values[0]
        else:
            flop_per_second = hardware_df_match[col].values[0]
        if not pd.isna(flop_per_second):
            print(f"Found {hardware_model} at {flop_per_second} FLOP/s")
            break
    if pd.isna(flop_per_second):
        print(f"Could not find FLOP/s for {hardware_model}")
        return None
    return flop_per_second

def get_simplified_hardware_model(hardware_model):
    return SIMPLIFIED_HARDWARE_NAMES.get(hardware_model)

def get_release_date(hardware_model, hardware_df):
    hardware_df_match = hardware_df[hardware_df['Hardware name'] == hardware_model]
    release_date = hardware_df_match['Release date'].values[0]
    return pd.to_datetime(release_date)

def get_server_lifetime(year):
    """
    Returns the estimated AI server lifetime in hours.
    """
    if year <= 2020:
        return 3 * HOURS_PER_YEAR
    elif year >= 2023:
        return 5 * HOURS_PER_YEAR
    else:
        return 4 * HOURS_PER_YEAR

def get_server_cost_overhead(hardware_model):
    if 'A100' in hardware_model:
        return 1.66
    elif 'V100' in hardware_model:
        return 1.69
    elif 'P100' in hardware_model:
        return 1.54
    else:
        # average
        return 1.64

# ==============================================================================
# ENERGY FUNCTIONS
# ==============================================================================

def power_usage_effectiveness(organization):
    org = organization.lower()
    hyperscalers = ['google', 'deepmind', 'microsoft', 'amazon', 'meta', 'facebook']
    if any([hs in org for hs in hyperscalers]):
        return 1.1
    return 1.25

def server_TDP_fraction(hardware_model):
    """
    Returns the fraction of the server's TDP that is used during training.
    """
    if "TPU" in hardware_model:
        return 0.43
    else:
        return 0.75

def chip_to_server_power(hardware_model):
    """
    Returns a multiplier to estimate server power from chip power.
    """
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

def energy_price(year):
    """
    Average US industrial electricity price
    """
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

def cluster_power_capacity(hardware_model, hardware_quantity, hardware_df, organization):
    """
    Returns the power capacity in kilowatts required to do the training run.
    """
    matching_hardware = hardware_df[hardware_df['Hardware name'] == hardware_model]
    chip_TDP_kw = matching_hardware['TDP (W)'].squeeze() / 1000
    if pd.isna(chip_TDP_kw):
        if "TPU v4" in hardware_model:
            chip_TDP_kw = 373 / 1000
        else:
            print("Unable to estimate chip TDP")
            return None
    # Adjust for whole server power draw (CPUs, memory, cooling)
    server_TDP_kw = chip_TDP_kw * chip_to_server_power(hardware_model)
    # Adjust for data center power distribution and cooling
    adj_server_power_kw = server_TDP_kw * power_usage_effectiveness(organization)
    cluster_kw = adj_server_power_kw * hardware_quantity
    return cluster_kw

# ==============================================================================
# PRICE FUNCTIONS
# ==============================================================================

def find_closest_price_dates(hardware_model, date, df, vendor=None, price_colname=None):
    """
    Finds the rows in the DataFrame with the closest 'Price date' to the given date and which match
    the hardware model, vendor (if provided), and price column (if provided).
    """
    # Filter the DataFrame based on vendor and hardware model
    def filter_df(_hardware_model):
        filtered_df = df
        if vendor is not None:
            filtered_df = df[df['Vendor'] == vendor]
        if price_colname is not None:
            filtered_df = filtered_df.dropna(subset=[price_colname])
        filtered_df = filtered_df[filtered_df['Hardware model'] == _hardware_model]
        return filtered_df

    filtered_df = filter_df(hardware_model)
    if len(filtered_df) == 0:
        # No exact match found, try soft matching
        simplified_hardware_model = get_simplified_hardware_model(hardware_model)
        if simplified_hardware_model is not None:
            print(f"Soft matching {hardware_model} to {simplified_hardware_model}")
            filtered_df = filter_df(simplified_hardware_model)
            if len(filtered_df) == 0:
                # try any version of the hardware name (using overlap of simplified name)
                for full_hardware_model in SIMPLIFIED_HARDWARE_NAMES.keys():
                    terms = simplified_hardware_model.split()
                    if all([term in full_hardware_model for term in terms]):
                        print(f"Soft matching {hardware_model} to {full_hardware_model}")
                        filtered_df = filter_df(full_hardware_model)
                        if len(filtered_df) > 0:
                            break

    # Convert the target date to datetime
    target_date = pd.to_datetime(date)

    # Find the row with the smallest date difference
    closest_row_df = filtered_df.iloc[(filtered_df['Price date'] - target_date).abs().argsort()]

    return closest_row_df

def get_training_start_date(row, backup_training_time=True):
    # Account for time between publication and training end
    # Special case for models where the gap is abnormally large
    if 'Gemini 1.0 Ultra' in row['Model']:
        training_start_date = pd.to_datetime('2023-05-10')
    elif row['Model'] == 'GPT-4':
        training_time = pd.Timedelta(hours=int(row['Training time (hours)']))
        training_start_date = pd.to_datetime('2022-08-15') - training_time
    elif 'GPT-3.5' in row['Model']:
        training_start_date = pd.to_datetime('2022-03-14')
    elif 'GPT-3' in row['Model']:
        training_start_date = pd.to_datetime('2019-10-01')
    else:
        if pd.isna(row['Training time (hours)']):
            if backup_training_time:
                print(f"No training time found, assuming {MEDIAN_TRAINING_TIME_DAYS} days\n")
                training_time = pd.Timedelta(days=MEDIAN_TRAINING_TIME_DAYS)
            else:
                print(f"No training time found\n")
                return None
        else:
            training_time = pd.Timedelta(hours=int(row['Training time (hours)']))
        # Can test different buffer times here: e.g. 20 days, 180 days
        buffer_time = pd.Timedelta(days=60)
        training_start_date = row['Publication date'] - (training_time + buffer_time)
    return training_start_date

def get_acquisition_date(row, backup_training_time=True):
    training_start_date = get_training_start_date(row, backup_training_time)
    if training_start_date is None:
        return None
    # Account for time between hardware acquisition and training start
    # TODO: test different buffer times: e.g. 30 days, 150 days
    buffer_time = pd.Timedelta(days=60)
    acquisition_date = training_start_date - buffer_time
    return acquisition_date

def select_vendor(row, org_to_cloud_vendor, default_vendor=True):
    orgs = row['Organization'].split(',')
    vendor = None
    for org in orgs:
        for key in org_to_cloud_vendor:
            if key in org.lower():
                vendor = org_to_cloud_vendor[key]
                break
    if default_vendor and vendor is None:
        # TODO: choose vendor based on cheapest price available?
        vendor = 'Amazon Web Services'  # default
    return vendor

def find_price_for_vendor_and_hardware_model(
    closest_price_dates_df,
    acquisition_date,
    price_colname,
    price_date_after=True,
):
    price_per_chip_hour = None
    for i, price_row in closest_price_dates_df.iterrows():
        if price_row['Price date'] <= acquisition_date:
            price_per_chip_hour = price_row[price_colname]
            price_id = price_row['Price source']
            price_date = price_row['Price date']
            break
    if price_per_chip_hour is None:
        for i, price_row in closest_price_dates_df.iterrows():
            if price_date_after and price_row['Price date'] > acquisition_date:
                price_per_chip_hour = price_row[price_colname]
                price_id = price_row['Price source']
                price_date = price_row['Price date']
                break
    if not pd.isna(price_per_chip_hour):
        return float(price_per_chip_hour), price_id, price_date
    else:
        print(f"Could not find price")
        print()
        return None, None, None

def apply_cud(price_per_chip_hour, vendor, price_type, default_price_type):
    adjusted_price_per_chip_hour = price_per_chip_hour
    if price_type != default_price_type:
        default_cud = DEFAULT_CUD[vendor][default_price_type]
        current_cud = DEFAULT_CUD[vendor][price_type]
        cud_ratio = (1 - default_cud) / (1 - current_cud)
        adjusted_price_per_chip_hour = price_per_chip_hour * cud_ratio
        print(f"Applying CUD: {price_per_chip_hour} * {cud_ratio} = {adjusted_price_per_chip_hour}")
    return adjusted_price_per_chip_hour

def find_price(
    row,
    price_df,
    hardware_df,
    pcd_hardware_model_colname,
    price_colname,
    org_to_cloud_vendor,
    price_date_after=True,
    default_vendor=True,
    backup_vendor=True,
    backup_price_type=True,
    backup_training_time=True,
):
    acquisition_date = get_acquisition_date(row, backup_training_time)
    if acquisition_date is None:
        return None, None
    hardware_model = row[pcd_hardware_model_colname]
    if pd.isna(hardware_model):
        print(f"Could not find hardware model for {row['Model']}\n")
        print()
        return None, None
    
    vendor = select_vendor(row, org_to_cloud_vendor, default_vendor=default_vendor)
    if vendor is None:
        print(f"Could not find vendor for {row['Model']}\n")
        print()
        return None, None
    print(f"Trying {hardware_model} at {acquisition_date}")

    possible_vendors = PRIORITY_VENDORS
    possible_vendors.extend([v for v in price_df['Vendor'].dropna().unique() if v not in PRIORITY_VENDORS])

    # Find the price of the hardware at the time of acquisition
    vendors = [vendor]
    if "TPU" not in hardware_model:
        # TPUs are only available from Google Cloud
        for possible_vendor in possible_vendors:
            if possible_vendor != vendor:
                # Means that we try the selected vendor first, then the other vendors
                vendors.append(possible_vendor)
    elif vendor != 'Google Cloud':
        # Means the hardware is a TPU but the cloud provider is not Google Cloud
        # This can happen if the hardware is an imputed value
        # This is not a good result for imputation, but we need to handle it
        vendor = 'Google Cloud'
        vendors = [vendor]

    price_types = [price_colname]
    for possible_price_type in ['Price per chip-hour (3-year CUD)', 'Price per chip-hour (1-year CUD)', 'Price per chip-hour (on-demand)']:
        if possible_price_type != price_colname:
            # Means that we try the default price type first, then the other types
            price_types.append(possible_price_type)
    
    for price_type in price_types:
        for vendor in vendors:
            print(f"Trying {vendor}, {price_type}")
            closest_price_dates_df = find_closest_price_dates(
                hardware_model, acquisition_date, price_df, vendor,
            )
            # TODO: is it better to try a different vendor before a different date?
            price_value, price_id, price_date = find_price_for_vendor_and_hardware_model(
                closest_price_dates_df, 
                acquisition_date,
                price_type,
                price_date_after=price_date_after,
            )
            if price_value is not None:
                if backup_price_type:
                    price_value = apply_cud(price_value, vendor, price_type, price_colname)
                print(f"Found price: {price_value} at {price_date}")
                print("Difference between acquisition date and price date:", acquisition_date - price_date, "\n")
                break
            # else: try again with a different vendor, price type
            if not backup_vendor:
                # Only do the first iteration
                break
        if not backup_price_type:
            # Only do the first iteration
            break
        if price_value is not None:
            break

    return price_value, price_id

def find_gpu_acquisition_price(price_df, hardware_model, price_colname):
    # Use a single alias (e.g. 'A100') to match hardware variants
    gpu_hardware_alias = None
    for alias in GPU_HARWARE_ALIASES:
        if alias in hardware_model:
            gpu_hardware_alias = alias
            break
    if gpu_hardware_alias is None:
        print(f"Could not find alias for {hardware_model}")
        return None

    # Sort price_df by date - we want the earliest price
    price_df = price_df.sort_values(by='Price date').dropna(subset=[price_colname])
    # Search for the best price closest to release date    
    matching_prices = price_df[price_df['Hardware model'].str.contains(gpu_hardware_alias)]
    chosen_price_row = None
    for _, price_row in matching_prices.iterrows():
        if 'DGX' in price_row['Notes']:
            chosen_price_row = price_row
            break
    if chosen_price_row is None:
        # Take the earliest price regardless of 'DGX' in the name
        chosen_price_row = matching_prices.iloc[0]
    
    if not pd.isna(chosen_price_row[price_colname]):
        return chosen_price_row
    else:
        print(f"Could not find price for {hardware_model}\n")
        return None

def find_TPU_equivalent_acquisition_price(hardware_model):
    price_value = TPU_EQUIVALENT_RELEASE_PRICES.get(hardware_model)
    if price_value is None:
        print(f"Could not find price for {hardware_model}\n")
        return None, None
    return price_value

def depreciate_by_hardware_progress(initial_price, start_date, end_date):
    """
    Depreciate the price of hardware over time by the hardware price-performance trend,
    using the ML_GPU_PRICE_PERFORMANCE_OOMS_PER_YEAR parameter.
    """
    years_since = (end_date - pd.to_datetime(start_date)).days / DAYS_PER_YEAR
    # Depreciate the value by hardware price-performance trend over the time period
    depreciation = ML_GPU_PRICE_PERFORMANCE_OOMS_PER_YEAR * years_since
    price_value_ooms = np.log10(initial_price) - depreciation
    end_price = 10 ** price_value_ooms
    return end_price

def find_hardware_acquisition_price(
    row,
    price_df,
    hardware_df,
    pcd_hardware_model_colname,
    price_colname,
):
    hardware_model = row[pcd_hardware_model_colname]
    if pd.isna(hardware_model):
        print(f"Could not find hardware model for {row['Model']}\n")
        return [None] * 4
    if ',' in hardware_model: 
        # comma indicates multiple types of hardware, which we don't handle
        print(f"Skipping {hardware_model}\n")
        return [None] * 4
    if "TPU" in hardware_model:
        price_id = None
        price_value = find_TPU_equivalent_acquisition_price(hardware_model)
        # Adjust single-unit prices for additional equipment e.g. CPU, intra-node interconnect
        price_value *= get_server_cost_overhead(hardware_model)
        # Assume the TPU was acquired at the public release date
        acquisition_date = get_release_date(hardware_model, hardware_df)
    else:
        chosen_price_row = find_gpu_acquisition_price(price_df, hardware_model, price_colname)
        if chosen_price_row is None:
            return [None] * 4
        price_id = chosen_price_row['Price source']
        price_value = chosen_price_row[price_colname]
        price_date = chosen_price_row['Price date']
        release_date = get_release_date(hardware_model, hardware_df)
        # Can test different buffer times here: e.g. 0 days, 180 days
        buffer_time = pd.Timedelta(days=90)
        if price_date < release_date + buffer_time:
            # Assume at least 3 months between release and when someone first acquired it
            acquisition_date = release_date + buffer_time
        else:
            acquisition_date = price_date
        # Adjust single-unit prices for additional equipment e.g. CPU, intra-node interconnect
        if 'single-unit' in chosen_price_row['Notes'].lower():
            price_value *= get_server_cost_overhead(hardware_model)
    return price_value, price_id, acquisition_date, hardware_model

def get_hardware_acquisition_price(
    row,
    price_df,
    hardware_df,
    pcd_hardware_model_colname,
    price_colname,
):
    price_value, price_id, acquisition_date, hardware_model = find_hardware_acquisition_price(
        row, price_df, hardware_df, pcd_hardware_model_colname, price_colname
    )
    if price_value is None:
        return None, None
    print(
        f"Estimated the value of {hardware_model} server, " +
        f"available from {acquisition_date}: {price_value} per chip\n"
    )
    return price_value, price_id

def get_hardware_value_at_training_start(
    row,
    price_df,
    hardware_df,
    pcd_hardware_model_colname,
    price_colname,
    backup_training_time=True,
):
    price_value, price_id, acquisition_date, hardware_model = find_hardware_acquisition_price(
        row, price_df, hardware_df, pcd_hardware_model_colname, price_colname
    )
    if price_value is None:
        return None, None
    # Depreciate the price due to hardware progress since being acquired
    training_start_date = get_training_start_date(row, backup_training_time)
    if 'TPU' not in hardware_model and training_start_date < acquisition_date:
        # For TPUs, training could have started before public availability
        # But for GPUs, training can only start after acquisition
        training_start_date = acquisition_date
    price_value = depreciate_by_hardware_progress(
        price_value, acquisition_date, training_start_date
    )
    print(
        f"Estimated the value of {hardware_model} server, available from {acquisition_date} " +
        f"and used from {training_start_date}: {price_value} per chip\n"
    )
    return price_value, price_id

# ==============================================================================
# INFLATION ADJUSTMENT
# ==============================================================================

def adjust_value_for_inflation(row, cost_colname, price_index, to_year_month):
    if pd.isna(row[cost_colname]):
        return row[cost_colname]
    
    from_date = str(row['Publication date'])
    from_year_month = from_date.rsplit('-', maxsplit=1)[0] + '-01'
    from_price_index = price_index[price_index['observation_date'] == from_year_month]['PCU518210518210'].values[0]
    to_price_index = price_index[price_index['observation_date'] == to_year_month]['PCU518210518210'].values[0]
    adjust_factor = to_price_index / from_price_index
    return row[cost_colname] * adjust_factor

def adjust_column_for_inflation(df, cost_colname, path_to_price_index, to_year_month):
    price_index = pd.read_csv(path_to_price_index)
    df[cost_colname + ' (inflation-adjusted)'] = df.apply(
        adjust_value_for_inflation, axis=1, args=(cost_colname, price_index, to_year_month)
    )
    return df

# ==============================================================================
# IMPUTATION FUNCTIONS
# ==============================================================================

def get_one_hot_df(df):
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # one-hot encode all categorical columns
    one_hot_df = pd.get_dummies(df, columns=categorical_cols)
    return one_hot_df

def knn_impute_categorical_column(dataframe, target_col, num_neighbors=5):
    """
    Use `KNeighborsClassifier` to impute the missing values in-place in `target_col`.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder
    
    # Separate the target and features
    full_features = dataframe.drop(target_col, axis=1)
    full_target = dataframe[target_col]

    # Remove rows with missing target
    missing_target_mask = full_target.isna()
    features = full_features[~missing_target_mask]
    target = full_target[~missing_target_mask]

    # Encode the target column
    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target)

    # Train a KNeighborsClassifier
    knc = KNeighborsClassifier(n_neighbors=num_neighbors)
    knc.fit(features, target_encoded)

    # Predict the missing values
    features_with_missing_target = full_features[missing_target_mask]
    predicted = knc.predict(features_with_missing_target)

    # Decode the predictions
    predicted_labels = label_encoder.inverse_transform(predicted)

    # Replace the missing values with the predictions
    dataframe.loc[missing_target_mask, target_col] = predicted_labels

    return dataframe

def knn_impute_numerical_pcd_data(one_hot_pcd_df, num_neighbors=5):
    from sklearn.impute import KNNImputer
    
    # instantiate the imputer
    imputer = KNNImputer(n_neighbors=num_neighbors)

    # impute the missing values in Training hardware, hardware quantity, Training time (hours)
    imputed = imputer.fit_transform(one_hot_pcd_df)

    # convert the numpy array back to a dataframe
    imputed_pcd_df = pd.DataFrame(imputed, columns=one_hot_pcd_df.columns)

    # convert Training hardware back to categorical
    imputed_pcd_df['Training hardware'] = ''
    for col in imputed_pcd_df.columns:
        if col.startswith('Training hardware_'):
            training_hardware = col.split('Training hardware_')[1]
            imputed_pcd_df['Training hardware'] = imputed_pcd_df['Training hardware'] + pd.Series([int(_) * training_hardware for _ in imputed_pcd_df[col]])
    imputed_pcd_df['Training hardware'].replace('', np.nan, inplace=True)

    return imputed_pcd_df

def knn_impute_pcd(pcd_df, num_neighbors=5):
    # Use k nearest neighbors
    # drop unneeded columns from pcd_df
    irrelevant_columns = ['Notability criteria', 'Notability criteria notes', 'Link', 'Citations', 'Parameters notes',
                        'Training compute notes', 'Training dataset notes', 'Dataset size notes',
                        'Inference compute notes', 'Approach', 'Confidence', 'Last modified', 'Created By', 'Benchmark data',
                        'Exclude', 'Authors by country', 'Training cost trends', 'Abstract', 'Compute cost notes',
                        'Training time notes', 'Authors',
                        'Training compute cost (2020 USD)', 'Organization categorization',
                        'Training dataset', 'Inference compute (FLOP)', 'Compute sponsor categorization',
                        'Finetune compute notes']
    pcd_df = pcd_df.drop(columns=irrelevant_columns)
    
    # fill column 'Training cloud compute vendor' using org_to_cloud_vendor dictionary
    org_to_cloud_vendor = {
        'Google': 'Google Cloud',
        'DeepMind': 'Google Cloud',
        'Google DeepMind': 'Google Cloud',
        'Google Brain': 'Google Cloud',
        'Microsoft': 'Microsoft Azure',
        'OpenAI': 'Microsoft Azure',
    }
    pcd_df['Training cloud compute vendor'] = pcd_df['Organization'].map(org_to_cloud_vendor)
    pcd_df['Training cloud compute vendor'] = pcd_df['Training cloud compute vendor'].fillna('Amazon Web Services')

    # convert large number columns to logarithmic
    parameters_col = pcd_df['Parameters']
    training_compute_col = pcd_df['Training compute (FLOP)']
    dataset_size_col = pcd_df['Training dataset size (datapoints)']
    pcd_df['log_params'] = np.log10(parameters_col)
    pcd_df['log_compute'] = np.log10(training_compute_col)
    pcd_df['log_dataset'] = np.log10(dataset_size_col)
    # drop raw columns
    raw_columns = ['Parameters', 'Training compute (FLOP)', 'Training dataset size (datapoints)']
    pcd_df.drop(columns=raw_columns, inplace=True)

    # convert datetime to float so that it can be used in kNN
    pcd_df['Publication date'] = datetime_to_float_year(pcd_df['Publication date'])

    # set the System column as the index for formatting purposes
    pcd_df = pcd_df.set_index('Model')
    one_hot_pcd_df = get_one_hot_df(pcd_df)
    imputed_pcd_df = knn_impute_numerical_pcd_data(one_hot_pcd_df, num_neighbors=num_neighbors)

    # Impute training hardware separately, because it is a categorical variable
    imputed_pcd_df = knn_impute_categorical_column(
        imputed_pcd_df,
        num_neighbors=num_neighbors,
        target_col='Training hardware'
    )

    # Restore the System column
    imputed_pcd_df['Model'] = pcd_df.index

    # set the System column as the index
    imputed_pcd_df = imputed_pcd_df.set_index('Model')

    # insert imputed values into pcd_df
    pcd_df['Training hardware'] = imputed_pcd_df['Training hardware']
    pcd_df['Hardware quantity'] = imputed_pcd_df['Hardware quantity']
    pcd_df['Hardware utilization'] = imputed_pcd_df['Hardware utilization']
    pcd_df['Training time (hours)'] = imputed_pcd_df['Training time (hours)']
    # calculate training time (chip hours) from training time and hardware quantity
    pcd_df['Training time (chip hours)'] = pcd_df['Training time (hours)'] * pcd_df['Hardware quantity']
    # Restore columns that were dropped
    pcd_df['Parameters'] = parameters_col
    pcd_df['Training compute (FLOP)'] = training_compute_col
    pcd_df['Training dataset size (datapoints)'] = dataset_size_col

    assert all(pcd_df['Training time (chip hours)'].notna())

    pcd_df['Model'] = pcd_df.index
    # Imputation converted datetime to float
    # Need to convert back to datetime
    pcd_df['Publication date'] = pcd_df['Publication date'].apply(float_year_to_datetime)

    return pcd_df

def most_common_impute(dataframe, target_col, time_col):
    """
    Impute the missing values in-place in `target_col` with the most common value for each year in `time_col`.
    Assumes `time_col` represents dates as a fractional year.
    """
    times = dataframe[time_col]
    # Time is a float year e.g. 2017.2. We want to group by the integer year e.g. 2017
    years = times.apply(int)

    def get_most_common_target(group):
        # Some values are multiple values separated by a comma
        # We want to split these and count each value separately
        split_values = group.str.split(',').explode()
        return split_values.mode().values[0]

    # Group by year
    # For hardware models, we also want to group Google TPU and GPUs separately
    if target_col == 'Training hardware':
        groups = dataframe.dropna(subset=[target_col]).groupby(
            [dataframe['Training hardware'].str.contains('TPU'), years]
        )
        grouped_targets = groups[target_col]

        # Impute the missing values with the most common value for each year
        most_common_targets = grouped_targets.apply(get_most_common_target)
        for (is_tpu, year), most_common_target in most_common_targets.items():
            mask = (years == year) & ((dataframe['Training hardware'].str.contains('TPU') == is_tpu) | dataframe['Training hardware'].isna())
            dataframe.loc[mask, target_col] = dataframe.loc[mask, target_col].fillna(most_common_target)
    else:
        grouped_targets = dataframe.dropna(subset=[target_col]).groupby(years)[target_col]

        # Impute the missing values with the most common value for each year
        most_common_targets = grouped_targets.apply(get_most_common_target)
        for year, most_common_target in most_common_targets.items():
            mask = years == year
            dataframe.loc[mask, target_col] = dataframe.loc[mask, target_col].fillna(most_common_target)

    return dataframe

def most_common_impute_training_hardware(pcd_df):
    """
    Impute the missing values in the `Training hardware` of `pcd_df` with the most common value
    for each year in the full PCD data.
    """
    # Load full PCD data to get as much data as possible
    full_pcd_df = load_pcd_df()

    # Publication date in datetime format
    full_pcd_df.dropna(subset=['Publication date'], inplace=True)
    full_pcd_df['Publication date'] = pd.to_datetime(full_pcd_df['Publication date'])
    full_pcd_df['Publication date'] = datetime_to_float_year(full_pcd_df['Publication date'])

    # Impute missing values in Training hardware
    imputed_pcd_df = most_common_impute(full_pcd_df, 'Training hardware', 'Publication date')

    frontier_systems = load_frontier_systems()
    pcd_df.loc[:, 'Training hardware'] = imputed_pcd_df.loc[
        imputed_pcd_df['Model'].isin(frontier_systems), 'Training hardware'
    ]

    # TODO: probably want to move this part one level up in the functions, like `knn_impute_pcd`
    for _, row in pcd_df.iterrows():
        if not(pd.isna(row['Training time (hours)']) or pd.isna(row['Hardware quantity'])):
            pcd_df['Training time (chip hours)'] = pcd_df['Training time (hours)'] * pcd_df['Hardware quantity']

    return pcd_df

# ==============================================================================
# COST ESTIMATION FUNCTIONS
# ==============================================================================

def estimate_chip_hours(row, hardware_df):
    if not pd.isna(row['Training chip-hours']):
        return row['Training chip-hours']
    hardware_quantity = row['Hardware quantity']
    training_time = row['Training time (hours)']
    if any([np.isnan(x) for x in [hardware_quantity, training_time]]):
        flop = row['Training compute (FLOP)']
        hardware_model = row['Training hardware']
        if not any([pd.isna(x) for x in [flop, hardware_model]]):
            print("Imputing training time from compute and hardware")
            flop_per_second = get_flop_per_second(hardware_model, hardware_df)
            flop_utilization = row['Hardware utilization']
            if pd.isna(flop_utilization):
                flop_utilization = MEDIAN_UTILIZATION

            training_chip_seconds = flop / (flop_per_second * flop_utilization)
            training_chip_hours = training_chip_seconds / SECONDS_PER_HOUR
        else:
            return None
    else:
        training_chip_hours = training_time * hardware_quantity
    return training_chip_hours

def cluster_energy_cost(hardware_model, total_chip_hours, hardware_df, organization, year):
    """
    hardware_model: name of the hardware used for the training run
    total_chip_hours: total number of chip-hours used for the training run (i.e. number of chips * training time)
    hardware_df: DataFrame containing hardware specs
    organization: name of the organization who did the training
    year: year in which the training run was conducted
    """
    matching_hardware = hardware_df[hardware_df['Hardware name'] == hardware_model]
    chip_TDP_kw = matching_hardware['TDP (W)'].squeeze() / 1000
    if pd.isna(chip_TDP_kw):
        if "TPU v4" in hardware_model:
            chip_TDP_kw = 373 / 1000
        else:
            print("Unable to estimate chip TDP")
            return None
    # Adjust for whole server power draw (CPUs, memory, cooling)
    server_TDP_kw = chip_TDP_kw * chip_to_server_power(hardware_model)
    # Adjust for average power draw
    server_power_kw = server_TDP_kw * server_TDP_fraction(hardware_model)
    # Adjust for data center power distribution and cooling
    adj_server_power_kw = server_power_kw * power_usage_effectiveness(organization)
    cluster_kwh = adj_server_power_kw * total_chip_hours
    return cluster_kwh * energy_price(year)

def estimate_hardware_capex_energy_cost(
    row, system_to_price, frontier_pcd_df, hardware_df, separate_components=False,
):
    system = row['Model']
    price = system_to_price.get(system)
    if price is None:
        return None

    hardware_model = row['Training hardware']

    training_chip_hours = estimate_chip_hours(row, hardware_df)

    # Hardware progress rate k OOMs/year 
    # => optimal to replace k * np.log(10) per year
    # See https://epochai.org/blog/the-longest-training-run#a-simple-framework-for-training-run-lengths
    hardware_replacement_per_year = ML_GPU_PRICE_PERFORMANCE_OOMS_PER_YEAR * np.log(10)
    price_per_chip_hour = price * hardware_replacement_per_year / HOURS_PER_YEAR
    amortized_hardware_cost = price_per_chip_hour * training_chip_hours

    interconnect_cost = amortized_hardware_cost * CLUSTER_INTERCONNECT_COST_FRACTION / (1 - CLUSTER_INTERCONNECT_COST_FRACTION)

    org = row['Organization']
    pub_year = row['Publication date'].year
    energy_cost = cluster_energy_cost(
        hardware_model, training_chip_hours, hardware_df, org, pub_year,
    )
    if separate_components:
        ai_chip_cost = amortized_hardware_cost / get_server_cost_overhead(hardware_model)
        extra_server_cost = amortized_hardware_cost - ai_chip_cost
        cost = {
            'AI accelerator chip cost': ai_chip_cost,
            'Other server components cost': extra_server_cost,
            'Cluster-level interconnect cost': interconnect_cost,
            'Energy cost': energy_cost,
        }
    else:
        cost = amortized_hardware_cost + interconnect_cost + energy_cost
        # Useful for comparing to cloud prices
        overall_cost_per_chip_hour = cost / training_chip_hours
        print(f"Overall cost per chip-hour for {hardware_model}:", overall_cost_per_chip_hour)

    # Check for base model
    if not pd.isna(row['Base model']):
        print(f"Skipping {system} because it is a fine-tuned version of a base model")
        return None

    return cost

def estimate_hardware_capex_energy(
    frontier_pcd_df,
    hardware_df,
    price_df,
    separate_components=False,
    impute_pcd_fn=None,
    **impute_kwargs,
):
    """
    Full pipeline for estimating the sum of amortized hardware CapEx and energy cost
    """
    if impute_pcd_fn is not None:
        frontier_pcd_df = impute_pcd_fn(frontier_pcd_df, **impute_kwargs)
    
    pcd_hardware_model_colname = 'Training hardware'
    price_colname = 'Price (hardware purchase)'
    system_to_price = {}

    for i, row in frontier_pcd_df.iterrows():
        print_safely(f"==== System: {row['Model']} ====")
        price, _ = get_hardware_value_at_training_start(
            row, price_df, hardware_df, pcd_hardware_model_colname, price_colname
        )
        if price is None:
            continue
        else:
            system_to_price[row['Model']] = price
        
    system_to_cost = {}
    for i, row in frontier_pcd_df.iterrows():
        print_safely(f"==== System: {row['Model']} ====")
        cost = estimate_hardware_capex_energy_cost(
            row, system_to_price, frontier_pcd_df, hardware_df, separate_components
        )
        if cost is None:
            print("Unable to estimate cost")
            continue
        else:
            print("Estimated cost:", cost)
        system_to_cost[row['Model']] = cost

    print("All costs:")
    print(system_to_cost)

    if separate_components:
        # Assign a new column in frontier_pcd_df for each cost component
        cost_component_names = [
            'AI accelerator chip cost',
            'Other server components cost',
            'Cluster-level interconnect cost',
            'Energy cost',
        ]
        for k in cost_component_names:
            system_to_component_cost = {
                system: system_to_cost[system][k] for system in system_to_cost
            }
            frontier_pcd_df[k] = frontier_pcd_df['Model'].map(system_to_component_cost)
        frontier_pcd_df['Cost'] = frontier_pcd_df[cost_component_names].sum(axis=1)
    else:
        frontier_pcd_df['Cost'] = frontier_pcd_df['Model'].map(system_to_cost)

    return frontier_pcd_df

def estimate_hardware_acquisition_cost(
    frontier_pcd_df,
    hardware_df,
    price_df,
    impute_pcd_fn=None,
    **impute_kwargs,
):
    """
    Full pipeline for estimating up-front server capex
    """
    print("Estimating up-front server capex")
    if impute_pcd_fn is not None:
        frontier_pcd_df = impute_pcd_fn(frontier_pcd_df, **impute_kwargs)
    
    pcd_hardware_model_colname = 'Training hardware'
    price_colname = 'Price (hardware purchase)'
    system_to_price = {}

    for i, row in frontier_pcd_df.iterrows():
        print_safely(f"==== System: {row['Model']} ====")
        price, _ = get_hardware_acquisition_price(
            row, price_df, hardware_df, pcd_hardware_model_colname, price_colname
        )
        if price is None:
            continue
        else:
            system_to_price[row['Model']] = price

    # Cost estimation
    def estimate_cost(row, system_to_price):
        system = row['Model']
        price = system_to_price.get(system)
        if price is None:
            print("No hardware price found")
            return None
        
        hardware_quantity = row['Hardware quantity']
        if pd.isna(hardware_quantity):
            print("No hardware quantity found")
            return None
        cost = hardware_quantity * price
        # Add interconnect cost
        cost *= 1 / (1 - CLUSTER_INTERCONNECT_COST_FRACTION)

        # Check for base model
        if not pd.isna(row['Base model']):
            print(f"Skipping {system} because it is a fine-tuned version of a base model")
            return None

        return cost
        
    system_to_cost = {}
    for i, row in frontier_pcd_df.iterrows():
        print_safely(f"==== System: {row['Model']} ====")
        cost = estimate_cost(row, system_to_price)
        if cost is None:
            print("Unable to estimate cost")
            continue
        else:
            print("Estimated cost:", cost)
        system_to_cost[row['Model']] = cost

    print(system_to_cost)

    frontier_pcd_df['Cost'] = frontier_pcd_df['Model'].map(system_to_cost)

    return frontier_pcd_df

def estimate_cloud_costs(
    frontier_pcd_df,
    hardware_df,
    price_df,
    impute_pcd_fn=None,
    **impute_kwargs,
):
    """
    Full cost estimation pipeline
    """
    if impute_pcd_fn is not None:
        frontier_pcd_df = impute_pcd_fn(frontier_pcd_df, **impute_kwargs)
    
    # TODO: centralize vendor mapping to reduce repetition
    org_to_cloud_vendor = {
        'google': 'Google Cloud',
        'deepmind': 'Google Cloud',
        'microsoft': 'Microsoft Azure',
        'openai': 'Microsoft Azure',
    }

    pcd_hardware_model_colname = 'Training hardware'
    price_colname = 'Price per chip-hour (3-year CUD)'
    system_to_price = {}

    for i, row in frontier_pcd_df.iterrows():
        print_safely(f"==== System: {row['Model']} ====")
        price, _ = find_price(row, price_df, hardware_df, pcd_hardware_model_colname, price_colname, org_to_cloud_vendor)
        if price is None:
            continue
        else:
            system_to_price[row['Model']] = price

    # Cost estimation
    def estimate_cost(row, system_to_price):
        system = row['Model']
        price = system_to_price.get(system)
        if price is None:
            return None

        if 'Training time (chip hours)' in row.index:
            chip_hours = row['Training time (chip hours)']
        else:
            chip_hours = estimate_chip_hours(row, hardware_df)
        if np.isnan(chip_hours):
            print("Unable to estimate chip hours")
            return None

        cost = price * chip_hours

        # Check for base model
        if not pd.isna(row['Base model']):
            print(f"Skipping {system} because it is a fine-tuned version of a base model")
            return None

        return cost
        
    system_to_cost = {}
    for i, row in frontier_pcd_df.iterrows():
        print_safely(f"==== System: {row['Model']} ====")
        cost = estimate_cost(row, system_to_price)
        if cost is None:
            print("Unable to estimate cost")
            continue
        else:
            print("Estimated cost:", cost)
        system_to_cost[row['Model']] = cost

    print("All costs:")
    print(system_to_cost)

    frontier_pcd_df['Cost'] = frontier_pcd_df['Model'].map(system_to_cost)

    return frontier_pcd_df

# ==============================================================================
# MAIN COST ANALYSIS WORKFLOW
# ==============================================================================

def main():
    """
    Main function that runs the cost analysis workflow
    """
    # Configuration
    compute_threshold_method = 'top_n'  # top_n, window_percentile
    compute_threshold = 10  # e.g. 10 to select top 10; 75 to select top 25%
    variant = '2025-03-17_exclude_finetunes_at_threshold_stage'  # whatever else distinguishes this run
    exclude_models_containing = []  # ['GNMT', 'AlphaZero', 'AlphaGo Master', 'AlphaGo Zero']

    # Imputation configuration
    enable_imputation = True  # Set to False to disable imputation
    imputation_method = 'most_common'  # 'knn', 'most_common', 'none'
    knn_neighbors = 5  # Number of neighbors for KNN imputation (if using KNN)

    # Run all three cost estimation methods
    estimation_methods = ['hardware-capex-energy', 'hardware-acquisition', 'cloud']
    estimation_method_lookup = {
        'hardware-capex-energy': estimate_hardware_capex_energy,
        'hardware-acquisition': estimate_hardware_acquisition_cost,
        'cloud': estimate_cloud_costs,
    }

    results_dir = f'results/all-methods-{compute_threshold_method}={compute_threshold}-{variant}/'
    os.makedirs(results_dir, exist_ok=True)

    print("Loading data...")
    frontier_pcd_df, hardware_df, price_df = load_data_for_cost_estimation(
        compute_threshold_method=compute_threshold_method, compute_threshold=compute_threshold,
    )

    print(f"Loaded {len(frontier_pcd_df)} frontier models, {len(hardware_df)} hardware entries, {len(price_df)} price entries")

    # Data quality report before imputation
    print("Data Quality Report (Before Imputation):")
    print(f"Models with known Training hardware: {frontier_pcd_df['Training hardware'].notna().sum()}/{len(frontier_pcd_df)}")
    print(f"Models with known Hardware quantity: {frontier_pcd_df['Hardware quantity'].notna().sum()}/{len(frontier_pcd_df)}")
    print(f"Models with known Hardware utilization: {frontier_pcd_df['Hardware utilization'].notna().sum()}/{len(frontier_pcd_df)}")
    print(f"Models with known Training time (hours): {frontier_pcd_df['Training time (hours)'].notna().sum()}/{len(frontier_pcd_df)}")

    # Apply imputation if enabled
    if enable_imputation and imputation_method != 'none':
        print(f"\nApplying {imputation_method} imputation...")
        if imputation_method == 'knn':
            # Apply KNN imputation
            frontier_pcd_df = knn_impute_pcd(frontier_pcd_df.copy(), num_neighbors=knn_neighbors)
            print(f"Applied KNN imputation with {knn_neighbors} neighbors")
        elif imputation_method == 'most_common':
            # Apply most common value imputation for training hardware
            frontier_pcd_df = most_common_impute_training_hardware(frontier_pcd_df.copy())
            print("Applied most common value imputation for training hardware")
        
        # Data quality report after imputation
        print("\nData Quality Report (After Imputation):")
        print(f"Models with known Training hardware: {frontier_pcd_df['Training hardware'].notna().sum()}/{len(frontier_pcd_df)}")
        print(f"Models with known Hardware quantity: {frontier_pcd_df['Hardware quantity'].notna().sum()}/{len(frontier_pcd_df)}")
        print(f"Models with known Hardware utilization: {frontier_pcd_df['Hardware utilization'].notna().sum()}/{len(frontier_pcd_df)}")
        print(f"Models with known Training time (hours): {frontier_pcd_df['Training time (hours)'].notna().sum()}/{len(frontier_pcd_df)}")
    else:
        print("\nSkipping imputation (disabled in configuration)")

    # Determine imputation function based on configuration
    if enable_imputation and imputation_method != 'none':
        if imputation_method == 'knn':
            impute_pcd_fn = knn_impute_pcd
            impute_kwargs = {'num_neighbors': knn_neighbors}
        elif imputation_method == 'most_common':
            impute_pcd_fn = most_common_impute_training_hardware
            impute_kwargs = {}
        else:
            impute_pcd_fn = None
            impute_kwargs = {}
    else:
        impute_pcd_fn = None
        impute_kwargs = {}

    # Run all three cost estimation methods
    cost_dfs = {}
    component_cost_df = None

    for estimation_method in estimation_methods:
        print(f"\n=== Running {estimation_method} estimation ===")
        cost_estimation_function = estimation_method_lookup[estimation_method]
        
        with open(f'{results_dir}/cost_estimation_{estimation_method}.out', 'w') as f:
            with redirect_stdout(f):
                if impute_pcd_fn is not None:
                    # Call with imputation parameters
                    cost_df = cost_estimation_function(
                        frontier_pcd_df.copy(), hardware_df, price_df,
                        impute_pcd_fn=impute_pcd_fn, **impute_kwargs
                    )
                else:
                    # Call without imputation
                    cost_df = cost_estimation_function(frontier_pcd_df.copy(), hardware_df, price_df)
        
        cost_dfs[estimation_method] = cost_df
        
        # Create component cost breakdown only for hardware-capex-energy method
        if estimation_method == 'hardware-capex-energy':
            frontier_pcd_df_copy = frontier_pcd_df.copy()
            with open(f'{results_dir}/component_cost_estimation.out', 'w') as f:
                with redirect_stdout(f):
                    if impute_pcd_fn is not None:
                        component_cost_df = cost_estimation_function(
                            frontier_pcd_df_copy, hardware_df, price_df,
                            separate_components=True, impute_pcd_fn=impute_pcd_fn, **impute_kwargs
                        )
                    else:
                        component_cost_df = cost_estimation_function(
                            frontier_pcd_df_copy, hardware_df, price_df, separate_components=True
                        )

    print(f"\nCost estimation completed for all methods")

    # Display results for each method
    for method, df in cost_dfs.items():
        print(f"\n=== {method} results ===")
        print(f"Total models: {len(df)}")
        print(f"Models with cost estimates: {df['Cost'].notna().sum()}")
        if 'Training time (hours)' in df.columns:
            print(f"Models with training time: {df.dropna(subset=['Cost'])['Training time (hours)'].notna().sum()}")
        if 'Hardware utilization' in df.columns:
            print(f"Models with hardware utilization: {df.dropna(subset=['Cost'])['Hardware utilization'].notna().sum()}")
        print(f"Cost range: ${df['Cost'].min():.0f} - ${df['Cost'].max():.0f}")

    # Apply exclusions to all cost dataframes
    for method in estimation_methods:
        for kw in exclude_models_containing:
            cost_dfs[method] = cost_dfs[method][cost_dfs[method]['Model'].str.contains(kw) == False]

    # Apply inflation adjustment to all cost dataframes
    for method in estimation_methods:
        cost_dfs[method] = adjust_column_for_inflation(cost_dfs[method], 'Cost', 'data/PCU518210518210.csv', '2025-07-01')

    # Create cost_dataset_3_estimates.csv with Model + 3 cost columns
    cost_comparison_df = pd.DataFrame()
    cost_comparison_df['Model'] = cost_dfs['hardware-capex-energy']['Model']

    # Add inflation-adjusted costs from each method
    for method in estimation_methods:
        method_df = cost_dfs[method]
        cost_comparison_df[f'{method.replace("-", "_")}_cost'] = method_df['Cost (inflation-adjusted)']

    # Display the comparison
    print("\nCost comparison across methods:")
    print(cost_comparison_df.dropna().head(10))

    # Save the 3-method comparison dataset
    cost_comparison_df.to_csv(results_dir + 'cost_dataset_3_estimates.csv', index=False)
    print(f"\nSaved cost_dataset_3_estimates.csv with {len(cost_comparison_df)} models")

    # Also keep the original detailed export for the hardware-capex-energy method
    cost_df = cost_dfs['hardware-capex-energy']
    keep_cols = [
        'Model',
        'Domain',
        'Task',
        'Model accessibility',
        'Reference',
        'Publication date',
        'Organization',
        'Parameters',
        'Training compute (FLOP)',
        'Training dataset size (datapoints)',
        'Epochs',
        'Training time (hours)',
        'Training hardware',
        'Base model',
        'Finetune compute (FLOP)',
        'Hardware quantity',
        'Hardware utilization',
        'Training cloud compute vendor',
        'Training data center',
        'Cost',
        'Cost (inflation-adjusted)',
    ]
    
    # Only keep columns that exist in the dataframe
    existing_cols = [col for col in keep_cols if col in cost_df.columns]
    cost_df[existing_cols].to_csv(results_dir + 'cost_dataset_detailed.csv', index=False)

    # Handle component costs if available
    if component_cost_df is not None:
        cost_component_names = [
            'AI accelerator chip cost',
            'Other server components cost',
            'Cluster-level interconnect cost',
            'Energy cost',
        ]
        
        for key in cost_component_names:
            if key in component_cost_df.columns:
                component_cost_df[f"{key} (%)"] = component_cost_df[key] / component_cost_df['Cost'] * 100
        
        cost_component_pc_names = [name + ' (%)' for name in cost_component_names]
        existing_pc_cols = [col for col in cost_component_pc_names if col in component_cost_df.columns]
        
        if existing_pc_cols:
            filtered_component_cost_df = component_cost_df.dropna(subset=existing_pc_cols).sort_values(by='Publication date')
            filtered_component_cost_df.to_csv(results_dir + 'cost_components.csv', index=False)
            
            # Average percentage for each component
            print("\nAverage component percentages:")
            print(filtered_component_cost_df[existing_pc_cols].mean())
            
            # Add power capacity calculation
            if 'Training hardware' in filtered_component_cost_df.columns and 'Hardware quantity' in filtered_component_cost_df.columns:
                filtered_component_cost_df = filtered_component_cost_df.dropna(subset=['Training hardware'])
                power_col = 'Power capacity for final training run (kW)'
                filtered_component_cost_df.loc[:, power_col] = [
                    cluster_power_capacity(row['Training hardware'], row['Hardware quantity'], hardware_df, row['Organization'])
                    for _, row in filtered_component_cost_df.iterrows()
                ]
                
                filtered_component_cost_df['Publication date (float)'] = datetime_to_float_year(
                    pd.to_datetime(filtered_component_cost_df['Publication date'])
                )

    print(f"\nCost analysis complete! Results saved to {results_dir}")

if __name__ == "__main__":
    main()