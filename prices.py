import numpy as np
import pandas as pd

from hardware import *
from parameters import *


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


def find_closest_price_dates(hardware_model, date, df, vendor=None, price_colname=None):
    """
    Finds the rows in the DataFrame with the closest 'Price date' to the given date and which match
    the hardware model, vendor (if provided), and price column (if provided).
    
    :param hardware_model: The hardware model to match.
    :param date: The target date to find the closest 'Price date' to.
    :param df: The DataFrame containing the hardware price data.
    :param vendor: The vendor to match, if applicable.
    :param price_colname: The price column to match, if applicable.
    :return: The rows from the DataFrame that match the criteria, in order of closest date.
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
    if 'Gemini Ultra' in row['System']:
        # https://blog.google/technology/ai/google-io-2023-keynote-sundar-pichai/#ai-responsibility
        # "This includes our next-generation foundation model, Gemini, which is still in training."
        # So might have been even earlier, but 
        training_start_date = pd.to_datetime('2023-05-10')
    elif 'GPT-4' in row['System']:
        # https://arxiv.org/abs/2303.08774
        # "This system card analyzes GPT-4 [...] Since it finished training in August of 2022 [...]"
        training_time = pd.Timedelta(hours=int(row['Training time (hours)']))
        training_start_date = pd.to_datetime('2022-08-15') - training_time
    elif 'GPT-3.5' in row['System']:
        # https://web.archive.org/web/20230314165432/https://openai.com/research/gpt-4
        # "A year [before March 14 2023], we trained GPT-3.5 as a first “test run” of the system."
        training_start_date = pd.to_datetime('2022-03-14')
    elif 'GPT-3' in row['System']:
        """
        1. Shevlane (2022)
        https://uploads-ssl.webflow.com/614b70a71b9f71c9c240c7a7/6262a1a55526a373cc93207d_Shevlane%20dissertation%20preprint.pdf
        p.66 of the PDF: A senior member of OpenAI (specified anonymously on p.27 of the PDF) 
        told the author "GPT-3 existed for a long time before the paper came out. We delayed the 
        paper. [...] But it's months, it doesn't really count."
        p.67 of the PDF: CAMERON said "Firstly, [the idea for a commercial API for GPT-3] 
        started out as a research API. It probably was . . . early January 2020."

        2. We think it plausibly could have been produced soon after the Microsoft deal was 
        announced in July 2019. Supposing the announcement coincided with Microsoft giving 
        OpenAI access to the necessary compute, and OpenAI already having almost everything 
        about GPT-3 planned in advance, and it took less than 1 month to train, then GPT-3 could 
        have been produced in August 2019.

        3. So we estimate August to January as our 90% CI, and halfway between (October) as the
        central estimate.
        """
        training_start_date = pd.to_datetime('2019-10-01')
    else:
        if pd.isna(row['Training time (hours)']):
            if backup_training_time:
                print(f"No training time found, assuming {MEDIAN_TRAINING_TIME_DAYS}\n")
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
        print(f"Could not find hardware model for {row['System']}\n")
        print()
        return None, None
    
    # Uncomment to test the effect of removing TPUs.
    # if 'TPU' in hardware_model:
    #     print(f"Skipping TPU {hardware_model}")
    #     return None, None

    vendor = select_vendor(row, org_to_cloud_vendor, default_vendor=default_vendor)
    if vendor is None:
        print(f"Could not find vendor for {row['System']}\n")
        print()
        return None, None
    print(f"Trying {hardware_model} at {acquisition_date}")

    # Find the price of the hardware at the time of acquisition
    vendors = [vendor]
    if "TPU" not in hardware_model:
        # TPUs are only available from Google Cloud
        for possible_vendor in ['Amazon Web Services', 'Microsoft Azure', 'Google Cloud', 'Lambda Labs']:
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
        return None, None

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
        return None, None
    

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
        print(f"Could not find hardware model for {row['System']}\n")
        return [None] * 4
    if ',' in hardware_model: 
        # comma indicates multiple types of hardware, which we don't handle
        print(f"Skipping {hardware_model}\n")
        return [None] * 4
    if "TPU" in hardware_model:
        # Uncomment to test the effect of removing TPUs.
        # print(f"Skipping TPU {hardware_model}")
        # return [None] * 4
    
        price_id = None
        price_value = find_TPU_equivalent_acquisition_price(hardware_model)
        # Adjust single-unit prices for additional equipment e.g. CPU, intra-node interconnect
        price_value *= get_server_cost_overhead(hardware_model)
        # Assume the TPU was acquired at the public release date
        acquisition_date = get_release_date(hardware_model, hardware_df)
    else:
        chosen_price_row = find_gpu_acquisition_price(price_df, hardware_model, price_colname)
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
