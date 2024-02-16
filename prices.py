import numpy as np
import pandas as pd

from parameters import *


def find_closest_price_dates(vendor, hardware_model, date, df):
    """
    Finds the row in the DataFrame with the closest 'Price date' to the given date
    which has a 'Vendor' equal to the specified vendor and a 'Hardware model' equal to the specified hardware_model.
    
    :param vendor: The vendor to match.
    :param hardware_model: The hardware model to match.
    :param date: The target date to find the closest 'Price date' to.
    :param df: The DataFrame containing the hardware price data.
    :return: The row from the DataFrame that matches the criteria.
    """
    # Filter the DataFrame based on vendor and hardware model
    filtered_df = df[(df['Vendor'] == vendor) & (df['Hardware model'] == hardware_model)]

    # Convert the target date to datetime
    target_date = pd.to_datetime(date)

    # Find the row with the smallest date difference
    closest_row_df = filtered_df.iloc[(filtered_df['Price date'] - target_date).abs().argsort()]

    return closest_row_df


def get_purchase_time(row):
    if pd.isna(row['Training time (hours)']):
        # TODO: fallback to using publication date and a default training time
        print("Training time is required but no value found")
        return None
    
    # Subtract training time plus 2 months from publication date
    training_time_offset = pd.Timedelta(hours=int(row['Training time (hours)']))
    low_buffer_time_offset = pd.Timedelta(days=30)
    mid_buffer_time_offset = pd.Timedelta(days=60)
    high_buffer_time_offset = pd.Timedelta(days=150)

    high_purchase_time = row['Publication date'] - (training_time_offset + low_buffer_time_offset)
    mid_purchase_time = row['Publication date'] - (training_time_offset + mid_buffer_time_offset)
    low_purchase_time = row['Publication date'] - (training_time_offset + high_buffer_time_offset)

    # TODO: sample from distribution of purchase times
    return mid_purchase_time


def select_vendor(row, org_to_cloud_vendor):
    orgs = row['Organization'].split(',')
    vendor = None
    for org in orgs:
        for key in org_to_cloud_vendor:
            if key in org.lower():
                vendor = org_to_cloud_vendor[key]
                break
    if vendor is None:
        vendor = 'Amazon Web Services'  # default
    return vendor


def find_price_for_vendor_and_hardware_model(closest_price_dates_df, purchase_time, price_colname):
    price_per_chip_hour = None
    for i, price_row in closest_price_dates_df.iterrows():
        if price_row['Price date'] <= purchase_time:
            price_per_chip_hour = price_row[price_colname]
            break
    if price_per_chip_hour is None:
        for i, price_row in closest_price_dates_df.iterrows():
            if price_row['Price date'] > purchase_time:
                price_per_chip_hour = price_row[price_colname]
                break
    if not pd.isna(price_per_chip_hour):
        return float(price_per_chip_hour)
    else:
        print(f"Could not find price")
        return None


def find_price(
    row, price_df, hardware_df, pcd_hardware_model_colname, price_colname, org_to_cloud_vendor,
):
    print(f"==== System: {row['System']} ====")
    purchase_time = get_purchase_time(row)
    if purchase_time is None:
        print()
        return None
    hardware_model = row[pcd_hardware_model_colname]
    if pd.isna(hardware_model):
        print(f"Could not find hardware model for {row['System']}")
        print()
        return None
    vendor = select_vendor(row, org_to_cloud_vendor)
    print(f"Trying {hardware_model} at {purchase_time}")

    # Find the price of the hardware at the time of purchase
    vendors = [vendor]
    if "TPU" not in hardware_model:
        # TPUs are only available from Google Cloud
        for possible_vendor in ['Amazon Web Services', 'Microsoft Azure', 'Google Cloud']:
            if possible_vendor != vendor:
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
            price_types.append(possible_price_type)
    
    for price_type in price_types:
        for vendor in vendors:
            print(f"Trying {vendor}, {price_type}")
            # TODO: fall back to soft matching of hardware models, e.g. "NVIDIA A100 SXM4 40 GB" falls back to "NVIDIA A100"
            closest_price_dates_df = find_closest_price_dates(
                vendor, hardware_model, purchase_time, price_df
            )
            price = find_price_for_vendor_and_hardware_model(
                closest_price_dates_df, purchase_time, price_type
            )
            if price is not None:
                print(f"Found price: {price}")
                # TODO: use standard committed-use discount (CUD) rates for each vendor to convert a price type to the desired one
                print()
                break
            # else: try again with a different vendor, price type
        if price is not None:
            break

    """
    TODO
    If the hardware type was already imputed, and we did not find any price for that hardware type
    from any cloud provider, then try the next nearest neighbor for the hardware type.
    """

    # if price is None:
    #     """
    #     We did not find any price from any cloud provider for the given hardware type. 
    #     So estimate a price from the FLOP/second performance of the GPU
    #     and the general trend in FLOP/$ for ML GPUs.
    #     # TODO: can we get a trend in cloud price FLOP/$ specifically?
    #     """
    #     print("Estimating price from FLOP/s and FLOP/$ trend")
    #     # Get the FLOP/second performance of the GPU from the hardware database
    #     flop_per_second_columns = [  # ordered by preference
    #         'FP16 Tensor Core',
    #         'Tensor Float 32 (TF32)',
    #         'FP32 (single precision) Performance (FLOP/s)',
    #     ]
    #     for col in flop_per_second_columns:
    #         hardware_df_match = hardware_df[hardware_df['Name of the hardware'] == hardware_model]
    #         flop_per_second = hardware_df_match[col].values[0]
    #         if not pd.isna(flop_per_second):
    #             flop_per_second_column = col
    #             break
    #     if pd.isna(flop_per_second):
    #         print(f"Could not find FLOP/s for {hardware_model}")
    #         return None
        
    #     """
    #     Calculate the FLOP/$ for ML GPUs at the time of purchase
    #     This is based on "FP32 price-performance" plot in the blog post:
    #     https://epochai.org/blog/trends-in-machine-learning-hardware
    #     Assumes a 2-year hardware replacement time.
    #     TODO: may want to use a different hardware replacement time, e.g. 4 years
    #     """ 
    #     if '16' in flop_per_second_column:
    #         flop_per_second_per_dollar_slope = 1 / 7
    #         # Assume a 2x improvement in FLOP/$ for FP16 over FP32
    #         flop_per_second_per_dollar_intercept = 15.55 + np.log10(2)
    #     else:
    #         # Use FP32
    #         flop_per_second_per_dollar_slope = 1 / 7  # OOMs/year
    #         flop_per_second_per_dollar_intercept = 15.55  # OOMs - at Jan 1, 2004
        
    #     intercept_date = pd.to_datetime('2004-01-01')
    #     purchase_time_offset = purchase_time - intercept_date
    #     purchase_time_offset_years = purchase_time_offset.days / DAYS_PER_YEAR
    #     flop_per_dollar = 10 ** (flop_per_second_per_dollar_intercept + \
    #         flop_per_second_per_dollar_slope * purchase_time_offset_years)
    #     # Calculate the price of the hardware at the time of purchase
    #     price_per_chip_second = flop_per_second / flop_per_dollar
    #     # Conver to price per hour
    #     price = price_per_chip_second * SECONDS_PER_HOUR

    #     print(f"Estimated price: {price}")
    #     print()

    return price
