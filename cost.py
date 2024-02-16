import numpy as np
import os
import pandas as pd

from data import *
from plotting import *
from prices import *
from imputation import *
from inflation import *


def estimate_costs(
    frontier_pcd_df,
    hardware_df,
    price_df,
    impute_pcd_data=False,
    impute_pcd_fn=knn_impute_pcd,
    **impute_kwargs,
):
    """
    Full cost estimation pipeline
    """
    # Imputation
    if impute_pcd_data:
        impute_pcd_fn(frontier_pcd_df, **impute_kwargs)
    else:
        # set the System column as the index for formatting purposes
        frontier_pcd_df = frontier_pcd_df.set_index('System')
        frontier_pcd_df['System'] = frontier_pcd_df.index
        for _, row in frontier_pcd_df.iterrows():
            if not(pd.isna(row['Training time (hours)']) or pd.isna(row['Hardware quantity'])):
                frontier_pcd_df['Training time (chip hours)'] = frontier_pcd_df['Training time (hours)'] * frontier_pcd_df['Hardware quantity']

    """
    Price selection
    1. Use a fixed mapping from Organization to cloud provider. If no mapping found, default to "Amazon Web Services".
    2. If there's a match for the hardware model, use that. Else, discard the ML system from the dataset.
    3. Use the price that is nearest to, but prior to, training time + 2 months before the publication date
    4. If there are no prices prior to that time, use the nearest price after that time
    5. If there are no prices for that hardware model and cloud provider at all, repeat steps 3 and 4 for "Microsoft Azure", then "Google Cloud" as the cloud provider.
    6. If there are no prices found from step 5, discard the ML system from the dataset.
    """
    
    # TODO: use the vendor mapping from imputation to reduce repetition
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
        price = find_price(row, price_df, hardware_df, pcd_hardware_model_colname, price_colname, org_to_cloud_vendor)
        if price is None:
            continue
        else:
            system_to_price[row['System']] = price

    # Cost estimation
    # cost = price_per_chip_hour * chip_hours
    # TODO move outside of the function
    def estimate_cost(row, system_to_price):
        system = row['System']
        price = system_to_price.get(system)
        if price is None:
            return None

        chip_hours = row['Training time (chip hours)']
        if np.isnan(chip_hours):
            return None

        cost = price * chip_hours

        # Check for base model
        if not pd.isna(row['Base model']):
            base_model_name = row['Base model']
            base_model = frontier_pcd_df[frontier_pcd_df['System'] == base_model_name].squeeze()
            base_cost = estimate_cost(base_model, system_to_price)
            if base_cost is None:
                return None
            else:
                cost += base_cost

        return cost
        
    system_to_cost = {}
    for i, row in frontier_pcd_df.iterrows():
        cost = estimate_cost(row, system_to_price)
        if cost is None:
            continue
        system_to_cost[row['System']] = cost

    print(system_to_cost)

    frontier_pcd_df['Cost'] = frontier_pcd_df['System'].map(system_to_cost)

    return frontier_pcd_df
