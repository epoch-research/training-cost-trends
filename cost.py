import numpy as np
import os
import pandas as pd

from data import *
from energy import *
from imputation import *
from inflation import *
from parameters import *
from plotting import *
from prices import *


def estimate_costs(
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
    else:
        for _, row in frontier_pcd_df.iterrows():
            if not(pd.isna(row['Training time (hours)']) or pd.isna(row['Hardware quantity'])):
                frontier_pcd_df['Training time (chip hours)'] = frontier_pcd_df['Training time (hours)'] * frontier_pcd_df['Hardware quantity']
    
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
        price, _ = find_price(row, price_df, hardware_df, pcd_hardware_model_colname, price_colname, org_to_cloud_vendor)
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


def estimate_amortized_hardware_costs(
    frontier_pcd_df,
    hardware_df,
    price_df,
    impute_pcd_fn=None,
    **impute_kwargs,
):
    """
    Full pipeline for estimating amortized hardware costs
    """
    print("Estimating amortized hardware costs")
    if impute_pcd_fn is not None:
        frontier_pcd_df = impute_pcd_fn(frontier_pcd_df, **impute_kwargs)
    
    pcd_hardware_model_colname = 'Training hardware'
    price_colname = 'Price (hardware purchase)'
    system_to_price = {}

    for i, row in frontier_pcd_df.iterrows():
        # TODO
        price, _ = find_purchase_price(
            row, price_df, hardware_df, pcd_hardware_model_colname, price_colname
        )
        if price is None:
            continue
        else:
            system_to_price[row['System']] = price

    # Cost estimation
    # TODO move outside of the function
    def estimate_cost(row, system_to_price):
        system = row['System']
        price = system_to_price.get(system)
        if price is None:
            return None

        hardware_quantity = row['Hardware quantity']
        training_time = row['Training time (hours)']
        hardware_lifetime = DEFAULT_HARDWARE_LIFETIME

        if any([np.isnan(x) for x in [hardware_quantity, training_time]]):
            # Impute training time
            # TODO: move this to a separate function
            if not any([pd.isna(row[x]) for x in ['Training compute (FLOP)', 'Training hardware']]):
                print("Imputing training time from compute and hardware")
                flop = row['Training compute (FLOP)']
                hardware_model = row['Training hardware']
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

        price_per_chip_hour = price / hardware_lifetime
        cost = price_per_chip_hour * training_chip_hours

        # Check for base model
        if not pd.isna(row['Base model']):
            base_model_name = row['Base model']
            base_model = frontier_pcd_df[frontier_pcd_df['System'] == base_model_name].squeeze()
            if base_model.empty:
                return None
            base_cost = estimate_cost(base_model, system_to_price)
            if base_cost is None:
                return None
            else:
                cost += base_cost

        return cost
        
    system_to_cost = {}
    for i, row in frontier_pcd_df.iterrows():
        print(f"==== System: {row['System']} ====")
        cost = estimate_cost(row, system_to_price)
        if cost is None:
            continue
        system_to_cost[row['System']] = cost

    print(system_to_cost)

    frontier_pcd_df['Cost'] = frontier_pcd_df['System'].map(system_to_cost)

    return frontier_pcd_df


def estimate_upfront_server_capex(
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
        # TODO
        price, _ = find_purchase_price(
            row, price_df, hardware_df, pcd_hardware_model_colname, price_colname
        )
        if price is None:
            continue
        else:
            system_to_price[row['System']] = price

    # Cost estimation
    # TODO move outside of the function
    def estimate_cost(row, system_to_price):
        system = row['System']
        price = system_to_price.get(system)
        if price is None:
            return None
        
        # training_time = row['Training time (hours)']
        hardware_quantity = row['Hardware quantity']
        cost = hardware_quantity * price

        # Check for base model
        if not pd.isna(row['Base model']):
            base_model_name = row['Base model']
            base_model = frontier_pcd_df[frontier_pcd_df['System'] == base_model_name].squeeze()
            if base_model.empty:
                return None
            base_cost = estimate_cost(base_model, system_to_price)
            if base_cost is None:
                return None
            else:
                cost += base_cost

        return cost
        
    system_to_cost = {}
    for i, row in frontier_pcd_df.iterrows():
        print(f"==== System: {row['System']} ====")
        cost = estimate_cost(row, system_to_price)
        if cost is None:
            continue
        system_to_cost[row['System']] = cost

    print(system_to_cost)

    frontier_pcd_df['Cost'] = frontier_pcd_df['System'].map(system_to_cost)

    return frontier_pcd_df


def estimate_hardware_capex_opex(
    frontier_pcd_df,
    hardware_df,
    price_df,
    impute_pcd_fn=None,
    **impute_kwargs,
):
    """
    Full pipeline for estimating the sum of amortized hardware CapEx and OpEx
    """
    if impute_pcd_fn is not None:
        frontier_pcd_df = impute_pcd_fn(frontier_pcd_df, **impute_kwargs)
    
    pcd_hardware_model_colname = 'Training hardware'
    price_colname = 'Price (hardware purchase)'
    system_to_price = {}

    for i, row in frontier_pcd_df.iterrows():
        # TODO
        price, _ = find_purchase_price(
            row, price_df, hardware_df, pcd_hardware_model_colname, price_colname
        )
        if price is None:
            continue
        else:
            system_to_price[row['System']] = price

    # Cost estimation
    # TODO move outside of the function
    def estimate_cost(row, system_to_price):
        system = row['System']
        price = system_to_price.get(system)
        if price is None:
            return None

        hardware_quantity = row['Hardware quantity']
        training_time = row['Training time (hours)']
        hardware_model = row['Training hardware']
        hardware_lifetime = DEFAULT_HARDWARE_LIFETIME

        if any([np.isnan(x) for x in [hardware_quantity, training_time]]):
            # Impute training time
            # TODO: move this to a separate function
            if not any([pd.isna(row[x]) for x in ['Training compute (FLOP)', 'Training hardware']]):
                print("Imputing training time from compute and hardware")
                flop = row['Training compute (FLOP)']
                flop_per_second = get_flop_per_second(hardware_model, hardware_df)
                flop_utilization = row['Hardware utilization']
                if pd.isna(flop_utilization):
                    flop_utilization = MEDIAN_UTILIZATION

                training_chip_seconds = flop / (flop_per_second * flop_utilization)
                training_chip_hours = training_chip_seconds / SECONDS_PER_HOUR
            else:
                print("Unable to estimate training chip hours")
                return None
        else:
            training_chip_hours = training_time * hardware_quantity

        total_chip_hours = training_chip_hours * 2  # TODO principled way to estimate total time

        cost = 0

        price_per_chip_hour = price / hardware_lifetime
        amortized_hardware_cost = price_per_chip_hour * training_chip_hours
        cost += amortized_hardware_cost

        # Add interconnect cost
        cost *= CLUSTER_INTERCONNECT_COST_OVERHEAD

        org = row['Organization']
        pub_year = row['Publication date'].year
        energy_cost = cluster_energy_cost(
            hardware_model, total_chip_hours, hardware_df, org, pub_year,
        )
        cost += energy_cost

        # Check for base model
        if not pd.isna(row['Base model']):
            base_model_name = row['Base model']
            base_model = frontier_pcd_df[frontier_pcd_df['System'] == base_model_name].squeeze()
            if base_model.empty:
                print("Base model specified, but not found in database")
                return None
            base_cost = estimate_cost(base_model, system_to_price)
            if base_cost is None:
                print("Base model found, but unable to estimate cost")
                return None
            else:
                cost += base_cost

        return cost
        
    system_to_cost = {}
    for i, row in frontier_pcd_df.iterrows():
        print(f"==== System: {row['System']} ====")
        cost = estimate_cost(row, system_to_price)
        if cost is None:
            print("Unable to estimate cost")
            continue
        else:
            print("Estimated cost:", cost)
        system_to_cost[row['System']] = cost

    print(system_to_cost)

    frontier_pcd_df['Cost'] = frontier_pcd_df['System'].map(system_to_cost)

    return frontier_pcd_df


def cluster_interconnect_cost_per_gpu(hardware_model, hardware_df):
    # TODO: move to parameters.py
    cluster_interconnect_cost_per_gbps = 11
    matching_row = hardware_df[hardware_df['Name of the hardware'] == hardware_model].squeeze()
    cluster_interconnect_bandwidth = matching_row['Internode bandwidth (bit/s)'] / 1e9
    if pd.isna(cluster_interconnect_bandwidth):
        # Assume there is no interconnect
        return 0
    cost_per_gpu = cluster_interconnect_cost_per_gbps * cluster_interconnect_bandwidth
    return cost_per_gpu


def cluster_energy_cost(hardware_model, total_chip_hours, hardware_df, organization, year):
    """
    hardware_model: name of the hardware used for the training run
    total_chip_hours: total number of chip-hours used for the training run (i.e. number of chips * training time)
    hardware_df: DataFrame containing hardware specs
    organization: name of the organization who did the training
    year: year in which the training run was conducted
    """
    matching_hardware = hardware_df[hardware_df['Name of the hardware'] == hardware_model]
    # TODO: handle missing values - look up similar hardware
    chip_TDP_kw = matching_hardware['TDP (W)'].squeeze() / 1000
    # Adjust for whole server power draw (CPUs, memory, cooling)
    server_TDP_kw = chip_TDP_kw * chip_to_server_power(hardware_model)
    # Adjust for average power draw
    server_power_kw = server_TDP_kw * server_TDP_fraction(hardware_model)
    # Adjust for data center power distribution and cooling
    adj_server_power_kw = server_power_kw * power_usage_effectiveness(organization)
    cluster_kwh = adj_server_power_kw * total_chip_hours
    return cluster_kwh * energy_price(year)
