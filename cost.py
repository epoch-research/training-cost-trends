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
from utils import *


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
        print_safely(f"==== System: {row['System']} ====")
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
            base_model_name = row['Base model']
            base_model = frontier_pcd_df[frontier_pcd_df['System'] == base_model_name].squeeze()
            base_cost = estimate_cost(base_model, system_to_price)
            if base_model.empty:
                print("Base model specified, but not found in database")
                return None
            if base_cost is None:
                print("Base model found, but unable to estimate cost")
                return None
            else:
                cost += base_cost

        return cost
        
    system_to_cost = {}
    for i, row in frontier_pcd_df.iterrows():
        print_safely(f"==== System: {row['System']} ====")
        cost = estimate_cost(row, system_to_price)
        if cost is None:
            print("Unable to estimate cost")
            continue
        else:
            print("Estimated cost:", cost)
        system_to_cost[row['System']] = cost

    print("All costs:")
    print(system_to_cost)

    frontier_pcd_df['Cost'] = frontier_pcd_df['System'].map(system_to_cost)

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
        print_safely(f"==== System: {row['System']} ====")
        price, _ = get_hardware_acquisition_price(
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
        # Add interconnect cost
        cost *= 1 / (1 - CLUSTER_INTERCONNECT_COST_FRACTION)

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
        print_safely(f"==== System: {row['System']} ====")
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
        print_safely(f"==== System: {row['System']} ====")
        price, _ = get_hardware_value_at_training_start(
            row, price_df, hardware_df, pcd_hardware_model_colname, price_colname
        )
        if price is None:
            continue
        else:
            system_to_price[row['System']] = price
        
    system_to_cost = {}
    for i, row in frontier_pcd_df.iterrows():
        print_safely(f"==== System: {row['System']} ====")
        cost = estimate_hardware_capex_energy_cost(
            row, system_to_price, frontier_pcd_df, hardware_df, separate_components
        )
        if cost is None:
            print("Unable to estimate cost")
            continue
        else:
            print("Estimated cost:", cost)
        system_to_cost[row['System']] = cost

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
            frontier_pcd_df[k] = frontier_pcd_df['System'].map(system_to_component_cost)
        frontier_pcd_df['Cost'] = frontier_pcd_df[cost_component_names].sum(axis=1)
    else:
        frontier_pcd_df['Cost'] = frontier_pcd_df['System'].map(system_to_cost)

    return frontier_pcd_df


def estimate_hardware_capex_energy_cost(
    row, system_to_price, frontier_pcd_df, hardware_df, separate_components=False,
):
    system = row['System']
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
        base_model_name = row['Base model']
        base_model = frontier_pcd_df[frontier_pcd_df['System'] == base_model_name].squeeze()
        if base_model.empty:
            print("Base model specified, but not found in database")
            return None
        base_cost = estimate_hardware_capex_energy_cost(
            base_model, system_to_price, frontier_pcd_df, hardware_df, separate_components,
        )
        if base_cost is None:
            print("Base model found, but unable to estimate cost")
            return None
        else:
            if separate_components:
                for k in cost:
                    cost[k] += base_cost[k]
            else:
                cost += base_cost
    return cost


def cluster_energy_cost(hardware_model, total_chip_hours, hardware_df, organization, year):
    """
    hardware_model: name of the hardware used for the training run
    total_chip_hours: total number of chip-hours used for the training run (i.e. number of chips * training time)
    hardware_df: DataFrame containing hardware specs
    organization: name of the organization who did the training
    year: year in which the training run was conducted
    """
    matching_hardware = hardware_df[hardware_df['Name of the hardware'] == hardware_model]
    chip_TDP_kw = matching_hardware['TDP (W)'].squeeze() / 1000
    if pd.isna(chip_TDP_kw):
        if "TPU v4" in hardware_model:
            """
            https://cloud.google.com/blog/topics/systems/tpu-v4-enables-performance-energy-and-co2e-efficiency-gains
            "Google's Cloud TPU v4 outperforms TPU v3 by 2.1x on average on a per-chip basis and improves performance/Watt by 2.7x."
            TPU v3 performance per Watt: 123 TFLOPS / 450W = 0.273 TFLOPS/W
            0.273 * 2.7 = 0.738 TFLOPS/W
            TPU v4 is 275 TFLOPS => 275 / 0.738 = 373W
            """
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
