import pandas as pd

from parameters import *


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
]


def get_flop_per_second(hardware_model, hardware_df):
    # Get FLOP/second from the hardware database
    flop_per_second_columns = [  # ordered by preference
        'FP16 Tensor Core',
        'Tensor Float 32 (TF32)',
        'FP16 (half precision) Performance (FLOP/s)',
        'FP32 (single precision) Performance (FLOP/s)',
    ]
    hardware_df_match = hardware_df[hardware_df['Name of the hardware'] == hardware_model]
    if 'TPU v1' in hardware_model:
        # Special case
        flop_per_second = hardware_df_match['INT8 Performance (OP/s)'].values[0]
        return flop_per_second
    for col in flop_per_second_columns:
        if col == 'FP16 (half precision) Performance (FLOP/s)':
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
    hardware_df_match = hardware_df[hardware_df['Name of the hardware'] == hardware_model]
    release_date = hardware_df_match['Release date'].values[0]
    return pd.to_datetime(release_date)


def get_server_lifetime(year):
    """
    Returns the estimated AI server lifetime in hours.
    
    This is part of a linear amortization model, where the value of the server depreciates linearly
    to zero at the end of the lifetime.

    From Feb 2022:
    https://www.datacenterfrontier.com/cloud/article/11427600/sturdier-servers-cloud-platforms-say-servers-living-longer-saving-billions
    "In July 2020, Microsoft extended the estimated useful life from 3 years to 4 years for servers, and from 2 years to 4 years for network equipment. The company estimated the change would boost operating income by $2.7 billion for fiscal year 2021."
    "In January 2021, Alphabet adjusted the estimated useful life of servers from 3 years to 4 years and network equipment from 3 years to 5 years. The company said the change would boost operating income by $2 billion for 2021."
    "Amazon says its extensions of useful life estimates in 2020 and 2021 raised operating income by $2 billion 2021, and will likely mean a $3.1 billion improvement for 2022."

    From Feb 2023:
    https://www.datacenterdynamics.com/en/news/google-increases-server-life-to-six-years-will-save-billions-of-dollars/
    "Google plans to increase the useful lives of its servers and some networking equipment to six years.
    "The company made the announcement in its earnings release, a day after Meta said it was increasing its server lifespan to five years."
    "Amazon Web Services is believed to run its servers for around five to six years, while Microsoft last year confirmed it had increased its server lifespans to six years."
    "Back in 2020, all four companies operated servers with a lifespan of just three years."

    We use 5 years for 2023 onwards on the assumption that state-of-the-art GPU servers are on the
    shorter side.
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
