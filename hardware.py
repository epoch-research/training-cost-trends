import pandas as pd


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
