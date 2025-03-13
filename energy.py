import pandas as pd


def power_usage_effectiveness(organization):
    org = organization.lower()
    """
    https://www.semianalysis.com/p/ai-datacenter-energy-dilemma-race
    "A typical enterprise colocation PUE is around 1.5-1.6, while most hyperscale datacenters are below 
    1.4 PUE, with some purpose build facilities (such as Google's) claim to achieve PUEs of below 1.10. 
    Most AI Datacenter specs aim for lower than 1.3 PUE."

    "For example at 80% utilization rate and a PUE of 1.25, [...]"

    Meta: https://sustainability.fb.com/data-centers/
    "Data center buildings we completed in 2022 exhibit a Power Usage Effectiveness (PUE) of 1.09"

    Google and Microsoft: Table 4 of https://arxiv.org/abs/2104.10350, all close to 1.1
    """
    hyperscalers = ['google', 'deepmind', 'microsoft', 'amazon', 'meta', 'facebook']
    if any([hs in org for hs in hyperscalers]):
        return 1.1
    return 1.25


def server_TDP_fraction(hardware_model):
    """
    Returns the fraction of the server's TDP that is used during training.
    """
    if "TPU" in hardware_model:
        """
        https://arxiv.org/abs/2104.10350, Table 4
        and
        https://gwern.net/doc/ai/scaling/hardware/2021-jouppi.pdf, Table 1
        Measured server power for several Google models is 37-53% of server TDP.
        The average value is 43%.

        Higher FLOP/s utilization values (>55%) may correlate with higher power usage than this.
        But overall this data suggests that TPUs use a lower fraction of TDP than NVIDIA GPUs.
        """
        return 0.43
    else:
        """
        Sources:

        https://arxiv.org/abs/2104.10350, Table 4
        Measured server power for GPT-3, at 330W, is ~53% or 88% of server TDP, depending on 
        whether they used DGX-1 (4 V100s, 1.5kW max) or DGX-2 (16 V100s, 10kW max).
        Note this is with a low (~20%) FLOP/s utilization for training GPT-3.
        
        https://docs.nvidia.com/nvidia-dgx-superpod-data-center-design-dgx-h100.pdf
        Table 4 shows expected average power equal to peak power for the DGX H100.
        "DGX H100 systems operate at or near peak utilization continuously when running AI 
        workloads."

        An experiment conducted by Epoch with a single NVIDIA H100 GPU found that at 70% FLOP/s 
        utilization, GPU power was at peak (700W).

        https://pages.cs.wisc.edu/~shivaram/cs744-readings/dc-computer-v3.pdf
        p.133
        For a typical new multi-megawatt data center, the authors assume the server's average 
        power draw is 75% of peak power. Note this number is not AI-specific.
        """
        return 0.75
    

def chip_to_server_power(hardware_model):
    """
    Returns a multiplier to estimate server power from chip power.
    """
    if "TPU" in hardware_model:
        """
        https://gwern.net/doc/ai/scaling/hardware/2021-jouppi.pdf, Table 1
        TPUv1:
        - System: 220W
        - Chip: 75W
        - Ratio: 2.93
        TPUv2: 
        - System: 460W
        - Chip: 280W
        - Ratio: 1.64
        TPUv3:
        - System: 660W
        - Chip: 450W
        - Ratio: 1.47
        TPUv4i: (different to TPUv4)
        - System: 275W
        - Chip: 175W
        - Ratio: 1.57

        Average of v2, v3 and v4i (v1 seems too dissimilar): 1.56
        """
        if "v1" in hardware_model:
            return 2.93
        elif "v2" in hardware_model:
            return 1.64
        elif "v3" in hardware_model:
            return 1.47
        else:
            return 1.56
    else:
        """
        Examples (using max specs):
        H100:
        - DGX 8x GPU: 10.2kW (https://resources.nvidia.com/en-us-dgx-systems/ai-enterprise-dgx)
        - GPU: 0.7kW (https://www.nvidia.com/en-us/data-center/h100/)
        - Ratio of per-GPU power: 1.82
        A100:
        - DGX 8x GPU: 6.5kW (https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-dgx-a100-datasheet.pdf)
        - GPU: 0.4kW (https://www.nvidia.com/en-us/data-center/a100/)
        - Ratio of per-GPU power: 2.03
        V100:
        - DGX-1 4x GPU: 1.5kW (https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/dgx-station/dgx-station-print-explorer-datasheet-letter-final-web.pdf)
        - DGX-2 16x GPU: 10kW (https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/dgx-station/dgx-station-print-explorer-datasheet-letter-final-web.pdf)
        - Ratio of per-GPU power: 1.25 to 2.08
        - GPU: 0.3kW (https://www.nvidia.com/en-gb/data-center/tesla-v100/)

        Average of ratios above: 1.8
        """
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
    https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_a
    Average US industrial electricity price as of January 2023: 8.32 Cents per Kilowatthour

    Previous years: https://www.statista.com/statistics/190680/us-industrial-consumer-price-estimates-for-retail-electricity-since-1970/
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
        2024: 0.0806,  # placeholder
    }
    return prices[year]


def cluster_power_capacity(hardware_model, hardware_quantity, hardware_df, organization):
    """
    hardware_model: name of the hardware used for the training run
    hardware_quantity: number of chips used for the training run
    hardware_df: DataFrame containing hardware specs
    organization: name of the organization who did the training

    Returns the power capacity in kilowatts required to do the training run.
    """
    matching_hardware = hardware_df[hardware_df['Hardware name'] == hardware_model]
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
    # Adjust for data center power distribution and cooling
    adj_server_power_kw = server_TDP_kw * power_usage_effectiveness(organization)
    cluster_kw = adj_server_power_kw * hardware_quantity
    return cluster_kw


# https://www.nrel.gov/docs/fy20osti/73901.pdf
# Converted by ChatGPT, only checked a few values for accuracy
us_state_energy_prices_mmbtu = {
    "Hawaii": 67.17,
    "Alaska": 47.89,
    "Rhode Island": 42.69,
    "Washington": 13.49,
    "Montana": 15.38,
    "Texas": 15.69,
    "California": 37.31,
    "Oregon": 17.54,
    "Nevada": 18.03,
    "Idaho": 19.52,
    "Utah": 17.97,
    "Arizona": 18.91,
    "New Mexico": 18.04,
    "Colorado": 22.45,
    "Wyoming": 21.99,
    "North Dakota": 22.37,
    "South Dakota": 22.97,
    "Nebraska": 20.27,
    "Kansas": 22.11,
    "Oklahoma": 15.89,
    "Minnesota": 21.61,
    "Iowa": 18.19,
    "Missouri": 17.80,
    "Arkansas": 16.06,
    "Louisiana": 15.55,
    "Wisconsin": 21.96,
    "Illinois": 21.09,
    "Michigan": 19.08,
    "Indiana": 17.10,
    "Ohio": 18.18,
    "Kentucky": 17.55,
    "Tennessee": 18.05,
    "Mississippi": 17.46,
    "Alabama": 18.13,
    "Florida": 22.94,
    "Georgia": 18.46,
    "South Carolina": 18.18,
    "North Carolina": 18.13,
    "Virginia": 17.36,
    "West Virginia": 16.75,
    "Pennsylvania": 21.48,
    "New York": 26.98,
    "New Jersey": 19.47,
    "Delaware": 24.53,
    "Maryland": 22.10,
    "Vermont": 21.09,
    "New Hampshire": 22.79,
    "Massachusetts": 29.65,
    "Connecticut": 24.12,
    "Maine": 38.40
}

US_STATE_ENERGY_PRICES_PER_KWH = {state: price / 293.071 for state, price in us_state_energy_prices_mmbtu.items()}

# Common data center states in the US
data_center_states = [
    "Virginia",
    "Texas",
    "California",
    "New York",
    "Illinois",
    "Oregon",
    "Nevada",
    "Washington",
    "Georgia",
    "Ohio",
    "North Carolina"
]
