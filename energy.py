

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
        # TODO: 90% CI of 0.3 to 0.8
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
        # TODO: 90% CI of 0.5 to 1.0
        return 0.75
    

def chip_to_server_power(hardware_model):
    """
    Returns a multiplier to estimate server power from chip power.
    """
    if "TPU" in hardware_model:
        """
        https://gwern.net/doc/ai/scaling/hardware/2021-jouppi.pdf, Table 1
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

        Average: 1.56
        """
        if "v2" in hardware_model:
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
        2015: 0.0691,
        2016: 0.0676,
        2017: 0.0688,
        2018: 0.0692,
        2019: 0.0681,
        2020: 0.0667,
        2021: 0.0718,
        2022: 0.0832,
        2023: 0.0806,
    }
    return prices[year]
