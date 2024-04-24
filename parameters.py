SECONDS_PER_HOUR = 60 * 60
DAYS_PER_YEAR = 365.25
HOURS_PER_YEAR = 24 * DAYS_PER_YEAR
SECONDS_PER_YEAR = SECONDS_PER_HOUR * 24 * DAYS_PER_YEAR
DEFAULT_HARDWARE_LIFETIME = 3 * HOURS_PER_YEAR  # see e.g. https://gwern.net/doc/ai/scaling/hardware/2021-jouppi.pdf amortization of 3 years
HARDWARE_COST_OVERHEAD = 1.6
MEDIAN_UTILIZATION = 0.375  # median of 33 known values

"""
https://www.semianalysis.com/p/ai-datacenter-energy-dilemma-race
"A typical enterprise colocation PUE is around 1.5-1.6, while most hyperscale datacenters are below 
1.4 PUE, with some purpose build facilities (such as Google's) claim to achieve PUEs of below 1.10. 
Most AI Datacenter specs aim for lower than 1.3 PUE."

"For example at 80% utilization rate and a PUE of 1.25, [...]"

TODO: if hyperscaler, set 1.1, else 1.25
"""
DEFAULT_PUE = 1.25

"""
https://pages.cs.wisc.edu/~shivaram/cs744-readings/dc-computer-v3.pdf, p.133
For a typical new multi-megawatt data center, the authors assume the server's average power draw is
75% of peak power.
Note this may vary for different types of servers, e.g. for GPUs.

TODO: maybe this should be closer to 100%: see p.6 of https://docs.nvidia.com/nvidia-dgx-superpod-data-center-design-dgx-h100.pdf
"""
SERVER_TDP_RATIO = 0.75

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
SERVER_CHIP_POWER_RATIO = 1.8

"""
https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_a
Average US industrial electricity price as of January 2023: 8.32 Cents per Kilowatthour
"""
DATA_CENTER_ENERGY_PRICE = 0.083
