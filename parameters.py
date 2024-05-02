SECONDS_PER_HOUR = 60 * 60
DAYS_PER_YEAR = 365.25
HOURS_PER_YEAR = 24 * DAYS_PER_YEAR
SECONDS_PER_YEAR = SECONDS_PER_HOUR * 24 * DAYS_PER_YEAR
DEFAULT_HARDWARE_LIFETIME = 3 * HOURS_PER_YEAR  # see e.g. https://gwern.net/doc/ai/scaling/hardware/2021-jouppi.pdf amortization of 3 years
SERVER_COST_OVERHEAD = 1.6  # TODO switch-case on known values for A100 etc.
CLUSTER_INTERCONNECT_COST_OVERHEAD = 1.15  # TODO switch-case on known values?
MEDIAN_UTILIZATION = 0.375  # median of 33 known values
