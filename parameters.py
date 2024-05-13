SECONDS_PER_HOUR = 60 * 60
DAYS_PER_YEAR = 365.25
HOURS_PER_YEAR = 24 * DAYS_PER_YEAR
SECONDS_PER_YEAR = SECONDS_PER_HOUR * 24 * DAYS_PER_YEAR
CLUSTER_INTERCONNECT_COST_OVERHEAD = 1.15  # TODO switch-case on known values?
MEDIAN_UTILIZATION = 0.375  # median of 33 known values
MEDIAN_TRAINING_TIME_DAYS = 793.5 / 24 # for running top-10 models
ML_GPU_PRICE_PERFORMANCE_OOMS_PER_YEAR = 0.14  # https://epochai.org/blog/trends-in-machine-learning-hardware

# 0.14 OOMs/year of hardware progress => optimum of 0.14 * np.log(10) replacements/year
# https://epochai.org/blog/trends-in-machine-learning-hardware
# https://epochai.org/blog/the-longest-training-run#a-simple-framework-for-training-run-lengths
HARDWARE_REPLACEMENT_PER_YEAR = 0.32
