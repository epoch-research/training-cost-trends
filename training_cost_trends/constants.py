"""Constants, hardware mappings, and configuration defaults."""

from pathlib import Path

import numpy as np

# ==============================================================================
# TIME AND ECONOMIC CONSTANTS
# ==============================================================================

SECONDS_PER_HOUR = 60 * 60
DAYS_PER_YEAR = 365.25
HOURS_PER_YEAR = 24 * DAYS_PER_YEAR
SECONDS_PER_YEAR = SECONDS_PER_HOUR * 24 * DAYS_PER_YEAR
CLUSTER_INTERCONNECT_COST_FRACTION = 0.19  # fraction of total cluster cost
MEDIAN_UTILIZATION = 0.375  # median of 33 known values
MEDIAN_TRAINING_TIME_DAYS = 793.5 / 24  # median of the running top-10 models
ML_GPU_PRICE_PERFORMANCE_OOMS_PER_YEAR = 0.14  # https://epochai.org/blog/trends-in-machine-learning-hardware

DEFAULT_RNG = np.random.default_rng(20240531)

DEFAULT_COMPUTE_THRESHOLD = 5

PRICE_INDEX_SERIES = "PCU518210518210"

# ==============================================================================
# FILE PATHS
# ==============================================================================

PACKAGE_DIR = Path(__file__).parent
PROJECT_DIR = PACKAGE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
PRICE_INDEX_FILE = DATA_DIR / "PCU518210518210.csv"
HARDWARE_PRICE_FILE = DATA_DIR / "Hardware prices.csv"
HARDWARE_FILE = DATA_DIR / "Chip dataset-Grid view.csv"
MODELS_FILE = DATA_DIR / "All ML Systems - full view.csv"

# ==============================================================================
# HARDWARE MAPPINGS
# ==============================================================================

# Mapping of simplified hardware names, for soft matching
SIMPLIFIED_HARDWARE_NAMES: dict[str, str] = {
    "NVIDIA Tesla V100 DGXS 16 GB": "NVIDIA V100",
    "NVIDIA Tesla V100 DGXS 32 GB": "NVIDIA V100",
    "NVIDIA Tesla V100S PCIe 32 GB": "NVIDIA V100",  # similar in specs
    "NVIDIA V100": "NVIDIA V100",
    "NVIDIA A100 PCIe": "NVIDIA A100",
    "NVIDIA A100 SXM4 40 GB": "NVIDIA A100",
    "NVIDIA A100 SXM4 80 GB": "NVIDIA A100",
    "NVIDIA A100": "NVIDIA A100",
    "NVIDIA H100 PCIe": "NVIDIA H100",
    "NVIDIA H100 SXM5": "NVIDIA H100",
    "NVIDIA H100": "NVIDIA H100",
}

GPU_HARDWARE_ALIASES: list[str] = [
    "A100",
    "V100",
    "H100",
    "P100",
    "K80",
    "K40",
    "Titan X",
    "GTX 580",
    "GTX TITAN",
    "Titan Black",
]

# ==============================================================================
# CLOUD VENDOR CONFIGURATION
# ==============================================================================

# Default committed-use discounts (CUD) for each cloud provider.
# See cud_estimate.ipynb
DEFAULT_CUD: dict[str, dict[str, float]] = {
    "Amazon Web Services": {
        "Price per chip-hour (on-demand)": 0,
        "Price per chip-hour (1-year CUD)": 0.41,
        "Price per chip-hour (3-year CUD)": 0.64,
    },
    "Google Cloud": {
        "Price per chip-hour (on-demand)": 0,
        "Price per chip-hour (1-year CUD)": 0.37,
        "Price per chip-hour (3-year CUD)": 0.56,
    },
    "Microsoft Azure": {
        "Price per chip-hour (on-demand)": 0,
        "Price per chip-hour (1-year CUD)": 0.25,
        "Price per chip-hour (3-year CUD)": 0.49,
    },
    # Assume Lambda Labs has average of above CUDs
    "Lambda Labs": {
        "Price per chip-hour (on-demand)": 0,
        "Price per chip-hour (1-year CUD)": 0.34,
        "Price per chip-hour (3-year CUD)": 0.56,
    },
}

# These numbers are kept precise to match our calculations
# Even though they have high uncertainty
TPU_EQUIVALENT_RELEASE_PRICES: dict[str, int] = {
    "Google TPU v1": 5463,
    "Google TPU v2": 5054,
    "Google TPU v3": 5276,
    "Google TPU v4": 5167,
}

PRIORITY_VENDORS: list[str] = ["Amazon Web Services", "Microsoft Azure", "Google Cloud"]

# Canonical mapping from organization name substrings (lowercase) to cloud vendors.
# Used by select_vendor() (substring match against lowercased org name) and
# estimate_cloud_costs(). Also used by knn_impute_pcd() with case-insensitive matching.
ORG_TO_CLOUD_VENDOR: dict[str, str] = {
    "google": "Google Cloud",
    "deepmind": "Google Cloud",
    "google deepmind": "Google Cloud",
    "google brain": "Google Cloud",
    "microsoft": "Microsoft Azure",
    "openai": "Microsoft Azure",
}
