# Introduction

This repository analyzes the rising costs of training frontier AI models, supporting the Epoch Research paper "The rising costs of training frontier AI models". The codebase has been streamlined into a single standalone Python script that handles all cost estimation tasks.

# Tips
- Use `python3` rather than `python` for relevant bash commands
- When you see an error, for example from using a bash command, write it to "ERRORS.md" with a full comment on the situation so that I can troubleshoot for future sessions. The description should be roughly between 1 compound sentence and 2 short paragraphs in length.

# Simplified Codebase Structure

## Core File: `cost_analysis_standalone.py`

This single comprehensive script (~1,465 lines) contains all necessary functions and constants for cost estimation analysis. It consolidates functionality that was previously split across multiple modules.

### Key Constants and Configuration (Lines 14-94)

**Time and Economic Constants:**
- `SECONDS_PER_HOUR = 3600`, `HOURS_PER_YEAR = 8766`, `DAYS_PER_YEAR = 365.25`
- `CLUSTER_INTERCONNECT_COST_FRACTION = 0.19` (19% of total cluster cost)
- `MEDIAN_UTILIZATION = 0.375` (37.5% median utilization rate)
- `ML_GPU_PRICE_PERFORMANCE_OOMS_PER_YEAR = 0.14` (hardware progress rate)

**Hardware Mappings:**
- `SIMPLIFIED_HARDWARE_NAMES` - Maps specific GPU variants to simplified names (e.g., "NVIDIA A100 SXM4 40 GB" → "NVIDIA A100")
- `GPU_HARWARE_ALIASES` - List of common GPU aliases for matching
- `TPU_EQUIVALENT_RELEASE_PRICES` - Fixed TPU pricing estimates

**Cloud Vendor Configuration:**
- `DEFAULT_CUD` - Committed-use discount rates for AWS, Google Cloud, Azure, Lambda Labs
- `PRIORITY_VENDORS` - Vendor preference order for price lookups

### Data Loading Functions

**Core Data Loaders:**
- `load_frontier_systems()` - Loads precomputed frontier model lists from JSON files
- `load_pcd_df()`, `load_hardware_df()`, `load_price_df()` - CSV data loaders
- `load_data_for_cost_estimation()` - Complete data pipeline integrating all data sources

### Hardware Analysis Functions

**Performance Lookup:**
- `get_flop_per_second()` - Hardware performance lookup with priority order:
  1. Tensor-FP16/BF16 performance (preferred)
  2. TF32 performance
  3. FP16 performance (TPUs only)
  4. FP32 performance (fallback)

**Server Economics:**
- `get_server_lifetime()` - Server depreciation timeline (3-5 years by era)
- `get_server_cost_overhead()` - Server vs. GPU cost multipliers (1.54-1.69x)

### Energy Cost Calculations

**Power Modeling:**
- `power_usage_effectiveness()` - Data center efficiency (1.1x for hyperscalers, 1.25x others)
- `server_TDP_fraction()` - Average power during training (43% TPUs, 75% GPUs)
- `energy_price()` - US industrial electricity prices by year (2010-2025)

### Price Discovery and Matching

**Temporal Price Matching:**
- `find_closest_price_dates()` - Hardware price lookup with exact and soft matching
- `get_training_start_date()` - Training timeline estimation with special cases
- `find_price()` - Multi-tier fallback system across vendors and CUD levels

### Data Imputation Methods

**KNN Imputation:**
- `knn_impute_pcd()` - Full KNN imputation pipeline with feature engineering
- Default: 5 neighbors for both numerical and categorical imputation

**Most Common Imputation:**
- `most_common_impute_training_hardware()` - Year-based mode imputation

### Cost Estimation Methods

**Method 1: Hardware CapEx + Energy (Primary)**
- `estimate_hardware_capex_energy()` - Amortized hardware + energy cost
- Uses optimal hardware replacement rate based on price-performance trends

**Method 2: Hardware Acquisition**
- `estimate_hardware_acquisition_cost()` - Upfront hardware purchase cost
- Includes server overhead and interconnect costs

**Method 3: Cloud Costs**
- `estimate_cloud_costs()` - Cloud rental cost with committed-use discounts
- Vendor selection based on organization mapping

### Main Analysis Workflow

**Configuration Options:**
- `compute_threshold_method` - Frontier model selection method
- `enable_imputation` - Toggle data imputation
- `imputation_method` - Choose imputation strategy

**Execution Pipeline:**
1. Data Loading - Load models, hardware specs, prices
2. Data Quality Assessment - Report missing data
3. Imputation - Apply selected imputation method
4. Cost Estimation - Run all three methods
5. Inflation Adjustment - Adjust to current prices
6. Results Export - Generate output files

## Data Files

### Core Datasets (in `data/` directory)
- `All ML Systems - full view.csv` - Epoch AI model database with training details
- `Hardware prices.csv` - GPU/TPU pricing data across vendors and time
- `Chip dataset-Grid view.csv` - Hardware technical specifications (FLOP/s, TDP, etc.)
- `PCU518210518210.csv` - Producer price index for inflation adjustment

### Precomputed Model Selections
- `frontier_systems_by_top_n.json` - Top-N frontier model selections
- `frontier_systems_by_window_percentile.json` - Percentile-based selections
- `frontier_systems_by_backward_window_percentile.json` - Backward-looking percentile
- `frontier_systems_by_residual_from_trend.json` - Residual-based selections
- `gpt-4_contributions.json` - GPT-4 specific analysis data

## Output Files

The script generates results in the `results/` directory with timestamped subdirectories:

### Standard Outputs
- `cost_dataset_3_estimates.csv` - Comparison of all three cost methods
- `cost_dataset_detailed.csv` - Detailed results for primary method
- `cost_components.csv` - Component cost breakdown
- `cost_estimation_*.out` - Detailed logs for each method

## Usage

To run the cost analysis:

```bash
python3 cost_analysis_standalone.py
```

The script will:
1. Load all required data from the `data/` directory
2. Apply the configured imputation method
3. Run all three cost estimation methods
4. Generate timestamped results in the `results/` directory

## Key Modifiable Parameters

### Economic Assumptions (Lines 17-24)
- `CLUSTER_INTERCONNECT_COST_FRACTION` - Network infrastructure cost percentage
- `MEDIAN_UTILIZATION` - Hardware utilization rate assumption
- `ML_GPU_PRICE_PERFORMANCE_OOMS_PER_YEAR` - Hardware progress rate

### Cost Method Selection
Enable/disable specific cost estimation methods in the main execution section

### Imputation Strategy
- Switch between KNN and most-common imputation
- Adjust KNN neighbor count
- Toggle imputation entirely

### Cloud Pricing (Lines 60-84)
- Update committed-use discount rates
- Modify vendor priority order
- Adjust TPU equivalent prices