# Introduction

Our current goal is to create a not-quite-standalone "update_model_costs.ipynb" notebook that we can run within the codebase every time we want updated cost data. We shouldn't be running regressions or plotting graphs, we just want a relatively neat and contained file.

# Tips
Use NotebookEdit to edit .ipynb files.
Use NotebookRead without a cell_id parameter - NotebookRead using cell_ids seems not to work at the moment (though NotebookEdit using cell_id works).
Use python3 rather than python for relevant bash commands.
All requirements should be imported via requirements.txt. If you need to import additional packages, add them to requirements.txt rather than using something like bash commands.
When you see an error, for example from using a bash command, write it to "ERRORS.md" with a full comment on the situation so that I can troubleshoot for future sessions. The description should be roughly between 1 compound sentence and 2 short paragraphs in length. I'll leave notes for how to avoid common issues here in Tips.

# Training Cost Trends Codebase Structure

This repository analyzes the rising costs of training frontier AI models, supporting the Epoch Research paper "The rising costs of training frontier AI models". 

## Core Python Modules

## Core File: `cost_analysis_standalone.py` (1,465 lines)

This single comprehensive script contains all necessary functions and constants for cost estimation analysis. It consolidates functionality from multiple modules into one executable file.

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

### Data Loading Functions (Lines 127-194)

**Core Data Loaders:**
- `load_frontier_systems()` - Loads precomputed frontier model lists from JSON files
- `load_pcd_df()`, `load_hardware_df()`, `load_price_df()` - CSV data loaders
- `load_data_for_cost_estimation()` - Complete data pipeline integrating all data sources

**Data Processing Choices:**
- Publication dates converted to datetime format with NaN handling
- Unicode character normalization (Σ symbol replacement)
- Automatic filtering to frontier models only

### Hardware Analysis Functions (Lines 196-301)

**Performance Lookup (Lines 200-227):**
- `get_flop_per_second()` - Hardware performance lookup with priority order:
  1. Tensor-FP16/BF16 performance (preferred)
  2. TF32 performance
  3. FP16 performance (TPUs only)
  4. FP32 performance (fallback)
- Special handling for TPU v1 using INT8 performance

**Server Economics (Lines 237-257):**
- `get_server_lifetime()` - Server depreciation timeline:
  - ≤2020: 3 years
  - 2021-2022: 4 years  
  - ≥2023: 5 years
- `get_server_cost_overhead()` - Server vs. GPU cost multipliers:
  - A100: 1.66x
  - V100: 1.69x
  - P100: 1.54x
  - Default: 1.64x

### Energy Cost Calculations (Lines 260-343)

**Power Modeling:**
- `power_usage_effectiveness()` - Data center efficiency:
  - Hyperscalers (Google, Microsoft, Amazon, Meta): 1.1x
  - Others: 1.25x
- `server_TDP_fraction()` - Average power during training:
  - TPUs: 43% of TDP
  - GPUs: 75% of TDP
- `chip_to_server_power()` - Chip-to-server power multipliers (hardware-specific)

**Energy Pricing:**
- `energy_price()` - US industrial electricity prices by year (2010-2025)
- Hardcoded annual values, no interpolation

### Price Discovery and Matching (Lines 346-696)

**Temporal Price Matching:**
- `find_closest_price_dates()` - Sophisticated price lookup with:
  - Exact hardware model matching
  - Soft matching via `SIMPLIFIED_HARDWARE_NAMES`
  - Vendor-specific filtering
  - Temporal proximity optimization

**Training Timeline Estimation:**
- `get_training_start_date()` - Estimates training start with:
  - Special cases for GPT-4, GPT-3.5, GPT-3, Gemini Ultra
  - Default: Publication date - training time - 60 days buffer
  - Fallback: Median training time (793.5 hours)
- `get_acquisition_date()` - Hardware acquisition: Training start - 60 days buffer

**Price Selection Logic:**
- `find_price()` - Multi-tier fallback system:
  1. Try selected vendor first
  2. Fall back to priority vendors (AWS, Azure, Google Cloud)
  3. Try different CUD commitment levels (3-year → 1-year → on-demand)
  4. Allow future price dates if no past prices available

**Hardware Depreciation:**
- `depreciate_by_hardware_progress()` - Uses `ML_GPU_PRICE_PERFORMANCE_OOMS_PER_YEAR` to depreciate hardware value over time

### Inflation Adjustment (Lines 698-717)

**CPI Adjustment:**
- inflation data in the PCU spreadsheet from https://fred.stlouisfed.org/series/PCU518210518210
- `adjust_value_for_inflation()` - Uses Producer Price Index (PCU518210518210.csv)
- `adjust_column_for_inflation()` - Batch processing for dataframes
- Default target: June 2025 prices

### Data Imputation Methods (Lines 719-931)

**KNN Imputation (Lines 730-864):**
- `knn_impute_pcd()` - Full KNN imputation pipeline:
  - Drops irrelevant columns (citations, notes, etc.)
  - Converts large numbers to log scale
  - One-hot encodes categorical variables
  - Imputes numerical values with KNeighborsRegressor
  - Separately imputes categorical training hardware with KNeighborsClassifier
  - Default: 5 neighbors

**Most Common Imputation (Lines 866-931):**
- `most_common_impute_training_hardware()` - Year-based mode imputation:
  - Groups models by publication year
  - Separates TPUs from GPUs for imputation
  - Uses full PCD dataset for better coverage
  - Handles comma-separated multiple values

**Imputation Configuration (Lines 1284-1317):**
- `enable_imputation` - Toggle imputation on/off
- `imputation_method` - Choose 'knn', 'most_common', or 'none'
- `knn_neighbors` - Number of neighbors for KNN (default: 5)

### Cost Estimation Methods (Lines 933-1237)

**Method 1: Hardware CapEx + Energy (Lines 985-1094)**
- `estimate_hardware_capex_energy()` - **Primary estimation method**
- Amortized hardware cost using optimal replacement rate
- Hardware replacement per year = `ML_GPU_PRICE_PERFORMANCE_OOMS_PER_YEAR * ln(10)`
- Total cost = Amortized hardware + Interconnect (19%) + Energy
- Uses depreciated hardware values at training start

**Method 2: Hardware Acquisition (Lines 1096-1162)**
- `estimate_hardware_acquisition_cost()` - Upfront hardware purchase cost
- Cost = Hardware quantity × Price per chip × Server overhead
- Adds interconnect cost: `Cost / (1 - CLUSTER_INTERCONNECT_COST_FRACTION)`
- Uses hardware acquisition prices

**Method 3: Cloud Costs (Lines 1164-1237)**
- `estimate_cloud_costs()` - Cloud rental cost estimation
- Uses committed-use discount rates (3-year CUD preferred)
- Cost = Price per chip-hour × Total chip-hours
- Vendor selection based on organization mapping

**Common Features:**
- All methods exclude fine-tuned models (base model field check)
- Chip-hours estimated from compute/hardware specs if missing
- Comprehensive logging and error handling

### Main Analysis Workflow (Lines 1243-1465)

**Configuration Options:**
- `compute_threshold_method` - Frontier model selection: 'top_n', 'window_percentile'
- `compute_threshold` - Selection threshold (e.g., 10 for top 10 models)
- `exclude_models_containing` - Model name exclusion filters
- `enable_imputation` - Toggle imputation processing
- `imputation_method` - Choose imputation strategy

**Execution Pipeline:**
1. **Data Loading** - Load frontier models, hardware specs, prices
2. **Data Quality Assessment** - Report missing data before/after imputation
3. **Imputation** - Apply selected imputation method if enabled
4. **Cost Estimation** - Run all three estimation methods in parallel
5. **Inflation Adjustment** - Apply CPI adjustment to June 2025
6. **Results Export** - Generate multiple output files

**Output Files:**
- `cost_dataset_3_estimates.csv` - Comparison of all three methods
- `cost_dataset_detailed.csv` - Detailed results for primary method
- `cost_components.csv` - Component cost breakdown
- `cost_estimation_*.out` - Detailed logs for each method

## Data Files

### Core Datasets
- `All ML Systems - full view.csv` - Epoch AI model database with training details
- `Hardware prices.csv` - GPU/TPU pricing data across vendors and time
- `Chip dataset-Grid view.csv` - Hardware technical specifications (FLOP/s, TDP, etc.)
- `PCU518210518210.csv` - Producer price index for inflation adjustment

### Precomputed Model Selections
- `frontier_systems_by_top_n.json` - Top-N frontier model selections
- `frontier_systems_by_window_percentile.json` - Percentile-based selections
- `frontier_systems_by_backward_window_percentile.json` - Backward-looking percentile
- `frontier_systems_by_residual_from_trend.json` - Residual-based selections

## Key Modifiable Parameters for Future Changes

### Economic Assumptions (Lines 17-24)
- `CLUSTER_INTERCONNECT_COST_FRACTION` - Network infrastructure cost percentage
- `MEDIAN_UTILIZATION` - Hardware utilization rate assumption
- `ML_GPU_PRICE_PERFORMANCE_OOMS_PER_YEAR` - Hardware progress rate

### Timing Assumptions (Lines 389-424)
- Training start buffer: 60 days before publication
- Hardware acquisition buffer: 60 days before training start
- Hardware acquisition minimum delay: 90 days after release (GPUs)

### Imputation Strategy (Lines 1284-1317)
- Switch between KNN and most-common imputation
- Adjust KNN neighbor count
- Enable/disable imputation entirely

### Cost Method Selection (Lines 1258-1264)
- Enable/disable specific cost estimation methods
- Change primary method for detailed outputs

### Inflation Target (Line 1376)
- Adjust target date for inflation adjustment
- Currently set to July 2025

### Cloud Pricing (Lines 60-84)
- Update committed-use discount rates
- Modify vendor priority order
- Adjust TPU equivalent prices

### `cost.py` (331 lines)
Main cost estimation pipeline with three approaches:
- `estimate_cloud_costs()` - Cloud rental cost estimation using committed-use discounts
- `estimate_hardware_acquisition_cost()` - Upfront hardware purchase cost calculation
- `estimate_hardware_capex_energy()` - Amortized hardware + energy cost (primary method)
- `estimate_chip_hours()` - Training time estimation from compute/hardware specs
- `cluster_energy_cost()` - Energy consumption and cost calculation

### `data.py` (77 lines)
Data loading and frontier model selection:
- `load_frontier_systems()` - Load precomputed frontier model lists from JSON files
- `load_pcd_df()`, `load_hardware_df()`, `load_price_df()` - CSV data loaders
- `load_data_for_cost_estimation()` - Complete data pipeline for cost analysis

### `hardware.py` (119 lines)
Hardware specifications and server cost calculations:
- `get_flop_per_second()` - Hardware performance lookup from specs database
- `get_server_lifetime()` - Server depreciation timeline (3-5 years by era)
- `get_server_cost_overhead()` - Server vs. GPU cost multipliers (1.54-1.69x)
- Hardware name mapping and GPU aliases for data matching

### `prices.py` (50+ lines visible)
Price data handling and cloud pricing:
- `find_closest_price_dates()` - Temporal price matching for hardware
- Cloud vendor mapping and committed-use discount rates
- TPU equivalent pricing calculations
- Hardware price lookup with vendor priority handling

### `parameters.py` (9 lines)
Global constants and configuration:
- Time conversion constants (SECONDS_PER_HOUR, HOURS_PER_YEAR)
- Hardware economics (CLUSTER_INTERCONNECT_COST_FRACTION = 19%)
- ML hardware trends (ML_GPU_PRICE_PERFORMANCE_OOMS_PER_YEAR = 0.14)
- Default utilization rates (MEDIAN_UTILIZATION = 37.5%)

### `plotting.py` (50+ lines visible)
Visualization utilities for paper figures:
- `human_format()` - Number formatting (K, M, B, T suffixes)
- `set_default_fig_layout()` - Standardized plot styling with Epoch branding
- Plotly-based chart generation with consistent formatting

### `utils.py` (149 lines)
Mathematical and utility functions:
- Date/time conversions (`datetime_to_float_year()`, `float_year_to_datetime()`)
- Growth rate calculations (`ooms_to_factor_per_year()`, `doublings_per_year_to_ooms()`)
- Statistical functions (`geomean()`, `wgeomean()`, lognormal distributions)
- Model selection utilities (`list_models()`, `select_models()`)
- Safe printing for Unicode handling (`print_safely()`)

### Supporting Modules
- `energy.py` - Power consumption and energy cost calculations
- `inflation.py` - CPI adjustment for historical price data
- `imputation.py` - Missing data imputation methods
- `regression.py` - Statistical analysis and trend fitting

## Core Analysis Notebooks

### `cost_analysis.ipynb` - **Primary Analysis Pipeline**
Main cost estimation workflow producing paper results:
- Implements all three cost estimation methods
- Generates regression analysis showing ~2.9x/year cost growth
- Produces cost breakdown by components (chips, servers, interconnect, energy)
- Creates publication-ready plots and datasets

### `cost_analysis_streamlined.ipynb` - **Streamlined Cost Data Update**
Streamlined version of the main cost analysis workflow for regular data updates:
- Runs all three cost estimation methods (hardware-capex-energy, hardware-acquisition, cloud) in a single notebook
- Applies configurable data imputation methods to handle missing data
- Exports results to CSV files without regression analysis or plotting
- Focuses on cost estimation and data generation rather than visualization
- Provides cost component breakdowns and comparison datasets

### `development_cost.ipynb` - **R&D Cost Analysis**
Total model development cost estimation:
- Personnel cost calculations including equity compensation
- Hardware amortization + energy + R&D staff cost integration
- Case studies: GPT-3, OPT-175B, GPT-4, Gemini Ultra

### `ml_model_selection.ipynb` - **Frontier Model Selection**
Implements multiple frontier model identification methods:
- Top-N selection (e.g., top 10 models by compute)
- Percentile-based selection (e.g., top 25% within time windows)
- Residual-based selection (models above trend line)
- Generates JSON files used by other analyses

## Analysis and Validation Notebooks

### `compare_frontier_selections.ipynb`
Comparative analysis of different frontier model selection methods with visualizations

### `compared_results.ipynb` - **Sensitivity Analysis**
Robustness testing across methods and parameters:
- Cost estimate comparisons between different approaches
- Sensitivity analysis for model selection parameters
- Method validation and uncertainty assessment

### `regression_analysis.ipynb`
Focused regression analysis (potentially overlaps with cost_analysis.ipynb)

### `imputation_sensitivity.ipynb`
Testing sensitivity to missing data imputation methods

### `validate_imputation.ipynb`
Validation of data imputation approaches and quality checks

### `data_exploration.ipynb` - **Exploratory Data Analysis**
Comprehensive data analysis:
- Hardware adoption patterns (A100, V100, TPU usage trends)
- GPU price trends and depreciation analysis
- Training time distributions and scaling patterns
- Hardware release date gap analysis

## Specialized Analysis Notebooks

### `uncertainty.ipynb` - **Uncertainty Quantification**
Quantifies uncertainty across cost estimate components:
- Confidence intervals for hardware acquisition costs
- Amortized cost uncertainty analysis
- Energy cost uncertainty assessment

### `h100_manufacturing_cost.ipynb` - **Hardware Cost Analysis**
H100 GPU manufacturing and server cost breakdown based on industry estimates

### `price_data_collection_utility.ipynb`
Data collection and processing utility for hardware pricing information

### `cud_estimate.ipynb`
Cloud committed-use discount (CUD) analysis for major cloud providers

## Data Files

### Core Datasets
- `All ML Systems - full view.csv` - Epoch AI model database snapshot with training details
- `Hardware prices.csv` - GPU/TPU pricing data across vendors and time
- `Chip dataset-Grid view.csv` - Hardware technical specifications (FLOP/s, TDP, etc.)
- `PCU518210518210.csv` - Producer price index for inflation adjustment

### Precomputed Model Selections
- `frontier_systems_by_top_n.json` - Top-N frontier model selections
- `frontier_systems_by_window_percentile.json` - Percentile-based selections
- `frontier_systems_by_backward_window_percentile.json` - Backward-looking percentile
- `frontier_systems_by_residual_from_trend.json` - Residual-based selections
- `gpt-4_contributions.json` - GPT-4 specific analysis data

## Configuration Files

### `requirements.txt` (150 lines)
Python dependencies including:
- Core: pandas, numpy, scipy, matplotlib, plotly
- ML: scikit-learn, statsmodels
- Notebooks: jupyter, ipython
- Web: requests, selenium (for data collection)
- Many potentially unused packages

### `LICENSE`
Project license file

### `README.md`
Basic setup and usage instructions with parameter configuration examples

## Key Workflows

1. **Main Analysis**: `ml_model_selection.ipynb` → `cost_analysis.ipynb` → paper results
2. **Development Costs**: `development_cost.ipynb` → R&D cost estimates
3. **Validation**: `compared_results.ipynb`, `uncertainty.ipynb` → robustness testing
4. **Data Quality**: `imputation_sensitivity.ipynb`, `validate_imputation.ipynb` → data validation