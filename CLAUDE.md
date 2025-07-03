## Introduction

Our current goal is to create a standalone "update_model_costs.ipynb" notebook that we can run within the codebase every time we want updated cost data. So we shouldn't be running regressions or plotting graphs, or recalculating all of our model parameters each time.

## Tips
Remember to use NotebookEdit to edit .ipynb files.

# Training Cost Trends Codebase Structure

This repository analyzes the rising costs of training frontier AI models, supporting the Epoch Research paper "The rising costs of training frontier AI models". 

## Core Python Modules

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

## Potential Simplifications

- `regression_analysis.ipynb` may be redundant with `cost_analysis.ipynb`
- Data validation notebooks could be consolidated
- Many dependencies in `requirements.txt` appear unused
- Utility notebooks could be moved to scripts/