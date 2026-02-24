# Introduction

This repository analyzes the rising costs of training frontier AI models, supporting the Epoch Research paper "The rising costs of training frontier AI models". The codebase is organized as a Python package (`training_cost_trends/`) with a backward-compatible entry point (`cost_analysis_standalone.py`).

# Tips
- Use `python3` rather than `python` for relevant bash commands
- When you see an error, for example from using a bash command, write it to "ERRORS.md" with a full comment on the situation so that I can troubleshoot for future sessions. The description should be roughly between 1 compound sentence and 2 short paragraphs in length.

# Package Structure

## `training_cost_trends/` — Main Package

### `constants.py` — Constants and Configuration
All economic constants, hardware mappings, cloud vendor configuration, and file paths.

Key constants:
- `CLUSTER_INTERCONNECT_COST_FRACTION = 0.19` (19% of total cluster cost)
- `MEDIAN_UTILIZATION = 0.375` (37.5% median utilization rate)
- `ML_GPU_PRICE_PERFORMANCE_OOMS_PER_YEAR = 0.14` (hardware progress rate)
- `SIMPLIFIED_HARDWARE_NAMES` — Maps GPU variants to simplified names
- `GPU_HARDWARE_ALIASES` — Common GPU aliases for matching
- `TPU_EQUIVALENT_RELEASE_PRICES` — Fixed TPU pricing estimates
- `DEFAULT_CUD` — Committed-use discount rates by cloud provider
- `PRIORITY_VENDORS` — Vendor preference order for price lookups
- `ORG_TO_CLOUD_VENDOR` — Organization-to-cloud-vendor mapping

### `utils.py` — Utility Functions
Date conversion (`datetime_to_float_year`, `float_year_to_datetime`), geometric means, `get_top_models()`, and safe unicode printing.

### `data_sources.py` — Data Source Abstractions
- `FredPath`, `AirtablePath` dataclasses for data source URLs
- Path parsers for CLI argument handling
- Airtable I/O (`airtable_to_df`, `write_costs_to_airtable`)
- FRED API client (`get_fred_df`)

### `data_loading.py` — Data Loaders
- `load_pcd_data()` — Epoch AI model database
- `load_hardware_data()` — Hardware technical specifications
- `load_price_data()` — Hardware pricing data
- All loaders support both local CSV and Airtable sources

### `hardware.py` — Hardware Specifications and Energy
- `get_flop_per_second()` — Performance lookup (Tensor-FP16/BF16 > TF32 > FP16 > FP32)
- `get_server_lifetime()` — Server depreciation timeline (3-5 years by era)
- `get_server_cost_overhead()` — Server vs. GPU cost multipliers
- `power_usage_effectiveness()` — PUE by organization type
- `energy_price()` — US industrial electricity prices by year
- `cluster_power_capacity()` — Training cluster power requirements

### `pricing.py` — Price Discovery and Inflation
- `find_closest_price_dates()` — Temporal price matching with soft matching
- `get_training_start_date()` — Training timeline estimation
- `find_price()` — Multi-tier fallback across vendors and CUD levels
- `depreciate_by_hardware_progress()` — Hardware value depreciation
- `adjust_column_for_inflation()` — Producer price index adjustment

### `imputation.py` — Data Imputation
- `knn_impute_pcd()` — Full KNN imputation pipeline (default: 5 neighbors)
- `most_common_impute_training_hardware()` — Year-based mode imputation

### `cost_estimation.py` — Three Cost Estimation Methods
- **Method 1** (`estimate_hardware_capex_energy`): Amortized hardware + energy cost
- **Method 2** (`estimate_hardware_acquisition_cost`): Upfront hardware purchase cost
- **Method 3** (`estimate_cloud_costs`): Cloud rental cost with committed-use discounts
- Shared pipeline helper `_run_estimation()` reduces duplication

### `main.py` — Orchestration
- `main()` — Coordinates loading, imputation, estimation, inflation adjustment, and export
- `load_all_data()` — Unified data loading
- `run_estimations()` — Runs all selected methods with file logging
- `export_results()` — CSV export and optional Airtable upload

### `__main__.py` — CLI Entry Point
Argparse-based CLI supporting all configuration as command-line arguments.

## `cost_analysis_standalone.py` — Backward-Compatible Entry Point
Thin wrapper that imports and runs the CLI from the package. Preserves compatibility with the CI workflow.

## Data Files

### Core Datasets (in `data/` directory)
- `All ML Systems - full view.csv` — Epoch AI model database with training details
- `Hardware prices.csv` — GPU/TPU pricing data across vendors and time
- `Chip dataset-Grid view.csv` — Hardware technical specifications (FLOP/s, TDP, etc.)
- `PCU518210518210.csv` — Producer price index for inflation adjustment

### Precomputed Model Selections (in `claude-data/` directory)
- `frontier_systems_by_window_percentile.json` — Percentile-based frontier selections
- `frontier_systems_by_backward_window_percentile.json` — Backward-looking percentile
- `frontier_systems_by_residual_from_trend.json` — Residual-based selections
- `gpt-4_contributions.json` — GPT-4 specific analysis data

## Output Files

Results are saved to `results/` with subdirectories per configuration:
- `cost_dataset_3_estimates.csv` — All three cost methods compared
- `cost_dataset_detailed.csv` — Detailed results for the primary method
- `cost_components.csv` — Component cost breakdown
- `cost_estimation_*.out` — Detailed logs for each method

## Usage

```bash
# Basic execution (uses local CSV data)
uv run cost_analysis_standalone.py

# List all options
uv run cost_analysis_standalone.py --help

# Run with specific options
uv run cost_analysis_standalone.py --imputation-method knn --knn-neighbors 10 --compute-threshold 10

# Run as a package
python -m training_cost_trends --help

# Run tests
uv run pytest
```
