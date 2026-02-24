"""Main orchestration function for the cost analysis workflow."""

import logging
import os
from functools import partial
from pathlib import Path

import pandas as pd

from .constants import (
    DEFAULT_COMPUTE_THRESHOLD,
    PRICE_INDEX_SERIES,
)
from .cost_estimation import (
    estimate_cloud_costs,
    estimate_hardware_acquisition_cost,
    estimate_hardware_capex_energy,
)
from .data_loading import load_hardware_data, load_pcd_data, load_price_data
from .data_sources import AirtablePath, FredPath, get_airtable_table, get_fred_df, write_costs_to_airtable
from .hardware import cluster_power_capacity
from .imputation import knn_impute_pcd, most_common_impute_training_hardware
from .pricing import adjust_column_for_inflation
from .utils import datetime_to_float_year, get_top_models

logger = logging.getLogger(__name__)


# ==============================================================================
# HELPERS
# ==============================================================================


def load_all_data(
    price_index_path: Path | FredPath,
    models_path: Path | AirtablePath,
    hardware_path: Path | AirtablePath,
    hardware_price_path: Path | AirtablePath,
    airtable_token: str | None,
    fred_api_key: str | None,
    compute_threshold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all data sources and return (full_pcd_df, frontier_pcd_df, price_df, hardware_df, price_index)."""
    logger.info("Loading data...")
    full_pcd_df = load_pcd_data(models_path, airtable_token)
    price_df = load_price_data(hardware_price_path, airtable_token)
    hardware_df = load_hardware_data(hardware_path, airtable_token)
    frontier_pcd_df = get_top_models(full_pcd_df, compute_threshold)

    if isinstance(price_index_path, Path):
        price_index = pd.read_csv(price_index_path)
    else:
        fred_data = get_fred_df(fred_api_key, price_index_path)
        price_index = fred_data.rename(
            columns={"date": "observation_date", "value": PRICE_INDEX_SERIES}
        )[["observation_date", PRICE_INDEX_SERIES]].astype(
            {"observation_date": "string", PRICE_INDEX_SERIES: "float"}
        )

    logger.info(
        "Loaded %d frontier models, %d hardware entries, %d price entries",
        len(frontier_pcd_df), len(hardware_df), len(price_df),
    )
    return full_pcd_df, frontier_pcd_df, price_df, hardware_df, price_index


def report_data_quality(df: pd.DataFrame, label: str) -> None:
    """Print a data quality report for the given DataFrame."""
    logger.info("Data Quality Report (%s):", label)
    for col in ["Training hardware", "Hardware quantity", "Hardware utilization (MFU)", "Training time (hours)"]:
        if col in df.columns:
            logger.info(
                "  Models with known %s: %d/%d", col, df[col].notna().sum(), len(df)
            )


def build_imputation_fn(
    imputation_method: str,
    full_pcd_df: pd.DataFrame,
    knn_neighbors: int,
) -> tuple:
    """Build the imputation function and kwargs based on the chosen method.

    Returns (impute_pcd_fn, impute_kwargs) tuple.
    """
    if imputation_method == "knn":
        return knn_impute_pcd, {"num_neighbors": knn_neighbors}
    elif imputation_method == "most_common":
        return partial(most_common_impute_training_hardware, full_pcd_df), {}
    else:
        return None, {}


def run_estimations(
    frontier_pcd_df: pd.DataFrame,
    hardware_df: pd.DataFrame,
    price_df: pd.DataFrame,
    estimation_methods: list[str],
    results_dir: str,
    impute_pcd_fn=None,
    impute_kwargs: dict | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    """Run all cost estimation methods and return (cost_dfs, component_cost_df)."""
    if impute_kwargs is None:
        impute_kwargs = {}

    estimation_method_lookup = {
        "hardware-capex-energy": estimate_hardware_capex_energy,
        "hardware-acquisition": estimate_hardware_acquisition_cost,
        "cloud": estimate_cloud_costs,
    }

    cost_dfs: dict[str, pd.DataFrame] = {}
    component_cost_df = None

    for estimation_method in estimation_methods:
        logger.info("Running %s estimation", estimation_method)
        cost_estimation_function = estimation_method_lookup[estimation_method]

        # Set up file logging for this estimation method
        file_handler = logging.FileHandler(f"{results_dir}/cost_estimation_{estimation_method}.out", mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger = logging.getLogger("training_cost_trends")
        root_logger.addHandler(file_handler)

        try:
            if impute_pcd_fn is not None:
                cost_df = cost_estimation_function(
                    frontier_pcd_df.copy(), hardware_df, price_df,
                    impute_pcd_fn=impute_pcd_fn, **impute_kwargs,
                )
            else:
                cost_df = cost_estimation_function(frontier_pcd_df.copy(), hardware_df, price_df)
        finally:
            root_logger.removeHandler(file_handler)
            file_handler.close()

        cost_dfs[estimation_method] = cost_df

        # Create component cost breakdown only for hardware-capex-energy method
        if estimation_method == "hardware-capex-energy":
            file_handler = logging.FileHandler(f"{results_dir}/component_cost_estimation.out", mode="w")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            root_logger.addHandler(file_handler)

            try:
                if impute_pcd_fn is not None:
                    component_cost_df = cost_estimation_function(
                        frontier_pcd_df.copy(), hardware_df, price_df,
                        separate_components=True, impute_pcd_fn=impute_pcd_fn, **impute_kwargs,
                    )
                else:
                    component_cost_df = cost_estimation_function(
                        frontier_pcd_df.copy(), hardware_df, price_df, separate_components=True,
                    )
            finally:
                root_logger.removeHandler(file_handler)
                file_handler.close()

    return cost_dfs, component_cost_df


def export_results(
    cost_dfs: dict[str, pd.DataFrame],
    component_cost_df: pd.DataFrame | None,
    estimation_methods: list[str],
    exclude_models_containing: list[str],
    price_index: pd.DataFrame,
    hardware_df: pd.DataFrame,
    results_dir: str,
) -> pd.DataFrame:
    """Apply exclusions, inflation adjustment, and export results to CSV. Returns cost_comparison_df."""
    # Display results for each method
    for method, df in cost_dfs.items():
        logger.info("=== %s results ===", method)
        logger.info("  Total models: %d", len(df))
        logger.info("  Models with cost estimates: %d", df["Cost"].notna().sum())
        if df["Cost"].notna().any():
            logger.info("  Cost range: $%.0f - $%.0f", df["Cost"].min(), df["Cost"].max())

    # Apply exclusions
    for method in estimation_methods:
        for kw in exclude_models_containing:
            cost_dfs[method] = cost_dfs[method][cost_dfs[method]["Model"].str.contains(kw) == False]

    # Apply inflation adjustment
    for method in estimation_methods:
        cost_dfs[method] = adjust_column_for_inflation(cost_dfs[method], "Cost", price_index, "2025-06-01")

    # Create 3-method comparison
    cost_comparison_df = pd.DataFrame()
    cost_comparison_df["Model"] = cost_dfs["hardware-capex-energy"]["Model"]
    for method in estimation_methods:
        method_df = cost_dfs[method]
        cost_comparison_df[f'{method.replace("-", "_")}_cost'] = method_df["Cost (inflation-adjusted)"]

    logger.info("Cost comparison across methods (first 10 non-null):")
    logger.info("\n%s", cost_comparison_df.dropna().head(10))

    cost_comparison_df.to_csv(results_dir + "cost_dataset_3_estimates.csv", index=False)
    logger.info("Saved cost_dataset_3_estimates.csv with %d models", len(cost_comparison_df))

    # Detailed export for hardware-capex-energy method
    cost_df = cost_dfs["hardware-capex-energy"]
    keep_cols = [
        "Model", "Domain", "Task", "Model accessibility", "Reference", "Publication date",
        "Organization", "Parameters", "Training compute (FLOP)", "Training dataset size (datapoints)",
        "Epochs", "Training time (hours)", "Training hardware", "Base model",
        "Finetune compute (FLOP)", "Hardware quantity", "Hardware utilization (MFU)",
        "Training cloud compute vendor", "Training data center", "Cost", "Cost (inflation-adjusted)",
    ]
    existing_cols = [col for col in keep_cols if col in cost_df.columns]
    cost_df[existing_cols].to_csv(results_dir + "cost_dataset_detailed.csv", index=False)

    # Handle component costs
    if component_cost_df is not None:
        cost_component_names = [
            "AI accelerator chip cost",
            "Other server components cost",
            "Cluster-level interconnect cost",
            "Energy cost",
        ]
        for key in cost_component_names:
            if key in component_cost_df.columns:
                component_cost_df[f"{key} (%)"] = component_cost_df[key] / component_cost_df["Cost"] * 100

        cost_component_pc_names = [name + " (%)" for name in cost_component_names]
        existing_pc_cols = [col for col in cost_component_pc_names if col in component_cost_df.columns]

        if existing_pc_cols:
            filtered = component_cost_df.dropna(subset=existing_pc_cols).sort_values(by="Publication date")
            filtered.to_csv(results_dir + "cost_components.csv", index=False)

            logger.info("Average component percentages:")
            logger.info("\n%s", filtered[existing_pc_cols].mean())

            if "Training hardware" in filtered.columns and "Hardware quantity" in filtered.columns:
                filtered = filtered.dropna(subset=["Training hardware"])
                power_col = "Power capacity for final training run (kW)"
                filtered.loc[:, power_col] = [
                    cluster_power_capacity(
                        row["Training hardware"], row["Hardware quantity"], hardware_df, row["Organization"]
                    )
                    for _, row in filtered.iterrows()
                ]
                filtered["Publication date (float)"] = datetime_to_float_year(
                    pd.to_datetime(filtered["Publication date"])
                )

    return cost_comparison_df


# ==============================================================================
# MAIN
# ==============================================================================


def main(
    price_index_path: Path | FredPath,
    models_path: Path | AirtablePath,
    hardware_path: Path | AirtablePath,
    hardware_price_path: Path | AirtablePath,
    update_table_path: AirtablePath | None = None,
    compute_threshold: int = DEFAULT_COMPUTE_THRESHOLD,
    threshold_method: str = "top_n",
    variant: str = "2025-03-17_exclude_finetunes_at_threshold_stage",
    imputation_method: str = "most_common",
    knn_neighbors: int = 5,
    exclude_models: list[str] | None = None,
    estimation_methods: list[str] | None = None,
) -> None:
    """Run the full cost analysis workflow."""
    if exclude_models is None:
        exclude_models = []
    if estimation_methods is None:
        estimation_methods = ["hardware-capex-energy", "hardware-acquisition", "cloud"]

    airtable_token = os.environ.get("AIRTABLE_TOKEN")
    fred_api_key = os.environ.get("FRED_API_KEY")

    if isinstance(price_index_path, FredPath) and not fred_api_key:
        raise ValueError("Missing `FRED_API_KEY` environment variable")

    using_airtable = (
        isinstance(models_path, AirtablePath)
        or isinstance(hardware_path, AirtablePath)
        or isinstance(hardware_price_path, AirtablePath)
        or update_table_path
    )
    if using_airtable and not airtable_token:
        raise ValueError("Missing `AIRTABLE_TOKEN` environment variable")

    results_dir = f"results/all-methods-{threshold_method}={compute_threshold}-{variant}/"
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    full_pcd_df, frontier_pcd_df, price_df, hardware_df, price_index = load_all_data(
        price_index_path, models_path, hardware_path, hardware_price_path,
        airtable_token, fred_api_key, compute_threshold,
    )

    report_data_quality(frontier_pcd_df, "Before Imputation")

    # Apply imputation if enabled
    enable_imputation = imputation_method not in ("none", None)
    if enable_imputation:
        logger.info("Applying %s imputation...", imputation_method)
        if imputation_method == "knn":
            frontier_pcd_df = knn_impute_pcd(frontier_pcd_df.copy(), num_neighbors=knn_neighbors)
            logger.info("Applied KNN imputation with %d neighbors", knn_neighbors)
        elif imputation_method == "most_common":
            frontier_pcd_df = most_common_impute_training_hardware(full_pcd_df, frontier_pcd_df.copy())
            logger.info("Applied most common value imputation for training hardware")

        report_data_quality(frontier_pcd_df, "After Imputation")
    else:
        logger.info("Skipping imputation (disabled)")

    # Build imputation function for estimation pipelines
    impute_pcd_fn, impute_kwargs = build_imputation_fn(
        imputation_method if enable_imputation else "none", full_pcd_df, knn_neighbors,
    )

    # Run estimations
    cost_dfs, component_cost_df = run_estimations(
        frontier_pcd_df, hardware_df, price_df, estimation_methods, results_dir,
        impute_pcd_fn, impute_kwargs,
    )

    # Export results
    cost_comparison_df = export_results(
        cost_dfs, component_cost_df, estimation_methods, exclude_models,
        price_index, hardware_df, results_dir,
    )

    logger.info("Cost analysis complete! Results saved to %s", results_dir)

    # Optionally upload to Airtable
    if update_table_path is not None:
        logger.info("Uploading results to %s...", update_table_path.get_url())
        table = get_airtable_table(airtable_token, update_table_path.app_id, update_table_path.table_id)
        write_costs_to_airtable(cost_comparison_df, table)
        logger.info("Results uploaded to Airtable successfully!")
