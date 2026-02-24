"""CLI entry point for the training cost trends analysis."""

import argparse
import logging

from .constants import (
    DEFAULT_COMPUTE_THRESHOLD,
    HARDWARE_FILE,
    HARDWARE_PRICE_FILE,
    MODELS_FILE,
    PRICE_INDEX_FILE,
)
from .data_sources import parse_airtable_path, parse_local_or_airtable_path, parse_local_or_fred_path
from .main import main
from .utils import relpath


def cli() -> None:
    """Parse CLI arguments and run the cost analysis."""
    parser = argparse.ArgumentParser(
        description="Estimate training costs of frontier AI models.",
    )

    airtable_path_format = "airtable://app<id>/tbl<id>"

    parser.add_argument(
        "--price-index-data",
        type=parse_local_or_fred_path,
        default=PRICE_INDEX_FILE,
        metavar="LOCAL_PATH|FRED_URL",
        help=f"Producer Price Index data (local path or fred://<series_id>) [default: {relpath(PRICE_INDEX_FILE)}]",
    )

    parser.add_argument(
        "--compute-threshold",
        type=int,
        default=DEFAULT_COMPUTE_THRESHOLD,
        metavar="N",
        help=f"Compute threshold for selecting frontier models (e.g. 5 to select top 5) [default: {DEFAULT_COMPUTE_THRESHOLD}]",
    )

    parser.add_argument(
        "--models-data",
        type=parse_local_or_airtable_path,
        default=MODELS_FILE,
        metavar="LOCAL_PATH|AIRTABLE_URL",
        help=f"Models (PCD) data (local path or {airtable_path_format}) [default: {relpath(MODELS_FILE)}]",
    )

    parser.add_argument(
        "--hardware-data",
        type=parse_local_or_airtable_path,
        default=HARDWARE_FILE,
        metavar="LOCAL_PATH|AIRTABLE_URL",
        help=f"Hardware data (local path or {airtable_path_format}) [default: {relpath(HARDWARE_FILE)}]",
    )

    parser.add_argument(
        "--hardware-price-data",
        type=parse_local_or_airtable_path,
        default=HARDWARE_PRICE_FILE,
        metavar="LOCAL_PATH|AIRTABLE_URL",
        help=f"Hardware price data (local path or {airtable_path_format}) [default: {relpath(HARDWARE_PRICE_FILE)}]",
    )

    parser.add_argument(
        "--update-table",
        type=parse_airtable_path,
        required=False,
        metavar="AIRTABLE_PATH",
        help=f"Table to write the results to ({airtable_path_format}) [default: None]",
    )

    # New configuration arguments (previously hardcoded)
    parser.add_argument(
        "--threshold-method",
        choices=["top_n", "window_percentile"],
        default="top_n",
        help="Method for selecting frontier models [default: top_n]",
    )

    parser.add_argument(
        "--variant",
        default="2025-03-17_exclude_finetunes_at_threshold_stage",
        help="Variant label for the results directory [default: 2025-03-17_exclude_finetunes_at_threshold_stage]",
    )

    parser.add_argument(
        "--imputation-method",
        choices=["knn", "most_common", "none"],
        default="most_common",
        help="Imputation method for missing data [default: most_common]",
    )

    parser.add_argument(
        "--no-imputation",
        action="store_true",
        help="Disable data imputation entirely",
    )

    parser.add_argument(
        "--knn-neighbors",
        type=int,
        default=5,
        metavar="K",
        help="Number of neighbors for KNN imputation [default: 5]",
    )

    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=[],
        metavar="KEYWORD",
        help="Exclude models whose names contain these keywords",
    )

    parser.add_argument(
        "--methods",
        nargs="*",
        choices=["hardware-capex-energy", "hardware-acquisition", "cloud"],
        default=["hardware-capex-energy", "hardware-acquisition", "cloud"],
        metavar="METHOD",
        help="Cost estimation methods to run [default: all three]",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s",
    )

    imputation_method = "none" if args.no_imputation else args.imputation_method

    main(
        price_index_path=args.price_index_data,
        models_path=args.models_data,
        hardware_path=args.hardware_data,
        hardware_price_path=args.hardware_price_data,
        update_table_path=args.update_table,
        compute_threshold=args.compute_threshold,
        threshold_method=args.threshold_method,
        variant=args.variant,
        imputation_method=imputation_method,
        knn_neighbors=args.knn_neighbors,
        exclude_models=args.exclude_models,
        estimation_methods=args.methods,
    )


if __name__ == "__main__":
    cli()
