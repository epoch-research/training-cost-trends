"""Data loading functions for models, hardware specs, and prices."""

import logging
from pathlib import Path

import pandas as pd

from .data_sources import AirtablePath, airtable_to_df, get_airtable_table

logger = logging.getLogger(__name__)


def load_data(
    path: Path | AirtablePath,
    airtable_token: str | None = None,
    columns_to_resolve: list[str] | None = None,
) -> pd.DataFrame:
    """Load data from a local CSV or an Airtable table."""
    if columns_to_resolve is None:
        columns_to_resolve = []

    if isinstance(path, AirtablePath):
        if not airtable_token:
            raise ValueError("Airtable token is required to load data from Airtable.")
        table = get_airtable_table(airtable_token, path.app_id, path.table_id)
        return airtable_to_df(table, columns_to_resolve=columns_to_resolve)
    else:
        return pd.read_csv(path)


def load_pcd_data(path: Path | AirtablePath, airtable_token: str | None = None) -> pd.DataFrame:
    """Load the Parameter, Compute, and Data (PCD) models dataset."""
    pcd_df = load_data(path, airtable_token, ["Training hardware", "Organization"]).astype(
        {"Training compute (FLOP)": "float64"}
    )
    pcd_df.dropna(subset=["Publication date"], inplace=True)
    pcd_df["Organization"] = pcd_df["Organization"].fillna("")
    pcd_df["Publication date"] = pd.to_datetime(pcd_df["Publication date"])
    return pcd_df


def load_hardware_data(path: Path | AirtablePath, airtable_token: str | None = None) -> pd.DataFrame:
    """Load hardware technical specifications dataset."""
    return load_data(path, airtable_token)


def load_price_data(path: Path | AirtablePath, airtable_token: str | None = None) -> pd.DataFrame:
    """Load hardware pricing dataset."""
    price_df = load_data(path, airtable_token, ["Hardware model"])
    price_df.dropna(subset=["Price date"], inplace=True)
    price_df["Price date"] = pd.to_datetime(price_df["Price date"])
    return price_df
