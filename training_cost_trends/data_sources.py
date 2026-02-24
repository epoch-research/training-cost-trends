"""Data source abstractions: FredPath, AirtablePath, path parsing, Airtable I/O, and FRED API."""

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from pyairtable import Api

logger = logging.getLogger(__name__)


# ==============================================================================
# PATH DATACLASSES
# ==============================================================================


@dataclass
class FredPath:
    """Represents a FRED API data source (fred://<series_id>)."""

    series: str

    def get_url(self) -> str:
        return f"fred://{self.series}"


@dataclass
class AirtablePath:
    """Represents an Airtable data source (airtable://app<id>/tbl<id>)."""

    app_id: str
    table_id: str

    def get_url(self) -> str:
        return f"airtable://{self.app_id}/{self.table_id}"


# ==============================================================================
# PATH PARSERS
# ==============================================================================


def parse_local_or_fred_path(value: str) -> Path | FredPath:
    """Parse a string to determine if it's a file path or a fred://<series_id> URL."""
    fred_pattern = r"^fred://([A-Za-z0-9]+)$"
    match = re.match(fred_pattern, value)

    if match:
        return FredPath(series=match.group(1))

    try:
        path = Path(value)
        if path.exists() or path.suffix:
            return path
    except Exception:
        pass

    raise argparse.ArgumentTypeError(
        f"Input must be either a valid file path or fred://<series_id> format. Got: {value}"
    )


def parse_airtable_path(value: str, raise_exception: bool = True) -> AirtablePath | None:
    """Parse a airtable://app<id>/tbl<id> URL."""
    airtable_pattern = r"^airtable://(app[A-Za-z0-9]+)/(tbl[A-Za-z0-9]+)$"
    match = re.match(airtable_pattern, value)

    if match:
        return AirtablePath(app_id=match.group(1), table_id=match.group(2))

    if raise_exception:
        raise argparse.ArgumentTypeError(
            f"Input must be in airtable://app<id>/tbl<id> format. Got: {value}"
        )
    return None


def parse_local_or_airtable_path(value: str) -> Path | AirtablePath:
    """Parse a string to determine if it's a file path or an airtable://app<id>/tbl<id> URL."""
    airtable_path = parse_airtable_path(value, raise_exception=False)
    if airtable_path:
        return airtable_path

    try:
        path = Path(value)
        if path.exists() or path.suffix:
            return path
    except Exception:
        pass

    raise argparse.ArgumentTypeError(
        f"Input must be either a valid file path or airtable://app<id>/tbl<id> format. Got: {value}"
    )


# ==============================================================================
# FRED API
# ==============================================================================


def get_fred_df(api_key: str, path: FredPath) -> pd.DataFrame:
    """Fetch data from the FRED API."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "api_key": api_key,
        "series_id": path.series,
        "file_type": "json",
    }

    response = requests.get(url, params=params)
    data = response.json()
    return pd.DataFrame(data["observations"])


# ==============================================================================
# AIRTABLE I/O
# ==============================================================================


def get_airtable_table(api_key: str, app_id: str, table_id: str):
    """Get an Airtable table object."""
    api = Api(api_key)
    return api.table(app_id, table_id)


def airtable_to_df(table, columns_to_resolve: list[str] | None = None) -> pd.DataFrame:
    """Convert an Airtable table to a DataFrame, resolving linked record dependencies."""
    if columns_to_resolve is None:
        columns_to_resolve = []

    logger.info("Resolving Airtable dependencies for table %s", table)

    dependency_fields = {
        field.name: field
        for field in table.schema().fields
        if field.type == "multipleRecordLinks" and field.name in columns_to_resolve
    }

    dependency_record_values: dict[str, dict[str, str]] = {}
    for field in dependency_fields.values():
        dependency_table = table.api.table(table.base.id, field.options.linked_table_id)
        primary_id = dependency_table.schema().primary_field_id

        logger.info("Resolving linked records for field %s from table %s", field.name, dependency_table)

        records = {}
        for record in dependency_table.all(fields=[primary_id]):
            id_list = list(record["fields"].values())
            if len(id_list) == 0:
                continue
            records[record["id"]] = id_list[0]
        dependency_record_values[field.name] = records

    record_fields = []
    record_ids = []

    records = table.all()
    for record in records:
        row = record["fields"]

        for column in columns_to_resolve:
            if column not in row:
                continue
            row[column] = [
                dependency_record_values[column][linked_record_id]
                for linked_record_id in row[column]
            ]

        # Handle lists by joining into comma-separated strings
        for column, value in row.items():
            if isinstance(value, list):
                row[column] = ",".join([str(v) if v is not None else "" for v in value])

        record_fields.append(row)
        record_ids.append(record["id"])

    return pd.DataFrame(record_fields, index=record_ids)


def write_costs_to_airtable(costs: pd.DataFrame, table) -> None:
    """Write cost estimates back to an Airtable table."""
    updates = []

    records = table.all(fields=["Model"])
    name_to_id = {
        record["fields"]["Model"]: record["id"]
        for record in records
        if "Model" in record["fields"]
    }

    for _, row in costs.iterrows():
        if row["Model"] not in name_to_id:
            logger.warning("Could not find %s in Airtable table", row["Model"])
            continue

        sanitize = lambda x: None if pd.isna(x) else x
        updates.append(
            {
                "id": name_to_id[row["Model"]],
                "fields": {
                    "Training compute cost (2023 USD)": sanitize(row["hardware_capex_energy_cost"]),
                    "Training compute cost (cloud)": sanitize(row["cloud_cost"]),
                    "Training compute cost (upfront)": sanitize(row["hardware_acquisition_cost"]),
                },
            }
        )

    table.batch_update(updates)
