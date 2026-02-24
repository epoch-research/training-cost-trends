"""Tests for data source path parsing."""

import argparse
from pathlib import Path

import pytest

from training_cost_trends.data_sources import (
    AirtablePath,
    FredPath,
    parse_airtable_path,
    parse_local_or_airtable_path,
    parse_local_or_fred_path,
)


class TestFredPath:
    def test_get_url(self):
        path = FredPath(series="PCU518210518210")
        assert path.get_url() == "fred://PCU518210518210"


class TestAirtablePath:
    def test_get_url(self):
        path = AirtablePath(app_id="appABC123", table_id="tblXYZ789")
        assert path.get_url() == "airtable://appABC123/tblXYZ789"


class TestParseAirtablePath:
    def test_valid_airtable_url(self):
        result = parse_airtable_path("airtable://appABC123/tblXYZ789")
        assert isinstance(result, AirtablePath)
        assert result.app_id == "appABC123"
        assert result.table_id == "tblXYZ789"

    def test_invalid_format_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_airtable_path("not-a-url")

    def test_invalid_format_no_raise(self):
        result = parse_airtable_path("not-a-url", raise_exception=False)
        assert result is None


class TestParseLocalOrFredPath:
    def test_fred_url(self):
        result = parse_local_or_fred_path("fred://PCU518210518210")
        assert isinstance(result, FredPath)
        assert result.series == "PCU518210518210"

    def test_local_csv_path(self):
        result = parse_local_or_fred_path("data/some_file.csv")
        assert isinstance(result, Path)

    def test_invalid_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_local_or_fred_path(":::invalid:::")


class TestParseLocalOrAirtablePath:
    def test_airtable_url(self):
        result = parse_local_or_airtable_path("airtable://appABC/tblXYZ")
        assert isinstance(result, AirtablePath)

    def test_local_csv_path(self):
        result = parse_local_or_airtable_path("data/some_file.csv")
        assert isinstance(result, Path)
