"""Utility functions for date conversion, statistics, and model selection."""

import logging

import numpy as np
import pandas as pd

from .constants import DEFAULT_COMPUTE_THRESHOLD

logger = logging.getLogger(__name__)


def datetime_to_float_year(datetimes: pd.Series) -> pd.Series:
    """Convert a Series of datetimes to fractional year floats."""
    return datetimes.dt.year + (datetimes.dt.month - 1) / 12 + (datetimes.dt.day - 1) / 365


def float_year_to_datetime(float_year: float) -> pd.Timestamp:
    """Convert a fractional year float to a Timestamp."""
    year = int(float_year)
    remainder = float_year - year
    days_in_year = 365 + int(pd.Timestamp(year=year, month=12, day=31).is_leap_year)
    day_of_year = int(remainder * days_in_year)
    return pd.Timestamp(year=year, month=1, day=1) + pd.to_timedelta(day_of_year, unit="D")


def print_safely(input_string: str) -> None:
    """Log strings, handling unicode encoding errors gracefully."""
    try:
        logger.info(input_string)
    except UnicodeEncodeError:
        encoded_string = "".join(
            [char if char.encode("ascii", "ignore") else f"\\u{ord(char):04x}" for char in input_string]
        )
        logger.info(
            "Could not log original string due to encoding error. Logging modified version: %s",
            encoded_string,
        )


def geomean(arr: np.ndarray) -> float:
    """Compute the geometric mean of an array."""
    return np.exp(np.mean(np.log(arr)))


def wgeomean(arr: np.ndarray, weights: np.ndarray) -> float:
    """Compute the weighted geometric mean of an array."""
    return np.exp(np.average(np.log(arr), weights=weights))


def get_top_models(models: pd.DataFrame, n: int = DEFAULT_COMPUTE_THRESHOLD) -> pd.DataFrame:
    """Return the top n models by training compute."""
    return models[models["Compute rank when published"] <= n].copy()


def relpath(p: "Path") -> str:
    """Return a relative path string from the current working directory."""
    from pathlib import Path

    return str(p.resolve().relative_to(Path.cwd().resolve()))
