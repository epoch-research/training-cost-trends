import numpy as np
import pandas as pd


def datetime_to_float_year(datetimes):
    date_floats = datetimes.dt.year + (datetimes.dt.month-1) / 12 + (datetimes.dt.day-1) / 365
    return date_floats


def float_year_to_datetime(float_year):
    year = int(float_year)
    remainder = float_year - year
    days_in_year = 365 + int(pd.Timestamp(year=year, month=12, day=31).is_leap_year)
    day_of_year = int(remainder * days_in_year)
    return pd.Timestamp(year=year, month=1, day=1) + pd.to_timedelta(day_of_year, unit='D')


def ooms_to_factor_per_year(ooms):
    return 10 ** ooms


def factor_per_year_to_ooms(factor_per_year):
    return np.log10(factor_per_year)


def ooms_to_doublings_per_year(ooms):
    return ooms * np.log2(10)


def doublings_per_year_to_ooms(doublings_per_year):
    return doublings_per_year / np.log2(10)


def ooms_to_doubling_time_months(ooms):
    doubling_time_years = 1 / ooms_to_doublings_per_year(ooms)
    return doubling_time_years * 12


def doubling_time_months_to_ooms(doubling_time_months):
    doubling_time_years = doubling_time_months / 12
    return doublings_per_year_to_ooms(1 / doubling_time_years)
