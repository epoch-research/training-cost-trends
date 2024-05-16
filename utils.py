import numpy as np
import pandas as pd
from scipy import stats


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


def printg(x):
    """
    Print `x` in general decimal format.
    Fixed point for smaller numbers, scientific notation for larger numbers.
    """
    print(f"{x:g}")


def printe(x):
    """
    Print `x` in scientific notation.
    """
    print(f"{x:e}")


def geomean(arr):
    return np.exp(np.mean(np.log(arr)))


def wgeomean(arr, weights):
    return np.exp(np.average(np.log(arr), weights=weights))


def lognorm_from_90_ci(p_5th, p_95th, num_samples):
    p_5th_log = np.log(p_5th)
    p_95th_log = np.log(p_95th)
    # Solve for mu and sigma
    sigma = (p_95th_log - p_5th_log) / (stats.norm.ppf(0.95) - stats.norm.ppf(0.05))
    mu = p_5th_log - stats.norm.ppf(0.05) * sigma
    # Generate lognormal samples
    dist = np.random.lognormal(mean=mu, sigma=sigma, size=num_samples)
    return dist


def lognorm_from_ci(p_low, p_high, ci, num_samples):
    ci_low = 50 - ci / 2
    ci_high = 50 + ci / 2
    p_low_log = np.log(p_low)
    p_high_log = np.log(p_high)
    # Solve for mu and sigma
    sigma = (p_high_log - p_low_log) / (stats.norm.ppf(ci_high/100) - stats.norm.ppf(ci_low/100))
    mu = p_low_log - stats.norm.ppf(ci_low/100) * sigma
    # Generate lognormal samples
    dist = np.random.lognormal(mean=mu, sigma=sigma, size=num_samples)
    return dist


def print_median_and_ci(samples, ci=[5, 95]):
    median_value = np.median(samples)
    lower_percentile, upper_percentile = np.percentile(samples, ci)

    # Formatting and printing the results
    formatted_median = "{:.2g}".format(float(median_value))
    formatted_low = "{:.2g}".format(float(lower_percentile))
    formatted_high = "{:.2g}".format(float(upper_percentile))

    ci_range = int(ci[1] - ci[0])
    print(f"Median: {formatted_median} [{ci_range}% CI: {formatted_low}, {formatted_high}]")
