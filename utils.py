def datetime_to_float(datetimes):
    date_floats = datetimes.dt.year + (datetimes.dt.month-1) / 12 + (datetimes.dt.day-1) / 365
    return date_floats
