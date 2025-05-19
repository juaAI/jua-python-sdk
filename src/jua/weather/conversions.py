from datetime import datetime

import numpy as np
import pandas as pd


def validate_init_time(init_time: datetime | str) -> datetime:
    """Validate and convert initialization time to datetime.

    Args:
        init_time: Initialization time as datetime or string

    Returns:
        Validated datetime object
    """
    as_datetime = (
        init_time
        if isinstance(init_time, datetime)
        else init_time_str_to_datetime(init_time)
    )
    return as_datetime


def datetime_to_init_time_str(dt: datetime) -> str:
    """Convert datetime to initialization time string format.

    Formats datetime as YYYYMMDDHH (e.g., 2025052806 for May 28, 2025, 6:00 AM).
    This is the format used for init times of the weather data.

    Args:
        dt: Datetime to convert

    Returns:
        Formatted initialization time string
    """
    return dt.strftime("%Y%m%d%H")


def init_time_str_to_datetime(init_time_str: str) -> datetime:
    """Convert initialization time string to datetime.

    Parses strings in YYYYMMDDHH format (e.g., 2025052806).
    This is the format used for init times of the weather data.

    Args:
        init_time_str: String in YYYYMMDDHH format

    Returns:
        Parsed datetime object
    """
    return datetime.strptime(init_time_str, "%Y%m%d%H")


def bytes_to_gb(bytes: int) -> float:
    """Convert bytes to gigabytes.

    Args:
        bytes: Size in bytes

    Returns:
        Size in gigabytes
    """
    return bytes / (1024**3)


def to_timedelta(
    hours: int | np.timedelta64 | list[int] | list[np.timedelta64] | None,
) -> np.timedelta64 | list[np.timedelta64] | None:
    """Convert hours or existing timedeltas to numpy timedelta64 objects.

    Handles various input types including integers (interpreted as hours),
    existing timedelta64 objects, lists of either, or None.

    Args:
        hours: Hours value(s) or timedelta object(s) to convert

    Returns:
        Converted timedelta64 object(s) or None

    Raises:
        ValueError: If input is an unsupported type
    """
    if isinstance(hours, list):
        return [to_timedelta(h) for h in hours]
    if isinstance(hours, int):
        return np.timedelta64(hours, "h").astype("timedelta64[ns]")
    if isinstance(hours, np.timedelta64):
        return hours
    if hours is None:
        return None
    raise ValueError(f"unexpected timedelta type: {hours}")


def to_datetime(dt: datetime | str) -> datetime:
    """Convert various datetime representations to Python datetime objects.

    Args:
        dt: Datetime as Python datetime object or string

    Returns:
        Python datetime object

    Raises:
        ValueError: If input is an unsupported type
    """
    if isinstance(dt, datetime):
        return dt
    if isinstance(dt, str):
        return pd.to_datetime(dt).to_pydatetime()
    raise ValueError(f"unexpected datetime type: {dt}")


def timedelta_to_hours(td: np.timedelta64 | int) -> int:
    """Convert timedelta to integer hours.

    Args:
        td: Timedelta object or integer hours

    Returns:
        Number of hours as integer

    Raises:
        ValueError: If input is an unsupported type
    """
    if isinstance(td, np.timedelta64):
        return int(td.astype("timedelta64[ns]") / np.timedelta64(1, "h"))
    if isinstance(td, int):
        return td
    raise ValueError(f"unexpected timedelta type: {td}")
