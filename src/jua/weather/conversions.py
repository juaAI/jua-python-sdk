from datetime import datetime

import numpy as np
import pandas as pd


def validate_init_time(init_time: datetime | str) -> datetime:
    as_datetime = (
        init_time
        if isinstance(init_time, datetime)
        else init_time_str_to_datetime(init_time)
    )
    return as_datetime


def datetime_to_init_time_str(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H")


def init_time_str_to_datetime(init_time_str: str) -> datetime:
    return datetime.strptime(init_time_str, "%Y%m%d%H")


def bytes_to_gb(bytes: int) -> float:
    return bytes / 1024 / 1024 / 1024


def to_timedelta(
    hours: int | np.timedelta64 | list[int] | list[np.timedelta64] | None,
) -> np.timedelta64 | list[np.timedelta64] | None:
    """
    Convert various timedelta representations to a NumPy timedelta64 object.

    Args:
        hours: The number of hours to convert to a timedelta

    Returns:
        np.timedelta64: A NumPy timedelta64 object representing the input hours
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
    if isinstance(dt, datetime):
        return dt
    if isinstance(dt, str):
        return pd.to_datetime(dt).to_pydatetime()
    raise ValueError(f"unexpected datetime type: {dt}")


def timedelta_to_hours(td: np.timedelta64 | int) -> int:
    if isinstance(td, np.timedelta64):
        return int(td.astype("timedelta64[ns]") / np.timedelta64(1, "h"))
    if isinstance(td, int):
        return td
    raise ValueError(f"unexpected timedelta type: {td}")
