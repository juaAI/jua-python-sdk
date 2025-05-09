from datetime import datetime


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
