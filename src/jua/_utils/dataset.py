import warnings
from datetime import datetime

import xarray as xr

from jua._utils.optional_progress_bar import OptionalProgressBar
from jua._utils.remove_none_from_dict import remove_none_from_dict
from jua._utils.spinner import Spinner
from jua.client import JuaClient
from jua.logging import get_logger
from jua.types.geo import LatLon, PredictionTimeDelta, SpatialSelection
from jua.weather.conversions import bytes_to_gb
from jua.weather.variables import Variables, rename_variables

logger = get_logger(__name__)


def _open_dataset(
    client: JuaClient,
    url: str,
    chunks: int | dict[str, int] | str = "auto",
    variables: list[Variables] | list[str] | None = None,
    time: datetime | None = None,
    prediction_timedelta: PredictionTimeDelta = None,
    latitude: SpatialSelection | None = None,
    longitude: SpatialSelection | None = None,
    points: list[LatLon] | LatLon | None = None,
    method: str | None = None,
    **kwargs,
) -> xr.Dataset:
    if points is not None and (latitude is not None or longitude is not None):
        raise ValueError(
            "Cannot provide both points and latitude/longitude. "
            "Please provide either points or latitude/longitude."
        )

    if "engine" not in kwargs:
        kwargs["engine"] = "zarr"

    if "decode_timedelta" not in kwargs:
        kwargs["decode_timedelta"] = True

    storage_options = kwargs.get("storage_options", {})
    if "auth" not in storage_options:
        storage_options["auth"] = client.settings.auth.get_basic_auth()
        kwargs["storage_options"] = storage_options

    kwargs["chunks"] = chunks

    sel_kwargs = {
        "time": time,
        "prediction_timedelta": prediction_timedelta,
        "latitude": latitude,
        "longitude": longitude,
        "points": points,
    }

    sel_kwargs = remove_none_from_dict(sel_kwargs)

    non_slice_kwargs = {k: v for k, v in sel_kwargs.items() if not isinstance(v, slice)}

    slice_kwargs = {k: v for k, v in sel_kwargs.items() if isinstance(v, slice)}

    # Suppress UserWarning about chunks
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*chunks separate the stored chunks.*"
        )
        dataset = xr.open_dataset(url, **kwargs)
    # We cannot call nearest on slices
    dataset = dataset.sel(**non_slice_kwargs, method=method).sel(**slice_kwargs)
    dataset = rename_variables(dataset)

    if variables is not None:
        dataset = dataset[[str(v) for v in variables]]

    return dataset


def _all_dims_nonzero(ds: xr.Dataset) -> bool:
    return all(ds.sizes.values())


def _open_mfdataset_custom(client: JuaClient, urls: list[str], **kwargs) -> xr.Dataset:
    # Combining time with nearest doesn't work
    time_to_select = None
    if "time" in kwargs and kwargs["time"] is not None:
        if kwargs.get("method", None) == "nearest" and not isinstance(
            kwargs["time"], slice
        ):
            time_to_select = kwargs["time"]
            del kwargs["time"]

    datasets = [_open_dataset(client, url, **kwargs) for url in urls]

    # Filter empty datasets
    datasets = [ds for ds in datasets if _all_dims_nonzero(ds)]
    combined = xr.combine_by_coords(datasets)

    if time_to_select is not None:
        combined = combined.sel(time=time_to_select, method="nearest")

    return combined


def open_dataset(
    client: JuaClient,
    urls: str | list[str],
    chunks: int | dict[str, int] | str = "auto",
    should_print_progress: bool | None = None,
    variables: list[Variables] | list[str] | None = None,
    time: datetime | None = None,
    prediction_timedelta: PredictionTimeDelta = None,
    latitude: SpatialSelection | None = None,
    longitude: SpatialSelection | None = None,
    points: list[LatLon] | LatLon | None = None,
    method: str | None = None,
    size_warning_threshold_gb: float = 1,
    compute: bool = True,
    **kwargs,
) -> xr.Dataset:
    if isinstance(urls, str):
        urls = [urls]

    if len(urls) == 0:
        raise ValueError("No URLs provided")

    if "engine" not in kwargs:
        kwargs["engine"] = "zarr"

    if "decode_timedelta" not in kwargs:
        kwargs["decode_timedelta"] = True

    storage_options = kwargs.get("storage_options", {})
    if "auth" not in storage_options:
        storage_options["auth"] = client.settings.auth.get_basic_auth()
        kwargs["storage_options"] = storage_options

    kwargs["chunks"] = chunks

    kwargs = {
        **kwargs,
        **{
            "variables": variables,
            "time": time,
            "prediction_timedelta": prediction_timedelta,
            "latitude": latitude,
            "longitude": longitude,
            "points": points,
            "method": method,
        },
    }

    with Spinner(
        "Preparing dataset...",
        enabled=client.settings.should_print_progress(should_print_progress),
    ):
        if len(urls) == 1:
            dataset = _open_dataset(client, urls[0], **kwargs)
        else:
            dataset = _open_mfdataset_custom(client, urls, **kwargs)

    logger.info("Opening dataset...")
    with OptionalProgressBar(client.settings, should_print_progress):
        if compute:
            dataset_size_gb = bytes_to_gb(dataset.nbytes)
            if dataset_size_gb > size_warning_threshold_gb:
                logger.warning(
                    f"Dataset is large! Size: {dataset_size_gb:.2f}GB.\n"
                    "Opening might take some time."
                )
            else:
                logger.info(f"Loading dataset of size {dataset_size_gb:.2f}GB")
            return dataset.compute()
        else:
            return dataset
