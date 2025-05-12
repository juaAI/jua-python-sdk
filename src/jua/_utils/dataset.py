from datetime import datetime

import xarray as xr

from jua._utils.optional_progress_bar import OptionalProgressBar
from jua._utils.remove_none_from_dict import remove_none_from_dict
from jua.client import JuaClient
from jua.logging import get_logger
from jua.types.geo import PredictionTimeDelta, SpatialSelection
from jua.weather._jua_dataset import rename_variables
from jua.weather.variables import Variables

logger = get_logger(__name__)


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
    method: str | None = None,
    **kwargs,
) -> xr.Dataset:
    """Create an xarray Dataset from one or more URLs.

    Opens datasets stored at the provided URLs using xarray's open_dataset or
    open_mfdataset functions, handling authentication and progress tracking.
    This is a utility function that configures common settings like authentication,
    chunking, and time decoding.

    Args:
        client: A JuaClient instance used for authentication and settings.
        urls: A single URL string or a list of URL strings pointing to dataset
            locations.
        chunks: Chunk sizes for dask array. Can be "auto" for automatic chunking,
            an integer for uniform chunk size, or a dictionary mapping dimension
            names to chunk sizes. Defaults to "auto".
        should_print_progress: Whether to display a progress bar during loading.
            If None, uses the client's default setting.
        time: The initial time of the dataset.
        prediction_timedelta: The prediction time delta of the dataset.
        lat: The latitude of the dataset.
        lon: The longitude of the dataset.
        method: 'nearest' or None
        **kwargs: Additional keyword arguments passed to xr.open_dataset() or
            xr.open_mfdataset(). Common options include:
            - engine: The engine to use for opening the dataset. Defaults to "zarr".
            - decode_timedelta: Whether to decode time delta data. Defaults to True.
            - storage_options: Dict of parameters for the storage backend.

    Returns:
        An xarray Dataset containing the loaded data.

    Raises:
        ValueError: If no URLs are provided.

    Examples:
        >>> ds = open_dataset(client, "https://data.jua.sh/forecasts/ept-2/2025042406.zarr/")
        >>> ds = open_dataset(client, [
            "https://data.jua.sh/forecasts/ept-2/2025042406.zarr/",
            "https://data.jua.sh/forecasts/ept-2/2025042407.zarr/",
        ])
    """
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

    logger.info("Opening dataset...")
    with OptionalProgressBar(client.settings, should_print_progress):
        if len(urls) == 1:
            dataset = xr.open_dataset(urls[0], **kwargs)
        else:
            if "parallel" not in kwargs:
                kwargs["parallel"] = True
            dataset = xr.open_mfdataset(urls, **kwargs)

        sel_kwargs = {
            "time": time,
            "prediction_timedelta": prediction_timedelta,
            "latitude": latitude,
            "longitude": longitude,
        }

        sel_kwargs = remove_none_from_dict(sel_kwargs)

        non_slice_kwargs = {
            k: v for k, v in sel_kwargs.items() if not isinstance(v, slice)
        }

        slice_kwargs = {k: v for k, v in sel_kwargs.items() if isinstance(v, slice)}

        # We cannot call nearest on slices
        dataset = dataset.sel(**non_slice_kwargs, method=method).sel(**slice_kwargs)
        dataset = rename_variables(dataset)

        if variables is not None:
            dataset = dataset[[str(v) for v in variables]]

        return dataset.compute()
