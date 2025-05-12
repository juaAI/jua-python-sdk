import xarray as xr

from jua._utils.optional_progress_bar import OptionalProgressBar
from jua.client import JuaClient


def open_dataset(
    client: JuaClient,
    urls: str | list[str],
    chunks: int | dict[str, int] | str = "auto",
    should_print_progress: bool | None = None,
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

    with OptionalProgressBar(client.settings, should_print_progress):
        if len(urls) == 1:
            return xr.open_dataset(urls[0], **kwargs)
        else:
            if "parallel" not in kwargs:
                kwargs["parallel"] = True
            return xr.open_mfdataset(urls, **kwargs)
