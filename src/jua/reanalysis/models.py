from enum import Enum


class ReanalysisModels(str, Enum):
    """Reanalysis datasets available through the Jua API.

    Reanalysis data is a reconstruction of past weather, so it has a single
    `time` dimension rather than the `init_time` + `prediction_timedelta` pair
    that forecast models use. That is why these are a separate enum from
    :class:`jua.weather.models.Models` rather than entries in it.

    Inherits from `str`, like :class:`jua.weather.models.Models`, so pydantic
    serialises members to their API name rather than leaving an enum object in
    the request body.

    Examples:
        >>> from jua import JuaClient
        >>> from jua.reanalysis import ReanalysisModels
        >>> client = JuaClient()
        >>> client.reanalysis.get_variables(model=ReanalysisModels.ERA5)
    """

    ERA5 = "arco_era5"
    """ECMWF ERA5 reanalysis on a 0.25 degree global grid, hourly.

    Served from Google's Analysis-Ready Cloud-Optimized (ARCO) ERA5 archive,
    which is why the API name is ``arco_era5``.
    """
