"""Interface with the Query Engine"""

from datetime import datetime
from typing import Literal

import pandas as pd
import xarray as xr
from pydantic import validate_call

from jua._api import QueryEngineAPI
from jua._utils.remove_none_from_dict import remove_none_from_dict
from jua.client import JuaClient
from jua.types.geo import LatLon, PredictionTimeDelta, SpatialSelection
from jua.weather._stream import process_arrow_streaming_response
from jua.weather._types.forecast import ForecastData
from jua.weather._types.query_payload_types import (
    ForecastQueryPayload,
    build_geo_filter,
    build_init_time_arg,
    build_prediction_timedelta,
)
from jua.weather.models import Models


class QueryEngine:
    """Internal API client for Jua's weather services.

    Note:
        This class is intended for internal use only and should not be used directly.
        End users should interact with the higher-level classes.
    """

    _FORECAST_ENDPOINT = "forecast/data"

    def __init__(self, jua_client: JuaClient):
        """Initialize the weather API client.

        Args:
            jua_client: JuaClient instance for authentication and settings.
        """
        self._api = QueryEngineAPI(jua_client)
        self._jua_client = jua_client

        # (30x30 HighRes grid), 1 month, 4 forecasts per day, 49 hours of forecast
        self._MAX_POINTS_PER_REQUEST = (361 * 361) * 31 * 4 * (2 * 24 + 1)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def get_forecast(
        self,
        model: Models,
        init_time: Literal["latest"] | datetime | list[datetime] | slice | None = None,
        variables: list[str] | None = None,
        prediction_timedelta: PredictionTimeDelta | None = None,
        latitude: SpatialSelection | None = None,
        longitude: SpatialSelection | None = None,
        points: list[LatLon] | LatLon | None = None,
        method: Literal["nearest", "bilinear"] = "nearest",
        stream: bool = True,
        print_progress: bool | None = None,
    ) -> ForecastData:
        """Get a forecast for a specific model and initialization time.

        Args:
            model_name: The name of the model for which to get the forecast.

            init_time: Filter by forecast initialization time. Can be:
                - None: All available initialization times (default)
                - A single datetime: Specific initialization time
                - A list of datetimes: Multiple specific times
                - A slice(start, end): Range of initialization times

            variables: List of weather variables to include.

            prediction_timedelta: Filter by forecast lead time. Can be:
                - None: All available lead times (default)
                - A single value (hours or timedelta): Specific lead time
                - A slice(start, stop): Range of lead times
                - A slice(start, stop, step): Lead times at regular intervals

            latitude: Latitude selection. Can be a single value, list of values, or
                a slice(min_lat, max_lat) for a geographical range.

            longitude: Longitude selection. Can be a single value, list of values, or
                a slice(min_lon, max_lon) for a geographical range.

            points: Specific geographic points to get forecasts for. Can be a single
                LatLon object or a list of LatLon objects.

            method: Interpolation method for selecting points:
                - "nearest": Use nearest grid point (default)
                - "bilinear": Bilinear interpolation to the selected point

            stream: Whether to stream the response content. Recommended.

            print_progress: Whether to display a progress bar during data loading.
                If None, uses the client's default setting. Only works when stream=True.

        Returns:
            Forecast data.

        Raises:
            ValueError: If the location parameters are invalid.
        """
        payload = ForecastQueryPayload(
            models=[model],
            init_time=build_init_time_arg(init_time),
            geo=build_geo_filter(latitude, longitude, points, method),
            prediction_timedelta=build_prediction_timedelta(prediction_timedelta),
            variables=variables,
        )

        est_requested_points = payload.num_requested_points()
        if est_requested_points > self._MAX_POINTS_PER_REQUEST:
            raise ValueError(
                "The requested data volume is too large for a single call. "
                f"Estimated size is {est_requested_points} points, which exceeds the "
                f"limit of {self._MAX_POINTS_PER_REQUEST}. The total rows equal "
                "number_of_points × number_of_lead_times × number_of_init_times. "
                "Please split your request into smaller chunks (e.g., fewer points, a "
                "smaller init_time range, or fewer lead times)."
            )

        if print_progress is None:
            print_progress = self._jua_client.settings.print_progress

        data = remove_none_from_dict(payload.model_dump())
        if model == Models.EPT2_E:
            data["aggregation"] = ["avg", "min", "max"]
            data["models"] = ["ept2_e"]

        response = self._api.post(
            self._FORECAST_ENDPOINT,
            data=data,
            query_params={"format": "arrow", "stream": str(stream).lower()},
            extra_headers={"Accept": "*/*", "Accept-Encoding": "identity"},
            stream=stream,
        )
        df = process_arrow_streaming_response(response, print_progress)
        if df.empty:
            raise ValueError("No data available for the given parameters.")

        return self._transform_dataframe(df)

    def _transform_dataframe(self, df: pd.DataFrame) -> xr.Dataset:
        """Transform a raw DataFrame into an xr.Dataset

        Args:
            df: DataFrame to convert. Must have `init_time`, `model`,
            `prediction_timedelta`, `latitude` and `longitude` columns.

        Returns:
            An xarray dataset with "init_time", "prediction_timedelta", "latitude",
            "longitude" dimesions.
        """
        # Parse times to correct units, enforce correct encoding
        df["init_time"] = df["init_time"].astype("datetime64[ns]")
        df["prediction_timedelta"] = df["prediction_timedelta"].astype(
            "timedelta64[ns]"
        )

        # Remove unused metadata columns
        num_models = len(df["model"].unique())
        if not num_models == 1:
            raise ValueError(f"Unexpected number of models returned: {num_models}")
        cols_to_drop = ["model"]
        if "time" in df.columns:
            cols_to_drop.append("time")
        df.drop(columns=cols_to_drop, inplace=True)

        # Set the correct index
        df.set_index(
            ["init_time", "prediction_timedelta", "latitude", "longitude"],
            inplace=True,
        )

        # Remove duplicates, if there are any (remove once duplicates are handeled)
        df = df.loc[~df.index.duplicated()]

        # Convert the DataFrame to a Dataset
        ds = xr.Dataset.from_dataframe(df)

        # Set the correct init_time encoding
        ds.init_time.encoding = {
            "dtype": "int64",
            "units": "nanoseconds since 1970-01-01T00:00:00",
        }
        return ds
