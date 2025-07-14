from datetime import datetime
from typing import Dict, List  # Added for type hinting clarity

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from jua.weather._xarray_patches import TypedDataset
from jua.weather.variables import rename_variable


@dataclass
class Point:
    lat: float
    lon: float


class PointResponse(BaseModel, extra="allow"):
    requested_latlon: Point
    returned_latlon: Point
    _variables: Dict[str, List[float]]  # Added type hint and initialization
    _statistics: (
        Dict[str, Dict[str, List[float]]] | None
    )  # Added type hint and initialization

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._variables = {
            rename_variable(k): v
            for k, v in kwargs.items()
            if k not in {"requested_latlon", "returned_latlon", "stats"}
        }
        self._statistics = None
        if "stats" in kwargs:
            self._statistics = {
                rename_variable(k): v for k, v in kwargs["stats"].items()
            }

        var_names = {k for k in self._variables.keys()}
        if self._statistics is not None:
            var_names.update({k for k in self._statistics.keys()})
        self._variable_names = list(sorted(var_names))

        self._statistic_names = []
        if self._statistics is not None:
            self._statistic_names = list(
                self._statistics[self._variable_names[0]].keys()
            )
            if len(self._variables) > 0:  # mean available
                self._statistic_names = ["mean"] + self._statistic_names

    @property
    def variables(self) -> Dict[str, List[float]]:
        return self._variables

    @property
    def statistics(self) -> Dict[str, Dict[str, List[float]]] | None:
        return self._statistics

    @property
    def variable_names(self) -> List[str]:
        return self._variable_names

    @property
    def statistic_names(self) -> List[str]:
        return self._statistic_names

    def __getitem__(self, key: str) -> List[float] | None:  # Added None to return type
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        if key not in self.variables:
            return None
        return self.variables[key]

    def get_statistics(self, key: str) -> List[List[float]] | None:
        """
        Returns:
            List[List[float]] | None: A list of lists of statistics.
                If no statistics are available, returns None.
                If the statistics are available, returns the statistics for the given
                key in the order of statistic_names.
        """
        if not isinstance(key, str):
            raise TypeError("Key must be a string")

        if self.statistics is None:
            return None

        stats = []
        stat_names = self._statistic_names
        if self._statistic_names[0] == "mean":
            stat_names = stat_names[1:]
            stats.append(self.variables[key])

        for stat in stat_names:
            stats.append(self.statistics[key][stat])

        return stats

    def __repr__(self):
        variables = "\n".join([f"{k}: {v}" for k, v in self.variables.items()])
        return (
            f"PointResponse(\nrequested_latlon={self.requested_latlon}\n"
            f"returned_latlon={self.returned_latlon}\n"
            f"{variables}\n)"
        )


@dataclass
class ForecastData:
    model: str
    id: str
    name: str
    init_time: datetime
    max_available_lead_time: int
    times: List[datetime]
    points: List[PointResponse]

    def to_xarray(self) -> TypedDataset | None:
        if len(self.points) == 0:
            return None

        variable_names = self.points[0].variable_names

        # Extract coordinate information
        returned_lats = [p.returned_latlon.lat for p in self.points]
        returned_lons = [p.returned_latlon.lon for p in self.points]

        lats = np.unique(returned_lats)
        lons = np.unique(returned_lons)

        prediction_timedeltas = [t - self.init_time for t in self.times]
        coords = {
            "time": [self.init_time],
            "prediction_timedelta": prediction_timedeltas,
            "latitude": lats,
            "longitude": lons,
        }
        dims: tuple[str, ...] = (
            "time",
            "prediction_timedelta",
            "latitude",
            "longitude",
        )
        if self.points[0].statistics is not None:
            dims = ("time", "stat", "prediction_timedelta", "latitude", "longitude")
            coords["stat"] = self.points[0].statistic_names

        ds = xr.Dataset(coords=coords)

        point_mapping: dict[tuple[float, float], PointResponse] = {}
        for points in self.points:
            point_mapping[(points.returned_latlon.lat, points.returned_latlon.lon)] = (
                points
            )

        # Create data variables for the dataset
        for var_key in variable_names:
            # Initialize array with explicit missing values (using numpy.nan)
            data_shape: tuple[int, ...] = (
                1,
                len(prediction_timedeltas),
                len(lats),
                len(lons),
            )
            if self.points[0].statistics is not None:
                num_stats = len(coords["stat"])
                data_shape = (
                    1,
                    num_stats,
                    len(prediction_timedeltas),
                    len(lats),
                    len(lons),
                )
            data_array = np.full(data_shape, np.nan)

            # Fill only the coordinates where we actually have data
            for lat_idx, lat in enumerate(lats):
                for lon_idx, lon in enumerate(lons):
                    if (lat, lon) not in point_mapping:
                        continue

                    point = point_mapping[(lat, lon)]
                    if point.statistics is not None:
                        data_array[0, :, :, lat_idx, lon_idx] = point.get_statistics(
                            var_key
                        )
                    else:
                        point_data = point[var_key]
                        if point_data is None or len(point_data) != len(
                            prediction_timedeltas
                        ):
                            num_values = None if point_data is None else len(point_data)
                            raise ValueError(
                                f"Forecast data for variable {var_key} at ({lat}, "
                                f"{lon}) has {num_values} values, but "
                                f"{len(prediction_timedeltas)} are expected."
                            )

                        data_array[0, :, lat_idx, lon_idx] = point_data

            # Add the variable to the dataset
            ds[var_key] = (dims, data_array)

        ds.attrs["model"] = self.model
        ds.attrs["id"] = self.id
        ds.attrs["name"] = self.name
        ds.attrs["init_time"] = str(self.init_time)
        ds.attrs["max_available_lead_time"] = self.max_available_lead_time
        return ds

    def to_pandas(self) -> pd.DataFrame | None:
        ds = self.to_xarray()
        if ds is None:
            return None
        return ds.to_dataframe()
