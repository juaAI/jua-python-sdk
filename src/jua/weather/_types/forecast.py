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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._variables = {
            rename_variable(k): v
            for k, v in kwargs.items()
            if k not in {"requested_latlon", "returned_latlon"}
        }

    @property
    def variables(self) -> Dict[str, List[float]]:
        return self._variables

    def __getitem__(self, key: str) -> List[float] | None:  # Added None to return type
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        if key not in self.variables:
            return None
        return self.variables[key]

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

        variable_keys = list(self.points[0].variables.keys())

        # Extract coordinate information
        returned_lats = [p.returned_latlon.lat for p in self.points]
        returned_lons = [p.returned_latlon.lon for p in self.points]

        lats = np.unique(returned_lats)
        lons = np.unique(returned_lons)

        prediction_timedeltas = [t - self.init_time for t in self.times]

        ds = xr.Dataset(
            coords={
                "time": [self.init_time],
                "prediction_timedelta": prediction_timedeltas,
                "latitude": lats,
                "longitude": lons,
            },
        )

        point_mapping: dict[tuple[float, float], PointResponse] = {}
        for points in self.points:
            point_mapping[(points.returned_latlon.lat, points.returned_latlon.lon)] = (
                points
            )

        # Create data variables for the dataset
        for var_key in variable_keys:
            # Initialize array with explicit missing values (using numpy.nan)
            data_shape = (1, len(prediction_timedeltas), len(lats), len(lons))
            data_array = np.full(data_shape, np.nan)

            # Fill only the coordinates where we actually have data
            for lat_idx, lat in enumerate(lats):
                for lon_idx, lon in enumerate(lons):
                    if (lat, lon) not in point_mapping:
                        continue
                    points = point_mapping[(lat, lon)]
                    var_values = points[var_key]
                    data_array[0, :, lat_idx, lon_idx] = var_values

            # Add the variable to the dataset
            ds[var_key] = (
                ("time", "prediction_timedelta", "latitude", "longitude"),
                data_array,
            )
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
