from datetime import datetime
from typing import Dict, List  # Added for type hinting clarity

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from jua.weather._xarray_patches import TypedDataset, as_typed_dataset
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
        requested_lats = [p.requested_latlon.lat for p in self.points]
        requested_lons = [p.requested_latlon.lon for p in self.points]
        returned_lats = [p.returned_latlon.lat for p in self.points]
        returned_lons = [p.returned_latlon.lon for p in self.points]

        # Create data variables
        data_vars = {}
        for var in variable_keys:
            # For each point, get values across all times
            values = []
            for point in self.points:
                if var in point.variables:
                    # Ensure the length matches self.times, pad with np.nan if necessary
                    point_data = point[var]
                    if point_data is not None and len(point_data) == len(self.times):
                        values.append(point_data)
                    else:
                        values.append([np.nan] * len(self.times))
                else:
                    values.append([np.nan] * len(self.times))

            # Create a variable with dimensions (point, time)
            data_vars[var] = (("point", "time"), np.array(values))

        # Create the dataset with point as primary dimension
        ds = (
            xr.Dataset(
                data_vars=data_vars,
                coords={
                    "time": self.times,
                    "point": np.arange(len(self.points)),
                    "requested_lat": ("point", requested_lats),
                    "requested_lon": ("point", requested_lons),
                    "returned_lat": ("point", returned_lats),
                    "returned_lon": ("point", returned_lons),
                },
                attrs={
                    "model": self.model,
                    "forecast_id": self.id,
                    "name": self.name,
                    "init_time": self.init_time.isoformat(),
                },
            )
            .set_index(point=["requested_lat", "requested_lon"])
            .set_index(point=["returned_lat", "returned_lon"])
        )

        return as_typed_dataset(ds)

    def to_pandas(self) -> pd.DataFrame | None:
        ds = self.to_xarray()
        if ds is None:
            return None
        return ds.to_dataframe()
