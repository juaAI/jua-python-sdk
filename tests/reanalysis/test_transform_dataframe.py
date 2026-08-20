"""Tests for the reanalysis DataFrame -> xarray transform.

The input frames here reproduce the exact column sets the live
``POST /v1/reanalysis/data`` returns, verified against dev:

  grid (bounding_box) : time, latitude, longitude, <vars>
  point (nearest)     : time, latitude, longitude, point, <vars>
  point (bilinear)    : time, point, latitude, longitude, <vars>

Note there is **no** ``model`` column, unlike the forecast endpoint.
"""

import numpy as np
import pandas as pd
import pytest

from jua.reanalysis._query_engine import ReanalysisQueryEngine
from jua.types.geo import LatLon

_TEMP = "air_temperature_at_height_level_2m"


def _grid_frame() -> pd.DataFrame:
    """2 timesteps x 2 lats x 2 lons."""
    times = pd.to_datetime(["2024-01-15T00:00:00", "2024-01-15T01:00:00"])
    rows = []
    value = 270.0
    for t in times:
        for lat in (52.5, 52.25):
            for lon in (13.0, 13.25):
                rows.append(
                    {"time": t, "latitude": lat, "longitude": lon, _TEMP: value}
                )
                value += 1.0
    return pd.DataFrame(rows)


def _point_frame() -> pd.DataFrame:
    """2 points x 3 timesteps, with grid-snapped lat/lon like `method=nearest`."""
    times = pd.to_datetime(
        ["2024-01-15T00:00:00", "2024-01-15T01:00:00", "2024-01-15T02:00:00"]
    )
    # point 0 requested (47.3769, 8.5417) -> snapped (47.5, 8.5)
    # point 1 requested (51.5074, -0.1278) -> snapped (51.5, -0.25)
    snapped = {0: (47.5, 8.5), 1: (51.5, -0.25)}
    rows = []
    value = 280.0
    for idx, (lat, lon) in snapped.items():
        for t in times:
            rows.append(
                {
                    "time": t,
                    "latitude": lat,
                    "longitude": lon,
                    "point": idx,
                    _TEMP: value,
                }
            )
            value += 1.0
    return pd.DataFrame(rows)


def test_grid_query_dims_and_coords():
    ds = ReanalysisQueryEngine.transform_dataframe(_grid_frame())

    assert set(ds.sizes) == {"time", "latitude", "longitude"}
    assert ds.sizes["time"] == 2
    assert ds.sizes["latitude"] == 2
    assert ds.sizes["longitude"] == 2
    # No init_time / prediction_timedelta anywhere: this is reanalysis.
    assert "init_time" not in ds.coords
    assert "prediction_timedelta" not in ds.coords
    assert _TEMP in ds.data_vars


def test_grid_query_values_land_on_the_right_cell():
    df = _grid_frame()
    ds = ReanalysisQueryEngine.transform_dataframe(df)

    expected = df.loc[
        (df["time"] == pd.Timestamp("2024-01-15T01:00:00"))
        & (df["latitude"] == 52.25)
        & (df["longitude"] == 13.25),
        _TEMP,
    ].iloc[0]
    actual = ds[_TEMP].sel(time="2024-01-15T01:00:00", latitude=52.25, longitude=13.25)
    assert float(actual) == pytest.approx(expected)


def test_point_query_dims_and_coords():
    points = [LatLon(lat=47.3769, lon=8.5417), LatLon(lat=51.5074, lon=-0.1278)]
    ds = ReanalysisQueryEngine.transform_dataframe(_point_frame(), points=points)

    assert set(ds.sizes) == {"points", "time"}
    assert ds.sizes["points"] == 2
    assert ds.sizes["time"] == 3

    # Both the actual grid cell and the originally requested point are kept.
    for coord in ("latitude", "longitude", "requested_lat", "requested_lon"):
        assert coord in ds.coords
        assert ds[coord].dims == ("points",)


def test_point_query_keeps_requested_and_actual_coordinates_aligned():
    points = [LatLon(lat=47.3769, lon=8.5417), LatLon(lat=51.5074, lon=-0.1278)]
    ds = ReanalysisQueryEngine.transform_dataframe(_point_frame(), points=points)

    for point in points:
        selected = ds.sel(points=str(point))
        assert float(selected["requested_lat"]) == pytest.approx(point.lat)
        assert float(selected["requested_lon"]) == pytest.approx(point.lon)

    zurich = ds.sel(points=str(points[0]))
    assert float(zurich["latitude"]) == pytest.approx(47.5)
    assert float(zurich["longitude"]) == pytest.approx(8.5)


def test_data_vars_are_float32():
    ds = ReanalysisQueryEngine.transform_dataframe(_grid_frame())
    assert ds[_TEMP].dtype == np.float32


def test_time_encoding_is_milliseconds():
    ds = ReanalysisQueryEngine.transform_dataframe(_grid_frame())
    assert ds.time.encoding["dtype"] == "int64"
    assert "milliseconds since" in ds.time.encoding["units"]


def test_grid_query_drops_duplicate_index_entries():
    df = pd.concat([_grid_frame(), _grid_frame()], ignore_index=True)
    ds = ReanalysisQueryEngine.transform_dataframe(df)
    # Duplicates collapse rather than exploding the dims.
    assert ds.sizes["time"] == 2
    assert ds.sizes["latitude"] == 2
    assert ds.sizes["longitude"] == 2


def test_transform_does_not_mutate_caller_frame():
    df = _point_frame()
    before = df.columns.tolist()
    points = [LatLon(lat=47.3769, lon=8.5417), LatLon(lat=51.5074, lon=-0.1278)]
    ReanalysisQueryEngine.transform_dataframe(df, points=points)
    assert df.columns.tolist() == before
