"""Regression tests for the hosted AIFS ENS model metadata.

The hosted ClickHouse ``aifs_ens`` model replaced the retired open-meteo
``ecmwf_aifs025_ensemble``. Unlike that point-only open-meteo model, the hosted
model exposes full grid access plus ensemble statistics, so it must not regress
into the "point queries only" behaviour gated by ``has_grid_access``.
"""

from jua.weather._model_meta import get_model_meta_info
from jua.weather.models import Models


def test_aifs_ens_is_fully_accessible() -> None:
    meta = get_model_meta_info(Models.AIFS_ENS)
    # Hosted ClickHouse grid model: grid/bbox slices must be allowed, not just points.
    assert meta.has_grid_access is True
    # 51-member ensemble: statistics available.
    assert meta.has_statistics is True
    # 0.25 degree global grid (1440x720 after south-pole drop).
    assert (meta.num_lats, meta.num_lons) == (720, 1440)
    # 0-360h forecast horizon.
    assert meta.full_forecasted_hours == 360


def test_retired_open_meteo_aifs_ensemble_is_removed() -> None:
    # The open-meteo ecmwf_aifs025_ensemble was retired; only the hosted
    # aifs_ens remains. Guard against the enum member being reintroduced.
    assert not hasattr(Models, "ECMWF_AIFS_ENSEMBLE")
    assert "ecmwf_aifs025_ensemble" not in {m.value for m in Models}
