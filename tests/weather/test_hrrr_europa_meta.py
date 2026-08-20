"""Regression tests for EPT2 HRRR / Europa ensemble metadata.

Both regional models expose ensemble statistics through the forecast API, so
``has_statistics`` must stay enabled — otherwise ``get_forecasts(..., statistics=...)``
is rejected client-side before the request is sent.
"""

from jua.weather._model_meta import get_model_meta_info
from jua.weather.models import Models


def test_ept2_hrrr_has_ensemble_statistics() -> None:
    meta = get_model_meta_info(Models.EPT2_HRRR)
    assert meta.has_grid_access is True
    assert meta.has_statistics is True


def test_ept2_europa_has_ensemble_statistics() -> None:
    meta = get_model_meta_info(Models.EPT2_EUROPA)
    assert meta.has_grid_access is True
    assert meta.has_statistics is True
