import pytest

from jua.types.geo import LatLon


def test_lat_lon_not_in_range():
    with pytest.raises(ValueError):
        LatLon(lat=91, lon=0)
    with pytest.raises(ValueError):
        LatLon(lat=-91, lon=0)
    with pytest.raises(ValueError):
        LatLon(lat=0, lon=181)


def test_label_to_key():
    assert LatLon(lat=0, lon=0, label="test").key == "test"
    assert LatLon(lat=0, lon=0, label="test label").key == "test_label"
