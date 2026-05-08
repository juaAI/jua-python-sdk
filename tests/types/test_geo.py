import pytest

from jua.types.geo import LatLon, validate_unique_point_keys


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


class TestValidateUniquePointKeys:
    def test_unique_labels_pass(self):
        points = [
            LatLon(lat=55.06, lon=13.00, label="Kriegers Flak"),
            LatLon(lat=55.00, lon=14.00, label="Bornholm"),
        ]
        validate_unique_point_keys(points)

    def test_unique_coords_without_labels_pass(self):
        points = [
            LatLon(lat=50.0, lon=8.0),
            LatLon(lat=51.0, lon=9.0),
        ]
        validate_unique_point_keys(points)

    def test_single_point_passes(self):
        validate_unique_point_keys([LatLon(lat=0, lon=0, label="solo")])

    def test_empty_list_passes(self):
        validate_unique_point_keys([])

    def test_duplicate_labels_rejected(self):
        points = [
            LatLon(lat=55.06, lon=13.00, label="Kriegers"),
            LatLon(lat=55.00, lon=14.00, label="Kriegers"),
        ]
        with pytest.raises(ValueError, match="points\\[0\\] and points\\[1\\]"):
            validate_unique_point_keys(points)

    def test_duplicate_labels_case_insensitive(self):
        points = [
            LatLon(lat=1.0, lon=2.0, label="Berlin"),
            LatLon(lat=3.0, lon=4.0, label="berlin"),
        ]
        with pytest.raises(ValueError, match="'berlin'"):
            validate_unique_point_keys(points)

    def test_duplicate_coords_without_labels_rejected(self):
        points = [
            LatLon(lat=50.0, lon=8.0),
            LatLon(lat=50.0, lon=8.0),
        ]
        with pytest.raises(ValueError, match="point_50.0_8.0"):
            validate_unique_point_keys(points)

    def test_third_duplicate_reports_first_occurrence(self):
        points = [
            LatLon(lat=1.0, lon=1.0, label="A"),
            LatLon(lat=2.0, lon=2.0, label="B"),
            LatLon(lat=3.0, lon=3.0, label="A"),
        ]
        with pytest.raises(ValueError, match="points\\[0\\] and points\\[2\\]"):
            validate_unique_point_keys(points)
