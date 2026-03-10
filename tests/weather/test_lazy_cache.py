import numpy as np
import pytest

from jua.weather._lazy_loading.cache import BBoxCache, ForecastCache, MergedBBox
from jua.weather.models import Models


class _DummyQueryEngine:
    pass


def _make_cache(grid_chunk: int = 2, nlat: int = 8, nlon: int = 8) -> ForecastCache:
    qe = _DummyQueryEngine()
    model = Models.EPT1_5
    variables: list[str] = []
    init_times = [np.datetime64("2024-01-01T00")]
    pred_tds = [np.timedelta64(0, "h")]
    lats = np.arange(nlat, dtype=float)
    lons = np.arange(nlon, dtype=float)
    kwargs: dict = {}
    return ForecastCache(
        query_engine=qe,
        model=model,
        variables=variables,
        init_times=init_times,
        prediction_timedeltas=pred_tds,
        latitudes=lats,
        longitudes=lons,
        original_kwargs=kwargs,
        grid_chunk=grid_chunk,
    )


def _bbox_for(
    cache: ForecastCache, lat_start: int, lat_end: int, lon_start: int, lon_end: int
):
    chunk_lats = cache._latitudes[lat_start:lat_end]
    chunk_lons = cache._longitudes[lon_start:lon_end]
    lat_min = round(chunk_lats[0], 3) - 0.001
    lat_max = round(chunk_lats[-1], 3) + 0.001
    lon_min = round(chunk_lons[0], 3) - 0.001
    lon_max = round(chunk_lons[-1], 3) + 0.001
    return (lat_min, lat_max, lon_min, lon_max)


def test_grouping_collects_chunk_members():
    cache = _make_cache(grid_chunk=2, nlat=8, nlon=8)
    # Two rectangles: a 1x2 on top-left, and a 1x1 at bottom-right
    spatial = {(0, 0), (0, 2), (6, 6)}
    groups = cache._merge_chunks_into_caches(spatial)

    # Expect two groups
    assert len(groups) == 2

    # Flatten chunks for easy comparison
    all_members = sorted((lat, lon) for g in groups for (lat, lon) in g.chunks)
    assert all_members == sorted([(0, 0), (0, 2), (6, 6)])


def test_adjacent_wrapper_matches_group_coords():
    cache = _make_cache(grid_chunk=2, nlat=8, nlon=8)
    spatial = {(0, 0), (0, 2), (2, 0), (2, 2)}  # forms a 2x2 chunks rectangle
    groups = cache._merge_chunks_into_caches(spatial)
    # Should merge into one rectangle covering all 4 chunks
    assert len(groups) == 1
    assert sorted(groups[0].chunks) == sorted([(0, 0), (0, 2), (2, 0), (2, 2)])


def test_single_chunk_merges_to_single_bbox():
    cache = _make_cache(grid_chunk=2, nlat=6, nlon=6)
    spatial = {(0, 0)}  # one chunk at origin (lat 0..2, lon 0..2)
    merged = cache._merge_chunks_into_caches(spatial)
    bboxes = [(g.lat_min, g.lat_max, g.lon_min, g.lon_max) for g in merged]
    expected = [_bbox_for(cache, 0, 2, 0, 2)]
    assert sorted(bboxes) == sorted(expected)


def test_full_2x2_block_merges_to_one_bbox():
    cache = _make_cache(grid_chunk=2, nlat=8, nlon=8)
    spatial = {(0, 0), (0, 2), (2, 0), (2, 2)}  # full 2x2 block of chunks
    merged = cache._merge_chunks_into_caches(spatial)
    bboxes = [(g.lat_min, g.lat_max, g.lon_min, g.lon_max) for g in merged]
    expected = [_bbox_for(cache, 0, 4, 0, 4)]
    assert sorted(bboxes) == sorted(expected)


def test_l_shape_splits_into_two_rectangles():
    cache = _make_cache(grid_chunk=2, nlat=8, nlon=8)
    # L-shape: top row two chunks, plus one below-left; missing bottom-right
    spatial = {(0, 0), (0, 2), (2, 0)}
    merged = cache._merge_chunks_into_caches(spatial)
    bboxes = [(g.lat_min, g.lat_max, g.lon_min, g.lon_max) for g in merged]
    expected = [
        _bbox_for(cache, 0, 2, 0, 4),  # top row two chunks wide
        _bbox_for(cache, 2, 4, 0, 2),  # bottom-left single chunk
    ]
    assert sorted(bboxes) == sorted(expected)


def test_disjoint_chunks_produce_multiple_bboxes():
    cache = _make_cache(grid_chunk=2, nlat=10, nlon=10)
    spatial = {(0, 0), (6, 6)}  # far apart chunks
    merged = cache._merge_chunks_into_caches(spatial)
    bboxes = [(g.lat_min, g.lat_max, g.lon_min, g.lon_max) for g in merged]
    expected = [
        _bbox_for(cache, 0, 2, 0, 2),
        _bbox_for(cache, 6, 8, 6, 8),
    ]
    assert sorted(bboxes) == sorted(expected)


def test_gap_in_row_splits_rectangles():
    cache = _make_cache(grid_chunk=2, nlat=8, nlon=8)
    # Two chunks on same row with a gap between them
    spatial = {(0, 0), (0, 4)}
    merged = cache._merge_chunks_into_caches(spatial)
    bboxes = [(g.lat_min, g.lat_max, g.lon_min, g.lon_max) for g in merged]
    expected = [
        _bbox_for(cache, 0, 2, 0, 2),
        _bbox_for(cache, 0, 2, 4, 6),
    ]
    assert sorted(bboxes) == sorted(expected)


def test_stitch_handles_missing_prediction_timedelta_in_bbox():
    """When a bbox has fewer lead times than global, missing slots should be NaN."""
    qe = _DummyQueryEngine()
    cache = ForecastCache(
        query_engine=qe,
        model=Models.EPT1_5,
        variables=["dummy"],
        init_times=[np.datetime64("2024-01-01T00")],
        prediction_timedeltas=[
            np.timedelta64(0, "h"),
            np.timedelta64(1, "h"),
            np.timedelta64(2, "h"),
        ],
        latitudes=np.array([0.0, 1.0]),
        longitudes=np.array([0.0, 1.0]),
        original_kwargs={},
        grid_chunk=2,
    )

    bbox = MergedBBox(
        lat_min=-0.001,
        lat_max=1.001,
        lon_min=-0.001,
        lon_max=1.001,
        lat_idx_start=0,
        lat_idx_end=2,
        lon_idx_start=0,
        lon_idx_end=2,
        chunks=[(0, 0)],
    )
    # Local bbox data has lead times [0h, 2h], missing 1h.
    bbox_cache = BBoxCache(
        init_idx=0,
        bbox=bbox,
        pred_td_global_to_local=np.array([0, -1, 1], dtype=int),
        variables={
            "dummy": np.array(
                [
                    [[10.0, 30.0], [11.0, 31.0]],
                    [[12.0, 32.0], [13.0, 33.0]],
                ],
                dtype=np.float32,
            )
        },
    )

    cache._bbox_cache[bbox_cache.id()] = bbox_cache
    cache._chunk_to_bbox[(0, 0, 0)] = bbox_cache.id()

    out = cache.get_variable(
        "dummy",
        (
            0,
            slice(None),
            np.array([0]),
            np.array([0]),
        ),
    )
    assert out.shape == (1, 3, 1, 1)
    assert out[0, 0, 0, 0] == 10.0
    assert np.isnan(out[0, 1, 0, 0])
    assert out[0, 2, 0, 0] == 30.0


def test_positional_index_rejects_scalar_out_of_bounds():
    cache = _make_cache()
    with pytest.raises(IndexError):
        cache._positional_to_indices(48, 48)


def test_positional_index_rejects_array_out_of_bounds():
    cache = _make_cache()
    with pytest.raises(IndexError):
        cache._positional_to_indices(np.array([0, 47, 48]), 48)
