from collections import defaultdict
from dataclasses import dataclass

from jua.weather.models import Models


@dataclass(frozen=True)
class TemporalResolution:
    """Internal class to store model temporal resolution

    Used for models with variable temporal resolution, such as EPT2.
    Resolutions are expressed in hours and may be fractional (e.g. ``0.5``
    for a 30-minute cadence such as EPT2 Helios).

    Attributes:
        base: The default temporal resolution for the model, in hours.
        special: The resolution of the model for prediction_timedelta ranges.
            Defined as `(resolution, from_hour, to_hour)`, where the model has a
            prediction every `resolution` hours when the prediction_timedelta is
            in the interval [`from_hour`, `to_hour`].
    """

    base: float
    special: tuple[tuple[float, int, int], ...] = tuple()

    def __post_init__(self) -> None:
        """Checks that the special cases make sense"""
        if self.base <= 0:
            raise ValueError(f"base resolution must be > 0, got {self.base}")

        # Validate each special tuple: (resolution, start_hour, end_hour)
        for res, start, end in self.special:
            if res <= 0 or not (0 <= start < end):
                raise ValueError(f"Malformed resolution {res}, [{start}, {end}].")

        if len(self.special) > 1:
            _, prev_start, prev_end = self.special[0]
            for _, start, end in self.special[1:]:
                if start < prev_start or start < prev_end:
                    raise ValueError(
                        "Special intervals must be non-overlapping and in increasing "
                        f"order. Overlap between interval ending at {prev_end} and "
                        f"starting at {start}."
                    )
                prev_start, prev_end = start, end

    def num_prediction_timedeltas(self, from_hour: int, to_hour: int) -> int:
        """Determines the number of `prediction_timedeltas` in an interval.

        Iterates internally in minutes so that sub-hourly resolutions
        (e.g. a 30-minute cadence) are counted correctly.

        Attributes:
            from_hour: The start hour for the interval
            to_hour: The end hour for the interval
        """
        if from_hour < 0 or to_hour < 0 or to_hour < from_hour:
            raise ValueError(
                "from_hour and to_hour must be non-negative and to_hour < from_hour. "
                f"Got from_hour={from_hour}, to_hour={to_hour}"
            )

        num_timedeltas = 0
        for minute in range(from_hour * 60, to_hour * 60 + 1):
            hour = minute / 60
            resolution = self.base
            for s_res, s_start, s_end in self.special:
                if s_start <= hour <= s_end:
                    resolution = s_res
                    break
            resolution_minutes = round(resolution * 60)
            if minute % resolution_minutes == 0:
                num_timedeltas += 1

        return num_timedeltas


@dataclass
class ModelMetaInfo:
    """Internal class to store meta information"""

    forecast_name_mapping: str | None = None
    full_forecasted_hours: int | None = None
    has_forecast_file_access: bool = False
    has_grid_access: bool = False
    has_statistics: bool = False
    num_lats: int = 720
    num_lons: int = 1440
    has_both_poles: bool = False
    forecasts_per_day: int = 4
    temporal_resolution: TemporalResolution = TemporalResolution(6)


_MODEL_META_INFO = defaultdict(ModelMetaInfo)
_MODEL_META_INFO[Models.EPT1_5] = ModelMetaInfo(
    has_grid_access=True,
    forecast_name_mapping="ept-1.5-b",
    full_forecasted_hours=480,
    has_forecast_file_access=True,
    num_lats=2221,
    num_lons=4440,
    has_both_poles=True,
    temporal_resolution=TemporalResolution(base=1),
)
_MODEL_META_INFO[Models.EPT1_5_EARLY] = ModelMetaInfo(
    has_grid_access=True,
    forecast_name_mapping="ept-1.5-early-b",
    full_forecasted_hours=480,
    has_forecast_file_access=True,
    num_lats=2221,
    num_lons=4440,
    has_both_poles=True,
    temporal_resolution=TemporalResolution(base=1),
)
_MODEL_META_INFO[Models.EPT2] = ModelMetaInfo(
    has_grid_access=True,
    forecast_name_mapping="ept-2",
    full_forecasted_hours=480,
    has_forecast_file_access=True,
    num_lats=2160,
    num_lons=4320,
    temporal_resolution=TemporalResolution(base=6, special=((1, 0, 10 * 24),)),
)
_MODEL_META_INFO[Models.EPT2_E] = ModelMetaInfo(
    has_grid_access=True,
    forecast_name_mapping="ept2-e",
    full_forecasted_hours=480,
    has_forecast_file_access=True,
    has_statistics=True,
    temporal_resolution=TemporalResolution(base=6, special=((1, 0, 10 * 24),)),
)
_MODEL_META_INFO[Models.EPT2_EARLY] = ModelMetaInfo(
    has_grid_access=True,
    forecast_name_mapping="ept-2-early",
    full_forecasted_hours=480,
    has_forecast_file_access=True,
    num_lats=1440,
    num_lons=2880,
    temporal_resolution=TemporalResolution(base=6, special=((1, 0, 10 * 24),)),
)
_MODEL_META_INFO[Models.EPT2_HRRR] = ModelMetaInfo(
    has_grid_access=True,
    full_forecasted_hours=48,
    num_lats=3600,
    num_lons=7200,
    forecasts_per_day=24,
    temporal_resolution=TemporalResolution(base=1),
)
_MODEL_META_INFO[Models.EPT2_RR] = ModelMetaInfo(
    has_grid_access=True,
    forecast_name_mapping="ept-2-rr",
    full_forecasted_hours=48,
    has_forecast_file_access=True,
    forecasts_per_day=24,
    temporal_resolution=TemporalResolution(base=1),
)
_MODEL_META_INFO[Models.EPT2_REASONING] = ModelMetaInfo(
    has_grid_access=True,
    forecast_name_mapping="ept-2-reasoning",
    full_forecasted_hours=480,
    temporal_resolution=TemporalResolution(base=6, special=((1, 0, 10 * 24),)),
)
_MODEL_META_INFO[Models.EPT2_HELIOS] = ModelMetaInfo(
    has_grid_access=True,
    full_forecasted_hours=48,
    forecasts_per_day=48,
    temporal_resolution=TemporalResolution(base=0.5),
)
_MODEL_META_INFO[Models.EPT2_EUROPA] = ModelMetaInfo(
    has_grid_access=True,
    full_forecasted_hours=48,
    forecasts_per_day=24,
    temporal_resolution=TemporalResolution(base=1),
)
_MODEL_META_INFO[Models.AIFS] = ModelMetaInfo(
    has_grid_access=True,
    forecast_name_mapping="aifs",
    full_forecasted_hours=480,
    has_forecast_file_access=True,
)
_MODEL_META_INFO[Models.AURORA] = ModelMetaInfo(
    has_grid_access=True,
    forecast_name_mapping="aurora",
    full_forecasted_hours=480,
    has_forecast_file_access=True,
    num_lats=2160,
    num_lons=4320,
)
_MODEL_META_INFO[Models.ECMWF_IFS_SINGLE] = ModelMetaInfo(
    has_grid_access=True,
    full_forecasted_hours=360,
    num_lats=2160,
    num_lons=4320,
    temporal_resolution=TemporalResolution(
        base=6,
        special=(
            (1, 0, 90),
            (3, 90, 144),
        ),
    ),
)
_MODEL_META_INFO[Models.NOAA_GFS_SINGLE] = ModelMetaInfo(
    has_grid_access=True,
    full_forecasted_hours=384,
    num_lats=720,
    num_lons=1440,
    temporal_resolution=TemporalResolution(
        base=6,
        special=(
            (1, 0, 138),
            (3, 138, 384),
        ),
    ),
)
_MODEL_META_INFO[Models.ICON_EU] = ModelMetaInfo(
    has_grid_access=True,
    full_forecasted_hours=120,
    num_lats=2880,
    num_lons=5760,
    forecasts_per_day=8,
    temporal_resolution=TemporalResolution(
        base=3,
        special=((1, 0, 78),),
    ),
)
_MODEL_META_INFO[Models.ICON_GLOBAL] = ModelMetaInfo(
    has_grid_access=True,
    full_forecasted_hours=180,
    num_lats=1440,
    num_lons=2880,
    forecasts_per_day=4,
    temporal_resolution=TemporalResolution(
        base=3,
        special=((1, 0, 78),),
    ),
)
_MODEL_META_INFO[Models.AIFS_ENS] = ModelMetaInfo(
    has_grid_access=True,
    full_forecasted_hours=360,
    has_statistics=True,
    # 0.25 degree grid (1440x720 after south-pole drop) — matches the default
    # num_lats=720 / num_lons=1440, so no override needed.
    temporal_resolution=TemporalResolution(base=6),
)
_MODEL_META_INFO[Models.ECMWF_AIFS_ENSEMBLE] = ModelMetaInfo(
    forecast_name_mapping="ecmwf_aifs025_ensemble",
    full_forecasted_hours=360,
    has_forecast_file_access=False,
    has_statistics=True,
)
_MODEL_META_INFO[Models.ECMWF_IFS_ENSEMBLE] = ModelMetaInfo(
    full_forecasted_hours=360,
    has_forecast_file_access=False,
    has_statistics=True,
    temporal_resolution=TemporalResolution(base=6, special=((3, 0, 144),)),
)


def get_model_meta_info(model: Models) -> ModelMetaInfo:
    return _MODEL_META_INFO[model]
