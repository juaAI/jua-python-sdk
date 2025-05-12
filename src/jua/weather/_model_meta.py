from dataclasses import dataclass

from jua.weather.models import Models


@dataclass
class Chunks:
    time: int
    prediction_timedelta: int
    latitude: int
    longitude: int

    def to_dict(self) -> dict[str, int]:
        return {
            "time": self.time,
            "prediction_timedelta": self.prediction_timedelta,
            "latitude": self.latitude,
            "longitude": self.longitude,
        }


@dataclass
class ModelMetaInfo:
    forecast_name_mapping: str | None = None
    forecast_zarr_version: int | None = 3
    hindcast_zarr_version: int | None = 3
    forecast_chunks: dict[str, int] | str = "auto"
    hindcast_chunks: dict[str, int] | str = "auto"


_MODEL_META_INFOR = {
    Models.EPT2: ModelMetaInfo(
        forecast_name_mapping="ept-2",
        forecast_zarr_version=3,
        hindcast_zarr_version=3,
    ),
    Models.EPT1_5: ModelMetaInfo(
        forecast_name_mapping="ept-1.5",
        forecast_zarr_version=2,
        hindcast_zarr_version=2,
        # Specified manually since "auto" throws error when calling .to_zarr()
        hindcast_chunks=Chunks(1, 1, 444, 741).to_dict(),
    ),
    Models.EPT1_5_EARLY: ModelMetaInfo(
        forecast_name_mapping="ept-1.5-early",
        forecast_zarr_version=2,
        hindcast_zarr_version=2,
        # Specified manually since "auto" throws error when calling .to_zarr()
        hindcast_chunks=Chunks(1, 1, 444, 741).to_dict(),
    ),
    Models.ECMWF_AIFS025_SINGLE: ModelMetaInfo(
        forecast_name_mapping=None,  # No forecast data available
        forecast_zarr_version=None,  # No forecast data available
        hindcast_zarr_version=3,
    ),
}


def get_model_meta_info(model: Models) -> ModelMetaInfo:
    return _MODEL_META_INFOR[model]
