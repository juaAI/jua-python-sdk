from dataclasses import dataclass

from jua.weather.models import Model


@dataclass
class ModelMetaInfo:
    forecast_name_mapping: str | None = None
    forecast_zarr_version: int | None = 3
    hindcast_zarr_version: int | None = 3


_MODEL_META_INFOR = {
    Model.EPT2: ModelMetaInfo(
        forecast_name_mapping="ept-2",
        forecast_zarr_version=3,
        hindcast_zarr_version=3,
    ),
    Model.EPT1_5: ModelMetaInfo(
        forecast_name_mapping="ept-1.5",
        forecast_zarr_version=2,
        hindcast_zarr_version=2,
    ),
    Model.EPT1_5_EARLY: ModelMetaInfo(
        forecast_name_mapping="ept-1.5-early",
        forecast_zarr_version=2,
        hindcast_zarr_version=2,
    ),
    Model.ECMWF_AIFS025_SINGLE: ModelMetaInfo(
        forecast_name_mapping=None,  # No forecast data available
        forecast_zarr_version=None,  # No forecast data available
        hindcast_zarr_version=3,
    ),
}


def get_model_meta_info(model: Model) -> ModelMetaInfo:
    return _MODEL_META_INFOR[model]
