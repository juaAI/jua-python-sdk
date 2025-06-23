from collections import defaultdict
from dataclasses import dataclass

from jua.weather.models import Models


@dataclass
class ModelMetaInfo:
    """Internal class to store meta information"""

    forecast_name_mapping: str | None = None
    full_forecasted_hours: int | None = None
    has_forecast_file_access: bool = False


_MODEL_META_INFO = defaultdict(ModelMetaInfo)
_MODEL_META_INFO[Models.EPT2] = ModelMetaInfo(
    forecast_name_mapping="ept-2",
    full_forecasted_hours=480,
    has_forecast_file_access=True,
)
_MODEL_META_INFO[Models.EPT1_5] = ModelMetaInfo(
    forecast_name_mapping="ept-1.5",
    full_forecasted_hours=480,
    has_forecast_file_access=True,
)
_MODEL_META_INFO[Models.EPT1_5_EARLY] = ModelMetaInfo(
    forecast_name_mapping="ept-1.5-early",
    full_forecasted_hours=480,
    has_forecast_file_access=True,
)
_MODEL_META_INFO[Models.EPT2_EARLY] = ModelMetaInfo(
    forecast_name_mapping="ept-2-early",
    full_forecasted_hours=480,
    has_forecast_file_access=True,
)
_MODEL_META_INFO[Models.AURORA] = ModelMetaInfo(
    forecast_name_mapping="aurora",
    full_forecasted_hours=480,
    has_forecast_file_access=True,
)
_MODEL_META_INFO[Models.AIFS] = ModelMetaInfo(
    forecast_name_mapping="aifs",
    full_forecasted_hours=480,
    has_forecast_file_access=True,
)
_MODEL_META_INFO[Models.EPT2_E] = ModelMetaInfo(
    forecast_name_mapping="ept2-e",
    full_forecasted_hours=480,
    has_forecast_file_access=True,
)
_MODEL_META_INFO[Models.ECMWF_IFS_SINGLE] = ModelMetaInfo(
    has_forecast_file_access=False,
)


def get_model_meta_info(model: Models) -> ModelMetaInfo:
    return _MODEL_META_INFO[model]
