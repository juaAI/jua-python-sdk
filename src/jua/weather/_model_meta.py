from collections import defaultdict
from dataclasses import dataclass

from jua.weather.models import Models


@dataclass
class ModelMetaInfo:
    """Internal class to store meta information"""

    forecast_name_mapping: str | None = None
    full_forecasted_hours: int | None = None
    is_jua_model: bool = False


_MODEL_META_INFO = defaultdict(ModelMetaInfo)
_MODEL_META_INFO[Models.EPT2] = ModelMetaInfo(
    forecast_name_mapping="ept-2",
    full_forecasted_hours=480,
    is_jua_model=True,
)
_MODEL_META_INFO[Models.EPT1_5] = ModelMetaInfo(
    forecast_name_mapping="ept-1.5",
    full_forecasted_hours=480,
    is_jua_model=True,
)
_MODEL_META_INFO[Models.EPT1_5_EARLY] = ModelMetaInfo(
    forecast_name_mapping="ept-1.5-early",
    full_forecasted_hours=480,
    is_jua_model=True,
)
_MODEL_META_INFO[Models.ECMWF_AIFS025_SINGLE] = ModelMetaInfo(
    forecast_name_mapping=None,  # No forecast data available
)


def get_model_meta_info(model: Models) -> ModelMetaInfo:
    return _MODEL_META_INFO[model]
