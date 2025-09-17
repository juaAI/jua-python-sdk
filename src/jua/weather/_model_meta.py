from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from jua.weather.models import Models


def _get_hours_since_march_31_2024() -> int:
    """Calculate hours from March 31, 2024 to today.
    
    This ensures that any forecast after March 31, 2024 will use API,
    and the threshold automatically updates each day.
    """
    march_31_2024 = datetime(2024, 3, 31)
    today = datetime.now()
    hours_diff = int((today - march_31_2024).total_seconds() / 3600)
    return hours_diff


@dataclass
class ModelMetaInfo:
    """Internal class to store meta information"""

    forecast_name_mapping: str | None = None
    full_forecasted_hours: int | None = None
    has_forecast_file_access: bool = False
    has_statistics: bool = False
    max_init_time_past_for_api_hours: int | None = 36 

    def get_api_threshold_hours(self) -> int:
        """Get the API threshold hours for this model.
        
        If max_init_time_past_for_api_hours is None, uses dynamic calculation
        from March 31, 2024 to today.
        """
        if self.max_init_time_past_for_api_hours is not None:
            return self.max_init_time_past_for_api_hours
        return _get_hours_since_march_31_2024()


_MODEL_META_INFO = defaultdict(ModelMetaInfo)
_MODEL_META_INFO[Models.EPT2] = ModelMetaInfo(
    forecast_name_mapping="ept-2",
    full_forecasted_hours=480,
    has_forecast_file_access=True,
    max_init_time_past_for_api_hours = None
)
_MODEL_META_INFO[Models.EPT2_RR] = ModelMetaInfo(
    forecast_name_mapping="ept-2-rr",
    full_forecasted_hours=48,
    has_forecast_file_access=True,
    max_init_time_past_for_api_hours=36,  
    
)
_MODEL_META_INFO[Models.EPT1_5] = ModelMetaInfo(
    forecast_name_mapping="ept-1.5-b",
    full_forecasted_hours=480,
    has_forecast_file_access=True,
    max_init_time_past_for_api_hours=36,  
)
_MODEL_META_INFO[Models.EPT1_5_EARLY] = ModelMetaInfo(
    forecast_name_mapping="ept-1.5-early-b",
    full_forecasted_hours=480,
    has_forecast_file_access=True,
    max_init_time_past_for_api_hours=36,  
)
_MODEL_META_INFO[Models.EPT2_EARLY] = ModelMetaInfo(
    forecast_name_mapping="ept-2-early",
    full_forecasted_hours=480,
    has_forecast_file_access=True,
    max_init_time_past_for_api_hours=36,  
)
_MODEL_META_INFO[Models.AURORA] = ModelMetaInfo(
    forecast_name_mapping="aurora",
    full_forecasted_hours=480,
    has_forecast_file_access=True,
    max_init_time_past_for_api_hours=36,  
)
_MODEL_META_INFO[Models.AIFS] = ModelMetaInfo(
    forecast_name_mapping="aifs",
    full_forecasted_hours=480,
    has_forecast_file_access=True,
    max_init_time_past_for_api_hours=36,  
)
_MODEL_META_INFO[Models.EPT2_E] = ModelMetaInfo(
    forecast_name_mapping="ept2-e",
    full_forecasted_hours=480,
    has_forecast_file_access=True,
    has_statistics=True,
    max_init_time_past_for_api_hours=36,  
)
_MODEL_META_INFO[Models.ECMWF_AIFS_ENSEMBLE] = ModelMetaInfo(
    forecast_name_mapping="ecmwf_aifs025_ensemble",
    full_forecasted_hours=360,
    has_forecast_file_access=False,
    has_statistics=True,
    max_init_time_past_for_api_hours=36,  
)
_MODEL_META_INFO[Models.ECMWF_IFS_SINGLE] = ModelMetaInfo(
    has_forecast_file_access=False,
)


def get_model_meta_info(model: Models) -> ModelMetaInfo:
    return _MODEL_META_INFO[model]
