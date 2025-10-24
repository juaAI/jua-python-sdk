from dataclasses import dataclass
from datetime import datetime

from jua.weather.models import Models


@dataclass(frozen=True)
class ModelRuns:
    """Represents one or more model runs for a specific model.

    A ModelRuns is a simple data container that combines a weather model with
    one or more initialization times. The initialization times can be specified
    as datetimes or as negative integer indices to select from recent runs.

    When used with market aggregates, the integer indices are automatically
    resolved to actual datetimes by querying the API for available forecasts.
    Multiple init_times for the same model are resolved in a single API call
    for efficiency.

    Attributes:
        model: The weather model (from Models enum).
        init_times: The initialization times. Can be:
            - A single datetime or integer
            - A list of datetimes and/or integers

            For integers:
            - Non-negative integers (0 to 12) select from recent runs:
              * 0 = latest run
              * 1 = second latest run
              * 2 = third latest run, etc.

    Examples:
        >>> from jua.market_aggregates import ModelRuns
        >>> from jua.weather import Models
        >>> from datetime import datetime
        >>>
        >>> # Single datetime
        >>> run1 = ModelRuns(Models.EPT2, datetime(2024, 8, 5, 0))
        >>>
        >>> # Single integer index for latest run
        >>> run2 = ModelRuns(Models.EPT2, 0)
        >>>
        >>> # Multiple integer indices
        >>> run3 = ModelRuns(Models.EPT2, [0, 1, 2])
        >>>
        >>> # Use with market aggregates
        >>> from jua.market_aggregates import AggregateVariables
        >>> from jua.types import MarketZones
        >>> germany = client.market_aggregates.get_market(market_zone=MarketZones.DE)
        >>> data = germany.compare_runs(
        ...     agg_variable=AggregateVariables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
        ...     model_runs=[run1, run2, run3],
        ... )
    """

    model: Models
    init_times: datetime | int | list[datetime | int]

    def __post_init__(self):
        """Validate the init_times parameter and normalize to list."""
        # Normalize to list for easier handling
        if not isinstance(self.init_times, list):
            init_times_list = [self.init_times]
        else:
            init_times_list = self.init_times

        # Validate each init_time
        for init_time in init_times_list:
            if isinstance(init_time, int):
                if init_time < 0:
                    raise ValueError(
                        f"init_time as integer must be non-negative, got {init_time}"
                    )
                if init_time > 12:
                    raise ValueError(
                        f"init_time as integer must be less than 48, got {init_time}"
                    )

    @property
    def model_name(self) -> str:
        """Get the string name of the model.

        Returns:
            The model name as a string.
        """
        return self.model.value

    def get_init_times_list(self) -> list[datetime | int]:
        """Get init_times as a list.

        Returns:
            List of init_times (normalized from single value or list).
        """
        if not isinstance(self.init_times, list):
            return [self.init_times]
        return self.init_times

    def __str__(self) -> str:
        """Get a string representation of the model runs.

        Returns:
            A string showing the model and init_times.
        """
        init_times_list = self.get_init_times_list()
        if len(init_times_list) == 1:
            init_time = init_times_list[0]
            if isinstance(init_time, datetime):
                return f"{self.model_name} @ {init_time}"
            else:
                return f"{self.model_name} @ index({init_time})"
        else:
            return f"{self.model_name} @ {len(init_times_list)} init_times"
