from datetime import datetime, time, timedelta
from typing import Callable

from pydantic import BaseModel, Field, field_validator

from jua.weather._types.pagination import Pagination
from jua.weather.variables import Variables


class ForecastInfo(BaseModel):
    """Information about a single available forecast."""

    init_time: datetime = Field(..., description="Forecast initialization time")
    max_prediction_timedelta: int | None = Field(
        None, description="Maximum available lead time in hours for this forecast"
    )


class AvailableForecastsQueryResult(BaseModel):
    """Result containing available forecast times per model."""

    forecasts_per_model: dict[str, list[ForecastInfo]] = Field(
        description="Dictionary mapping model names to lists of available forecasts"
    )
    pagination: Pagination | None = Field(
        None, description="Pagination information if results were paginated"
    )


class AvailableForecasts:
    """A paginated result for available forecasts with convenient next page access.

    This class wraps the available forecasts result and provides an easy way to
    fetch subsequent pages of results.

    Attributes:
        forecasts: List of forecast information for the queried model
        limit: Maximum number of results per page
        offset: Current offset in the result set
        has_more: Whether more results are available

    Examples:
        >>> # Get first page of available forecasts
        >>> result = model.get_available_forecasts(limit=20)
        >>> print(f"Found {len(result.forecasts)} forecasts")
        >>>
        >>> # Fetch next page if available
        >>> if result.has_more:
        ...     next_result = result.next()
        ...     print(f"Next page has {len(next_result.forecasts)} forecasts")
    """

    def __init__(
        self,
        forecasts: list[ForecastInfo],
        pagination: Pagination,
        fetch_next: Callable[[int], "AvailableForecasts"],
    ):
        """Initialize paginated result wrapper.

        Args:
            forecasts: List of forecast information for the model
            pagination: Pagination information
            fetch_next: Callable to fetch the next page given a new offset
        """
        self._forecasts = forecasts
        self._pagination = pagination
        self._fetch_next = fetch_next

    @property
    def forecasts(self) -> list[ForecastInfo]:
        """Get the list of forecasts for this page."""
        return self._forecasts

    @property
    def limit(self) -> int:
        """Get the page size limit."""
        return self._pagination.limit

    @property
    def offset(self) -> int:
        """Get the current offset."""
        return self._pagination.offset

    @property
    def has_more(self) -> bool:
        """Check if more results are available.

        Returns True if the current page returned a full page of results,
        suggesting there may be more pages available.
        """
        return len(self._forecasts) >= self.limit

    def next(self) -> "AvailableForecasts":
        """Fetch the next page of results.

        Returns:
            A new AvailableForecasts instance with the next page of results

        Raises:
            ValueError: If no more results are available

        Examples:
            >>> result = model.get_available_forecasts(limit=20)
            >>> while result.has_more:
            ...     result = result.next()
            ...     # Process results...
        """
        if not self.has_more:
            raise ValueError("No more results available")

        next_offset = self.offset + self.limit
        return self._fetch_next(next_offset)

    def __repr__(self) -> str:
        """Get string representation of the paginated results."""
        return (
            f"<AvailableForecasts "
            f"num_forecasts={len(self._forecasts)}, offset={self.offset}, "
            f"limit={self.limit}, has_more={self.has_more}>"
        )

    def __len__(self) -> int:
        """Get the number of forecasts in the current page."""
        return len(self._forecasts)

    def __iter__(self):
        """Iterate over forecasts in the current page."""
        return iter(self._forecasts)

    def __getitem__(self, index):
        """Access forecasts by index or slice."""
        return self._forecasts[index]


class LatestForecastInfo(BaseModel):
    """Information about the latest available forecast for a model."""

    init_time: datetime = Field(..., description="Latest forecast initialization time")
    prediction_timedelta: int = Field(
        ..., description="Maximum available lead time in hours for this forecast"
    )


class LatestForecastInfoQueryResult(BaseModel):
    """Result containing the latest forecast information per model."""

    forecasts_per_model: dict[str, LatestForecastInfo] = Field(
        ...,
        description="Mapping of model identifiers to their latest forecast information",
    )


class GridInfo(BaseModel):
    """Information about the spatial grid of a forecast model."""

    num_latitudes: int = Field(
        ..., description="Number of latitude points in the model grid"
    )
    num_longitudes: int = Field(
        ..., description="Number of longitude points in the model grid"
    )


class RunDefinition(BaseModel):
    lead_time_set: list[int] = Field(
        default_factory=list,
        description="List of available lead times for this run in minutes",
    )
    dissemination_time: time | None = Field(
        default=None,
        description=(
            "Wall-clock *time of day* at which this run is published by "
            "the upstream data provider. Combined with "
            "``dissemination_day_offset`` and the init_time to get the "
            "absolute availability datetime — use "
            ":meth:`expected_dissemination_datetime` rather than "
            "interpreting this field in isolation."
        ),
    )
    dissemination_day_offset: int | None = Field(
        default=None,
        description=(
            "Days after the init date when this run is published. "
            "``None`` means 'same day as init, with an implicit +1 day "
            "rollover if ``dissemination_time`` is earlier than the "
            "init's time of day' — the convention used by every "
            "existing NWP model (e.g. ECMWF 18Z init disseminating at "
            "03:00 rolls to the next day). An explicit integer is "
            "absolute (no rollover): e.g. ``5`` for ECMWF SEAS5, which "
            "publishes on day-1 + 5 = the 6th of every month."
        ),
    )

    def expected_dissemination_datetime(
        self, init_datetime: datetime, init_time_of_day: time
    ) -> datetime | None:
        """Return the wall-clock datetime when this run is expected to be
        available, given the init datetime it was generated from.

        ``None`` is returned when ``dissemination_time`` is unknown.

        Args:
            init_datetime: The initialization datetime of this run
                (e.g. ``datetime(2026, 5, 1, 0, 0)`` for a 00Z run).
            init_time_of_day: The init's time of day (e.g. ``time(0, 0)``
                for a 00Z run) — used to decide whether the implicit
                ``<1 day`` rollover should fire when
                ``dissemination_day_offset`` is ``None``.

        Example::

            # ECMWF SEAS5: init on 2026-05-01 00:00 UTC, dissem_time=12:00,
            # dissem_day_offset=5 → available 2026-05-06 12:00 UTC.
            run.expected_dissemination_datetime(
                datetime(2026, 5, 1, 0, 0), time(0, 0)
            )
            # → datetime(2026, 5, 6, 12, 0)
        """
        if self.dissemination_time is None:
            return None
        dt = init_datetime.replace(
            hour=self.dissemination_time.hour,
            minute=self.dissemination_time.minute,
            second=0,
            microsecond=0,
        )
        if self.dissemination_day_offset is not None:
            dt += timedelta(days=self.dissemination_day_offset)
        elif self.dissemination_time < init_time_of_day:
            # Legacy implicit rollover for models that disseminate on the
            # same calendar day at a time earlier than the init — e.g.
            # ECMWF 18Z run disseminating at 03:00 next-day UTC.
            dt += timedelta(days=1)
        return dt


class GridBounds(BaseModel):
    min_lat: float = Field(
        ...,
        description="The minimum latitude of the grid",
    )
    max_lat: float = Field(
        ...,
        description="The maximum latitude of the grid",
    )
    min_lon: float = Field(
        ...,
        description="The minimum longitude of the grid",
    )
    max_lon: float = Field(
        ...,
        description="The maximum longitude of the grid",
    )


class ModelMetadata(BaseModel):
    """Metadata for a single forecast model.

    The variables field contains Variables enum members that provide rich metadata
    about each available weather variable including display names, units, and more.
    """

    name: str = Field(..., description="Model identifier")
    grid: str | None = Field(
        default=None,
        description="A human-readable description of the grid for this model",
    )
    is_ensemble_model: bool = Field(
        default=False,
        description="Whether this model is an ensemble model",
    )
    is_limited_model: bool = Field(
        default=False,
        description="Limited models only allow access to point forecasts and cannot "
        "provide access to historical data.",
    )
    variables: list[Variables] = Field(
        ...,
        description="List of available weather variables for this model.",
    )
    daily_runs: dict[time, RunDefinition] = Field(
        default_factory=dict,
        description="Mapping of daily runs to their dissemination times "
        "and available lead times",
    )
    grid_bounds: GridBounds | None = Field(
        default=None,
        description="The bounds of the grid for this model in latitude and longitude.\n"
        "Information might not be available for third party models.",
    )

    def __repr__(self) -> str:
        vars = [v.name.upper() for v in self.variables]
        return f"<ModelMetadata(model={self.name}, variables={vars}, grid={self.grid})>"

    def __str__(self) -> str:
        model_str = f"  model: {self.name}\n"
        var_str = "  variables:\n"
        for v in self.variables:
            var_str += 4 * " " + v.name.upper() + "\n"

        grid_str = ""
        if self.grid:
            grid_str = f"  grid: {self.grid}"

        return f"ModelMetadata:\n{model_str}{var_str}{grid_str}"

    @field_validator("variables", mode="before")
    @classmethod
    def parse_variables(cls, value):
        """Parse variable strings to Variables enum members.

        Converts the string variable names from the API into Variables enum members,
        providing access to additional metadata like display names and units.
        """
        if not isinstance(value, list):
            return value

        parsed_variables = []
        for var in value:
            if isinstance(var, Variables):
                # Already a Variables enum member
                parsed_variables.append(var)
            elif isinstance(var, str):
                # Try to find matching Variables enum member by name
                try:
                    # Variables enum members have .name property that matches the string
                    parsed_var = next(v for v in Variables if v.value.name == var)
                    parsed_variables.append(parsed_var)
                except StopIteration:
                    # Variable not in enum - this shouldn't happen in production
                    # but handle gracefully for forward compatibility
                    raise ValueError(f"Variable '{var}' is not recognized.")

        return parsed_variables


class MetaQueryResult(BaseModel):
    """Result containing metadata for one or more forecast models."""

    models: list[ModelMetadata] = Field(..., description="List of model metadata")
