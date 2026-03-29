from collections import defaultdict
from collections.abc import Callable
from datetime import datetime

import pandas as pd
import xarray as xr
from pydantic import validate_call

from jua._api import QueryEngineAPI
from jua._utils.remove_none_from_dict import remove_none_from_dict
from jua.client import JuaClient
from jua.market_aggregates.model_run import ModelRuns
from jua.market_aggregates.variables import (
    AggregateVariable,
    AggregateVariables,
    MWWeighting,
    TemporalAggregation,
)
from jua.types import MarketZones
from jua.weather.models import Models


class EnergyMarket:
    """Represents an energy market with access to aggregated forecast data.

    An EnergyMarket provides access to spatially aggregated forecast data for specific
    market zones. Data is weighted by the appropriate factor (wind capacity, solar
    capacity, or population) depending on the variable requested. This is useful for
    energy market analysis where you need regionally aggregated forecasts.

    Examples:
        >>> from datetime import datetime
        >>>
        >>> from jua import JuaClient
        >>> from jua.market_aggregates import AggregateVariables, ModelRuns
        >>> from jua.weather import Models
        >>> from jua.types import MarketZones
        >>>
        >>> client = JuaClient()
        >>> germany = client.market_aggregates.get_market(
        ...     market_zone=[MarketZones.DE]
        ... )
        >>>
        >>> # Use ModelRuns to specify which forecasts to get
        >>> model_runs = [
        ...     ModelRuns(Models.EPT2, [0, 1]),  # Latest and 2nd latest EPT2
        ...     ModelRuns(Models.EPT1_5, 0),     # Latest EPT1_5
        ... ]
        >>>
        >>> # Get wind data for Germany
        >>> data = germany.compare_runs(
        ...     agg_variable=AggregateVariables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
        ...     model_runs=model_runs,
        ...     max_lead_time=48,
        ... )
    """

    def __init__(
        self,
        client: JuaClient,
        market_zone: MarketZones | str | list[MarketZones | str],
    ):
        """Initialize an energy market instance.

        Args:
            client: JuaClient instance for API communication.
            market_zone: The market zones or list of market zones to aggregate data for.
        """
        self._client = client
        self._query_engine_api = QueryEngineAPI(jua_client=self._client)

        # Convert MarketZones enum to strings if needed
        if isinstance(market_zone, (MarketZones, str)):
            market_zone = [market_zone]
        self.market_zone = [
            z.zone_name if isinstance(z, MarketZones) else str(z) for z in market_zone
        ]

    @property
    def zone(self) -> list[str]:
        """Get the market zones for this market.

        Returns:
            List of market zone identifiers.
        """
        return self.market_zone

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def compare_runs(
        self,
        agg_variable: AggregateVariable | AggregateVariables,
        model_runs: list[ModelRuns],
        min_lead_time: int = 0,
        max_lead_time: int | None = None,
        temporal_aggregation: TemporalAggregation | None = None,
    ) -> xr.Dataset:
        """Compare multiple model runs for a specific variable in this market.

        This method fetches spatially aggregated forecast data for the market zones
        configured in this EnergyMarket instance. The aggregation automatically uses
        the appropriate weighting for the specified variable.

        Args:
            agg_variable: The AggregateVariable specifying which variable to query.

            model_runs: List of ModelRuns instances specifying which model forecasts to
                query. Each ModelRuns contains a model and one or more init_times
                (datetimes or non-negative integers).

            min_lead_time: Minimum forecast lead time in hours (default: 0).

            max_lead_time: Maximum forecast lead time in hours.
                If None, returns all available lead times.

            temporal_aggregation: Optional temporal resampling configuration.
                When provided, the ``time`` dimension is resampled to the
                specified frequency (e.g. daily) using the chosen method
                (e.g. mean, sum). Applied client-side after fetching data.

        Returns:
            xarray.Dataset containing ``model_run`` and ``time`` dimensions, with
            ``prediction_timedelta`` and the queried variable as data_vars.
            When ``temporal_aggregation`` is set, the ``time`` dimension reflects
            the resampled frequency.

        Raises:
            RuntimeError: If the API request fails.

        Examples:
            >>> from jua import JuaClient
            >>> from jua.market_aggregates import (
            ...     AggregateVariables, ModelRuns,
            ...     TemporalAggregation, AggregationFrequency, AggregationMethod,
            ... )
            >>> from jua.weather import Models
            >>> from jua.types import MarketZones
            >>>
            >>> client = JuaClient()
            >>> germany = client.market_aggregates.get_market(
            >>>     market_zone=[MarketZones.DE]
            >>> )
            >>>
            >>> # Hourly data (default)
            >>> ds = germany.compare_runs(
            ...     agg_variable=AggregateVariables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
            ...     model_runs=[ModelRuns(Models.EPT2, 0)],
            ...     max_lead_time=48,
            ... )
            >>>
            >>> # Daily mean aggregation
            >>> ds_daily = germany.compare_runs(
            ...     agg_variable=AggregateVariables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
            ...     model_runs=[ModelRuns(Models.EPT2, 0)],
            ...     max_lead_time=120,
            ...     temporal_aggregation=TemporalAggregation(
            ...         AggregationFrequency.DAILY, AggregationMethod.MEAN,
            ...     ),
            ... )
        """
        if isinstance(agg_variable, AggregateVariables):
            var = agg_variable.value
        else:
            var = agg_variable

        attrs = {
            "var_name": var.name,
            "var_display_name": var.display_name,
            "unit": var.unit,
            "weighting": var.weighting,
            "market_zone": self.market_zone,
            "min_lead_time": min_lead_time,
            "max_lead_time": max_lead_time,
        }

        def _build_params(models: list[Models], init_time: datetime) -> dict:
            params: dict = {
                "models": [m.value for m in models],
                "init_time": init_time.isoformat(),
                "weighting": var.weighting.value,
                "variables": [var.name],
                "market_zones": self.market_zone,
                "include_time": True,
            }
            if min_lead_time > 0:
                params["min_prediction_timedelta"] = min_lead_time
            if max_lead_time is not None:
                params["max_prediction_timedelta"] = max_lead_time
            return params

        all_dataframes = self._fetch_dataframes(
            model_runs=model_runs,
            build_params=_build_params,
            error_context="data",
        )

        if not all_dataframes:
            ds = xr.Dataset()
            ds.assign_attrs(**attrs)
            return ds

        ds = self._build_dataset(all_dataframes, attrs)

        # Rename the prefixed variable column to the clean variable name
        avg_col = f"avg__{var.name}"
        if avg_col in ds.data_vars:
            ds = ds.rename(name_dict={avg_col: var.name})

        if temporal_aggregation is not None:
            ds = _apply_temporal_aggregation(ds, temporal_aggregation)

        return ds

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def compare_runs_mw(
        self,
        weighting: MWWeighting,
        model_runs: list[ModelRuns],
        min_lead_time: int = 0,
        max_lead_time: int | None = None,
        temporal_aggregation: TemporalAggregation | None = None,
    ) -> xr.Dataset:
        """Compare multiple model runs with output in MW.

        Like :meth:`compare_runs`, but applies power curves and returns
        predicted megawatt (MW) values instead of raw weather variables.

        The response columns depend on the weighting:

        - ``"wind_capacity"`` -> ``wind_onshore_mw``, ``wind_offshore_mw``
        - ``"solar_capacity"`` -> ``solar_mw``

        Args:
            weighting: Capacity weighting scheme. Must be
                ``"wind_capacity"`` or ``"solar_capacity"``.

            model_runs: List of ModelRuns instances specifying which model
                forecasts to query.

            min_lead_time: Minimum forecast lead time in hours (default: 0).

            max_lead_time: Maximum forecast lead time in hours. If ``None``,
                returns all available lead times.

            temporal_aggregation: Optional temporal resampling configuration.
                When provided, the ``time`` dimension is resampled to the
                specified frequency (e.g. daily) using the chosen method
                (e.g. mean, sum). Applied client-side after fetching data.

        Returns:
            ``xarray.Dataset`` with ``model_run`` and ``time`` dimensions
            and MW data variables (e.g. ``wind_onshore_mw``).
            When ``temporal_aggregation`` is set, the ``time`` dimension
            reflects the resampled frequency.

        Raises:
            RuntimeError: If the API request fails.

        Examples:
            >>> from jua import JuaClient
            >>> from jua.market_aggregates import (
            ...     ModelRuns,
            ...     TemporalAggregation, AggregationFrequency, AggregationMethod,
            ... )
            >>> from jua.weather import Models
            >>> from jua.types import MarketZones
            >>>
            >>> client = JuaClient()
            >>> germany = client.market_aggregates.get_market(MarketZones.DE)
            >>>
            >>> # Hourly MW data
            >>> ds = germany.compare_runs_mw(
            ...     weighting="wind_capacity",
            ...     model_runs=[ModelRuns(Models.EPT2, [0, 1])],
            ...     max_lead_time=48,
            ... )
            >>>
            >>> # Daily mean MW data
            >>> ds_daily = germany.compare_runs_mw(
            ...     weighting="wind_capacity",
            ...     model_runs=[ModelRuns(Models.EPT2, [0, 1])],
            ...     max_lead_time=120,
            ...     temporal_aggregation=TemporalAggregation(
            ...         AggregationFrequency.DAILY,
            ...     ),
            ... )
        """
        attrs = {
            "unit": "MW",
            "weighting": weighting,
            "market_zone": self.market_zone,
            "min_lead_time": min_lead_time,
            "max_lead_time": max_lead_time,
        }

        def _build_params(models: list[Models], init_time: datetime) -> dict:
            params: dict = {
                "models": [m.value for m in models],
                "init_time": init_time.isoformat(),
                "weighting": weighting,
                "market_zones": self.market_zone,
                "include_time": True,
                "unit": "mw",
            }
            if min_lead_time > 0:
                params["min_prediction_timedelta"] = min_lead_time
            if max_lead_time is not None:
                params["max_prediction_timedelta"] = max_lead_time
            return params

        all_dataframes = self._fetch_dataframes(
            model_runs=model_runs,
            build_params=_build_params,
            error_context="MW data",
        )

        if not all_dataframes:
            ds = xr.Dataset()
            ds.assign_attrs(**attrs)
            return ds

        ds = self._build_dataset(all_dataframes, attrs)

        if temporal_aggregation is not None:
            ds = _apply_temporal_aggregation(ds, temporal_aggregation)

        return ds

    def __repr__(self) -> str:
        """Get string representation of the energy market.

        Returns:
            A string representation suitable for debugging.
        """
        return f"<EnergyMarket zones={self.zone}>"

    def __str__(self) -> str:
        """Get string representation of the energy market.

        Returns:
            A string representation with the market zones.
        """
        return f"EnergyMarket({', '.join(self.zone)})"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_dataframes(
        self,
        model_runs: list[ModelRuns],
        build_params: Callable[[list[Models], datetime], dict],
        error_context: str,
    ) -> list[pd.DataFrame]:
        """Resolve model runs and fetch data from the query engine.

        Args:
            model_runs: List of ModelRuns to query.
            build_params: Callable ``(models, init_time) -> dict`` that
                produces the query-string parameters for a single API call.
            error_context: Label used in error messages (e.g. "data", "MW data").

        Returns:
            List of DataFrames, one per successful API response.
        """
        all_model_runs: dict[Models, list[datetime | int]] = defaultdict(list)
        for model_run in model_runs:
            all_model_runs[model_run.model].extend(model_run.get_init_times_list())

        model_to_init_times: dict[Models, list[datetime]] = {}
        for model, init_times in all_model_runs.items():
            model_to_init_times[model] = self._resolve_init_times_for_model(
                model, init_times
            )

        init_time_to_models: dict[datetime, list[Models]] = defaultdict(list)
        for model, resolved_times in model_to_init_times.items():
            for init_time in resolved_times:
                init_time_to_models[init_time].append(model)

        all_dataframes: list[pd.DataFrame] = []
        for init_time in sorted(init_time_to_models.keys()):
            models = init_time_to_models[init_time]
            params = build_params(models, init_time)

            try:
                response = self._query_engine_api.get(
                    "forecast/market-aggregate",
                    params=remove_none_from_dict(params),
                    requires_auth=True,
                )
                data = response.json()
                df = pd.DataFrame(data)
                if not df.empty:
                    all_dataframes.append(df)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to fetch {error_context} for models "
                    f"{[m.value for m in models]} at "
                    f"init_time {init_time.isoformat()}: {e}"
                ) from e

        return all_dataframes

    @staticmethod
    def _build_dataset(
        dataframes: list[pd.DataFrame],
        attrs: dict,
    ) -> xr.Dataset:
        """Combine fetched DataFrames into an ``xr.Dataset``.

        Args:
            dataframes: Non-empty list of DataFrames from the query engine.
            attrs: Attributes to attach to the dataset.

        Returns:
            An ``xr.Dataset`` with ``model_run`` and ``time`` dimensions.
        """
        df = pd.concat(dataframes, ignore_index=True)

        df["time"] = pd.to_datetime(df["time"])
        df["init_time"] = pd.to_datetime(df["init_time"])
        df["model_run"] = (
            df["model"] + " " + df["init_time"].dt.strftime("%Y-%m-%dT%H:%M")
        )

        model_per_run = df.groupby("model_run")["model"].first()
        init_time_per_run = df.groupby("model_run")["init_time"].first()
        df_for_ds = df.drop(columns=["model", "init_time"])

        ds = xr.Dataset.from_dataframe(df_for_ds.set_index(["model_run", "time"]))
        ds = ds.assign_attrs(**attrs)
        ds.coords["model"] = ("model_run", model_per_run.values)
        ds.coords["init_time"] = ("model_run", init_time_per_run.values)

        return ds

    def _resolve_init_times_for_model(
        self, model: Models, init_times: list[datetime | int]
    ) -> list[datetime]:
        """Resolve multiple init_times for a model in a single API call.

        This method efficiently resolves all integer indices for a model by making
        a single API call with limit=max(abs(all_integers)).

        Args:
            model: The model to query.
            init_times: List of init_times (mix of datetimes and integers).

        Returns:
            List of resolved datetimes, sorted in increasing (chronological) order,
            with duplicates removed.

        Raises:
            ValueError: If any integer index is out of range.
            RuntimeError: If the API call fails.
        """
        resolved_times = set()

        # Separate datetimes and integers
        datetimes = [t for t in init_times if isinstance(t, datetime)]
        integers = [t for t in init_times if isinstance(t, int)]

        # Add datetimes directly
        resolved_times.update(datetimes)

        # If there are integers, resolve them via API call
        if integers:
            # Find the maximum index needed (highest number = furthest back)
            max_index = max(integers)

            # Need limit = max_index + 1 because 0-indexed (index 0 requires 1 item,
            # index 2 requires 3 items, etc.)
            params = {
                "models": [model.value],
                "limit": max_index + 1,
                "offset": 0,
            }

            try:
                response = self._query_engine_api.get(
                    "forecast/available-forecasts",
                    params=params,
                    requires_auth=True,
                )
                data = response.json()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to fetch available forecasts for {model.value}: {e}"
                ) from e

            forecasts_per_model = data.get("forecasts_per_model", {})
            model_key = model.value

            if model_key not in forecasts_per_model:
                raise RuntimeError(
                    f"No forecasts found for model {model_key} in API response"
                )

            forecast_infos = forecasts_per_model[model_key]

            if not forecast_infos:
                raise ValueError(
                    f"No available forecasts found for model {model.value}"
                )

            if len(forecast_infos) <= max_index:
                raise ValueError(
                    f"Requested indices up to {max_index} but only "
                    f"{len(forecast_infos)} forecast(s) available for model "
                    f"{model.value}"
                )

            for index in integers:
                if index >= len(forecast_infos):
                    raise ValueError(
                        f"Requested index {index} but only {len(forecast_infos)} "
                        f"forecast(s) available for model {model.value}"
                    )

                position = index
                forecast_info = forecast_infos[position]

                init_time_str = forecast_info["init_time"]
                try:
                    if "T" in init_time_str:
                        init_time_str = (
                            init_time_str.replace("Z", "").split("+")[0].split(".")[0]
                        )
                        resolved_time = datetime.fromisoformat(init_time_str)
                    else:
                        resolved_time = datetime.fromisoformat(init_time_str)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to parse init_time '{init_time_str}': {e}"
                    ) from e

                resolved_times.add(resolved_time)

        return sorted(resolved_times)


def _apply_temporal_aggregation(
    ds: xr.Dataset,
    aggregation: TemporalAggregation,
) -> xr.Dataset:
    """Resample the ``time`` dimension of a dataset.

    Applies the aggregation per ``model_run`` group so that each run is
    resampled independently.

    Args:
        ds: Dataset with a ``time`` dimension.
        aggregation: Resampling configuration (frequency + method).

    Returns:
        A new Dataset with the ``time`` dimension resampled.
    """
    attrs = ds.attrs.copy()
    coord_attrs = {name: ds.coords[name].attrs.copy() for name in ds.coords}

    # prediction_timedelta is derived from init_time + time, so it cannot
    # survive a resample in a meaningful way — drop before resampling and
    # let users recompute if needed.
    has_prediction_timedelta = "prediction_timedelta" in ds.data_vars
    if has_prediction_timedelta:
        ds = ds.drop_vars("prediction_timedelta")

    resampler = ds.resample(time=aggregation.frequency.value)
    method_fn = getattr(resampler, aggregation.method.value)
    ds = method_fn()

    attrs["temporal_aggregation_frequency"] = aggregation.frequency.value
    attrs["temporal_aggregation_method"] = aggregation.method.value
    ds = ds.assign_attrs(**attrs)

    for name, a in coord_attrs.items():
        if name in ds.coords:
            ds.coords[name].attrs.update(a)

    return ds
