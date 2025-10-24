from collections import defaultdict
from datetime import datetime

import pandas as pd
import xarray as xr
from pydantic import validate_call

from jua._api import QueryEngineAPI
from jua._utils.remove_none_from_dict import remove_none_from_dict
from jua.client import JuaClient
from jua.market_aggregates.model_run import ModelRuns
from jua.market_aggregates.variables import AggregateVariable, AggregateVariables
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

        Returns:
            xarray.Dataset containing `model_run` and `time` dimensions, with
            `prediction_timedelta` and the queried variable as data_vars.

        Raises:
            RuntimeError: If the API request fails.

        Examples:
            >>> # Basic usage with integer indices
            >>> from jua import JuaClient
            >>> from jua.market_aggregates import AggregateVariables, ModelRuns
            >>> from jua.weather import Models
            >>> from jua.types import MarketZones
            >>>
            >>> client = JuaClient()
            >>> germany = client.market_aggregates.get_market(
            >>>     market_zone=[MarketZones.DE]
            >>> )
            >>>
            >>> # Using integers to specify the runs
            >>> model_runs = [
            ...     ModelRuns(Models.EPT2, 0),  # Latest EPT2
            ...     ModelRuns(Models.EPT1_5, [0, 1]),  # Latest 2 EPT1_5 runs
            ... ]
            >>> ds = germany.compare_runs(
            ...     agg_variable=AggregateVariables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
            ...     model_runs=model_runs,
            ...     max_lead_time=48,
            ... )
            >>>
            >>> # Using datetime to specify the runs
            >>> model_runs = [
            ...     ModelRuns(
            ...         Models.EPT2,
            ...         [
            ...             datetime(2025, 10, 2, 0),
            ...             datetime(2025, 10, 1, 0),
            ...         ]
            ...     ),
            ... ]
            >>> ds = germany.compare_runs(
            ...     agg_variable=AggregateVariables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M,
            ...     model_runs=model_runs,
            ... )
            >>>
            >>> # Query multiple variables for the same market
            >>> wind_data = germany.compare_runs(
            ...     agg_variable=AggregateVariables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
            ...     model_runs=[ModelRuns(Models.EPT2, [0, 1])],
            ... )
            >>> temp_data = germany.compare_runs(
            ...     agg_variable=AggregateVariables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M,
            ...     model_runs=[ModelRuns(Models.EPT2, [0, 1])],
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

        # Group by model and collect all init_times for each model
        all_model_runs: dict[Models, list[datetime | int]] = defaultdict(list)
        for model_run in model_runs:
            init_times_list = model_run.get_init_times_list()
            all_model_runs[model_run.model].extend(init_times_list)

        # For each model, resolve all init_times
        model_to_init_times: dict[Models, list[datetime]] = {}
        for model, init_times in all_model_runs.items():
            model_to_init_times[model] = self._resolve_init_times_for_model(
                model, init_times
            )

        # Map each init_time to the list of models that need it
        # Structure: {init_time: [model1, model2, ...]}
        init_time_to_models: dict[datetime, list[Models]] = defaultdict(list)
        for model, resolved_times in model_to_init_times.items():
            for init_time in resolved_times:
                init_time_to_models[init_time].append(model)

        # Query each init_time once with all models that need it
        all_dataframes = []
        for init_time in sorted(init_time_to_models.keys()):
            models = init_time_to_models[init_time]
            params = {
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
                    f"Failed to fetch data for models {[m.value for m in models]} at "
                    f"init_time {init_time.isoformat()}: {e}"
                ) from e

        if not all_dataframes:
            ds = xr.Dataset()
            ds.assign_attrs(**attrs)
            return ds

        df = pd.concat(all_dataframes, ignore_index=True)

        # create a model run column
        df["time"] = pd.to_datetime(df["time"])
        df["init_time"] = pd.to_datetime(df["init_time"])
        df["model_run"] = (
            df["model"] + " " + df["init_time"].dt.strftime("%Y-%m-%dT%H:%M")
        )

        # Extract unique model and init_time per model_run
        model_per_run = df.groupby("model_run")["model"].first()
        init_time_per_run = df.groupby("model_run")["init_time"].first()
        df_for_ds = df.drop(columns=["model", "init_time"])

        # generate dataset
        ds = xr.Dataset.from_dataframe(df_for_ds.set_index(["model_run", "time"]))
        ds = ds.assign_attrs(**attrs)
        ds.coords["model"] = ("model_run", model_per_run.values)
        ds.coords["init_time"] = ("model_run", init_time_per_run.values)

        # update variable name
        ds = ds.rename(name_dict={f"avg__{var.name}": var.name})
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

            # Make a single API call to get all needed forecasts
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

            # Parse the response
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

            # Check if we have enough forecasts
            if len(forecast_infos) <= max_index:
                raise ValueError(
                    f"Requested indices up to {max_index} but only "
                    f"{len(forecast_infos)} forecast(s) available for model "
                    f"{model.value}"
                )

            # Resolve each integer index
            for index in integers:
                if index >= len(forecast_infos):
                    raise ValueError(
                        f"Requested index {index} but only {len(forecast_infos)} "
                        f"forecast(s) available for model {model.value}"
                    )

                # Index 0 means latest, 1 means 2nd latest, etc.
                position = index
                forecast_info = forecast_infos[position]

                # Parse the init_time string to datetime
                init_time_str = forecast_info["init_time"]
                try:
                    # Try parsing ISO format with timezone
                    if "T" in init_time_str:
                        # Remove timezone info for simplicity
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

        # Return sorted list in increasing (chronological) order
        return sorted(resolved_times)
