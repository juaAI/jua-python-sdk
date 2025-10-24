"""Example demonstrating the use of MarketZones enum.

This example shows how to use the MarketZones enum to query market aggregate data
with type-safe market zone identifiers instead of raw strings.
"""

from datetime import datetime

from jua import JuaClient
from jua.market_aggregates import AggregateVariables, ModelRuns
from jua.types import Countries, MarketZones
from jua.weather import Models


def main():
    # Initialize the Jua client
    client = JuaClient()

    # Example: Using MarketZones enum (type-safe)
    print("Example: Using MarketZones names")
    print(f"Zone name: {MarketZones.DE.zone_name}")
    print(f"Country name: {MarketZones.DE.country_name}")
    print(f"\nZone name: {MarketZones.NO_NO1.zone_name}")
    print(f"Country name: {MarketZones.NO_NO1.country_name}")
    print()

    # Create energy market using MarketZones enum
    ir_nir = client.market_aggregates.get_market([MarketZones.IE, MarketZones.GB_NIR])

    # Get the market aggregates for the latest EPT2 and ECMWF IFS runs
    model_runs = [ModelRuns(Models.EPT2, 0), ModelRuns(Models.ECMWF_IFS_SINGLE, 0)]
    ds = ir_nir.compare_runs(
        agg_variable=AggregateVariables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
        model_runs=model_runs,
        max_lead_time=24,
    )

    print("Retrieved dataset:")
    print(ds)
    print()

    # Example: Using strings
    print("Example: Using string identifiers")
    de_mz = client.market_aggregates.get_market(market_zone="DE")
    ds2 = de_mz.compare_runs(
        agg_variable=AggregateVariables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
        model_runs=model_runs,
        max_lead_time=24,
    )
    print(f"Retrieved dataset: {ds2}")
    print()

    # Example: Filtering zones by country using Countries enum
    print("Example: Filter zones by country")
    norway_zones = MarketZones.filter_by_country(Countries.NORWAY)
    print(f"Norwegian zones: {[z.zone_name for z in norway_zones]}")

    # Create market for all Norwegian zones
    norway = client.market_aggregates.get_market(market_zone=norway_zones)
    print(f"Norway market zones: {norway.zone}")
    print()

    # Example: Filter using string
    print("Example: Filter by country using string")
    german_zones = MarketZones.filter_by_country("Germany")
    print(f"German zones: {[z.zone_name for z in german_zones]}")
    print()

    # Example: Exploring available market zones
    print("Example: Exploring MarketZones")
    print(f"Total available zones: {len(list(MarketZones))}")
    print(f"Total available countries: {len(list(Countries))}")

    # Show Australian zones
    aus_zones = MarketZones.filter_by_country(Countries.AUSTRALIA)
    print(f"\nAustralian zones: {[z.zone_name for z in aus_zones]}")

    # Example: Query different variables for 3 filtered Australian market zones
    print("\nExample: Query multiple variables for Australian zones")
    australia = client.market_aggregates.get_market(market_zone=aus_zones[:3])

    wind = australia.compare_runs(
        agg_variable=AggregateVariables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
        model_runs=[ModelRuns(Models.EPT2, [0, 1, 2])],
        max_lead_time=24,
    )

    temp = australia.compare_runs(
        agg_variable=AggregateVariables.AIR_TEMPERATURE_AT_HEIGHT_LEVEL_2M,
        model_runs=[ModelRuns(Models.EPT2, [0, 1, 2])],
        max_lead_time=24,
    )

    print(f"Wind data: {wind}")
    print(f"Temperature data: {temp}")

    # Example: Fetch historical EPT2 market aggregates using specific datetimes
    print("\nExample: Historical EPT2 market aggregates for Germany")
    germany = client.market_aggregates.get_market(market_zone=MarketZones.DE)

    # Define specific historical initialization times
    historical_model_runs = [
        ModelRuns(
            Models.EPT2,
            [
                datetime(2025, 1, 1, 0),
                datetime(2025, 1, 1, 6),
                datetime(2025, 1, 1, 12),
                datetime(2025, 1, 1, 18),
            ],
        )
    ]

    # Fetch historical wind aggregates
    historical_wind = germany.compare_runs(
        agg_variable=AggregateVariables.WIND_SPEED_AT_HEIGHT_LEVEL_10M,
        model_runs=historical_model_runs,
        max_lead_time=48,
    )

    print(f"Historical wind data shape: {historical_wind.dims}")
    print(f"Model runs: {historical_wind.model_run.values}")
    print(f"Init times: {historical_wind.init_time.values}")
    print(f"Dataset:\n{historical_wind}")


if __name__ == "__main__":
    main()
