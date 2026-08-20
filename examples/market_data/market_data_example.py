"""Example demonstrating the zone-addressed market_data API.

Shows how to query European power market data (renewable generation, load,
day-ahead forecasts, and prices) by market zone using a single unified
vocabulary, without caring which underlying source serves each zone.
"""

from datetime import datetime, timedelta, timezone

from jua import JuaClient


def main():
    client = JuaClient()
    md = client.market_data

    # --- Discovery ---
    print("Available zones:")
    print(f"  {md.get_zones()}")
    print()

    print("Available variables (all zones):")
    print(f"  {md.get_variables()}")
    print()

    print("Available variables for GB:")
    print(f"  {md.get_variables(market_zone='GB')}")
    print()

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=2)

    # --- Germany: renewables + day-ahead price (served from ENTSOE) ---
    print("DE solar/wind + day-ahead prices:")
    de = md.get_data(
        market_zone="DE",
        variables=["solar", "wind", "day_ahead_prices"],
        start_time=start,
        end_time=end,
        time_zone="Europe/Berlin",
    )
    print(de.head())
    print()

    # --- Germany: actual demand vs day-ahead load forecast ---
    # load_forecast is served for ENTSO-E zones (DE/FR/NL/BE) but not GB.
    print("DE load vs day-ahead load forecast:")
    de_load = md.get_data(
        market_zone="DE",
        variables=["load", "load_forecast"],
        start_time=start,
        end_time=end,
        time_zone="Europe/Berlin",
    )
    print(de_load.groupby("variable")["value"].mean().round(1))
    print()

    # --- Great Britain: renewables + the GB-only wind split ---
    # GB additionally exposes wind broken into transmission-connected and
    # distribution-embedded generation (actuals and day-ahead forecasts); the
    # SDK fetches the total and its components in one call even though the
    # backend cannot return them together.
    print("GB solar/wind + transmission/embedded split:")
    gb = md.get_data(
        market_zone="GB",
        variables=["solar", "wind", "wind_transmission", "wind_embedded"],
        start_time=start,
        end_time=end,
        time_zone="Europe/London",
    )
    print(gb.groupby("variable")["value"].mean().round(1))
    print()

    # --- Combined request across zones in one call ---
    print("Combined DE + GB solar:")
    combined = md.get_data(
        market_zone=["DE", "GB"],
        variables=["solar"],
        start_time=start,
        end_time=end,
    )
    print(combined.groupby("market_zone")["value"].describe())


if __name__ == "__main__":
    main()
