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

    # --- Great Britain: renewables (served from the UK power feed) ---
    print("GB solar/wind:")
    gb = md.get_data(
        market_zone="GB",
        variables=["solar", "wind"],
        start_time=start,
        end_time=end,
        time_zone="Europe/London",
    )
    print(gb.head())
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
