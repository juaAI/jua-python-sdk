"""Query realised (actual) power market data for Germany and Great Britain.

Uses the zone-addressed ``market_data`` API to pull observed solar generation,
wind generation, and load for ``DE`` and ``GB`` over a recent window. The same
unified call works for both zones even though they are served by different
underlying sources (ENTSO-E for DE, the UK-power feed for GB).
"""

from datetime import datetime, timedelta, timezone

from jua import JuaClient

# Realised quantities (no forecasts, no prices) and the local time zone to
# return each zone's ``time`` column in.
ACTUAL_VARIABLES = ["solar", "wind", "load"]
ZONE_TIME_ZONE = {
    "DE": "Europe/Berlin",
    "GB": "Europe/London",
}


def main():
    client = JuaClient()
    md = client.market_data

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=2)

    for zone, time_zone in ZONE_TIME_ZONE.items():
        print(f"=== {zone} actuals ({time_zone}) ===")
        df = md.get_data(
            market_zone=zone,
            variables=ACTUAL_VARIABLES,
            start_time=start,
            end_time=end,
            time_zone=time_zone,
        )
        if df.empty:
            print("  No data returned.\n")
            continue

        print(df.head())
        print()
        print("Mean by variable:")
        print(df.groupby("variable")["value"].mean().round(1))
        print()


if __name__ == "__main__":
    main()
