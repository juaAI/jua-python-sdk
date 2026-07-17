"""Example demonstrating the use of the PowerForecast API.

This example shows how to query renewable energy generation forecasts in MW
using the Jua SDK's power forecast interface.
"""

from datetime import datetime, timezone

from jua import JuaClient


def main():
    client = JuaClient()
    pf = client.power_forecast

    # --- Metadata ---

    print("Available zones:")
    zones = pf.get_zones()
    print(f"  {zones}")
    print()

    print("Available PSR types:")
    psr_types = pf.get_psr_types()
    print(f"  {psr_types}")
    print()

    if zones:
        zone = zones[0]
        print(f"PSR types for {zone}:")
        zone_psr_types = pf.get_psr_types(zone_key=zone)
        print(f"  {zone_psr_types}")
        print()

        print(f"Recent init times for {zone}:")
        init_times = pf.get_init_times(zone_key=zone, limit=5)
        for it in init_times:
            print(f"  {it.init_time}  (max horizon: {it.max_prediction_timedelta} min)")
        print()

    # --- Serving versions (stable / latest / pin a run id) ---
    #
    # Omitting ``version`` (or passing ``version="stable"``) follows the
    # production fleet. ``version="latest"`` follows the preview pointers.
    # To freeze today's stable checkpoint so a later promote does not change
    # your query, take ``model_version`` where ``is_stable`` is true.

    print("Serving catalog for DE Solar:")
    versions = pf.get_versions(zone_key="DE", psr_type="Solar")
    for v in versions:
        flags = []
        if v.is_stable:
            flags.append("stable")
        if v.is_latest:
            flags.append("latest")
        label = f" [{', '.join(flags)}]" if flags else ""
        print(f"  {v.model_version}{label}")
    print()

    stable = next((v for v in versions if v.is_stable), None)
    if stable is not None:
        print(f"Pinned stable run id for DE Solar: {stable.model_version}")
        ds_pinned = pf.get_data(
            zone_keys=["DE"],
            psr_types=["Solar"],
            init_time="latest",
            max_prediction_timedelta=2880,
            version=stable.model_version,
        )
        print(ds_pinned)
        print()

    print("Preview pointers: version='latest' for DE Solar")
    ds_latest = pf.get_data(
        zone_keys=["DE"],
        psr_types=["Solar"],
        init_time="latest",
        max_prediction_timedelta=2880,
        version="latest",
    )
    print(ds_latest)
    print()

    print("Per-cell pins: DE Solar on latest, DE Load on stable")
    ds_pins = pf.get_data(
        zone_keys=["DE"],
        psr_types=["Solar", "Load"],
        init_time="latest",
        max_prediction_timedelta=1440,
        version="stable",
        version_pins=[
            {"zone_key": "DE", "psr_type": "Solar", "version": "latest"},
        ],
    )
    print(ds_pins)
    print()

    # --- Horizon mode: latest forecast ---

    print("Horizon mode: latest forecast for all zones, Solar (stable default)")
    ds = pf.get_data(
        zone_keys=zones,
        psr_types=["Solar"],
        init_time="latest",
        max_prediction_timedelta=2880,
    )
    print(ds)
    print()

    # --- Horizon mode: specific init time ---

    if zones:
        print(f"Horizon mode: latest 2 forecasts for {zones[0]}, all PSR types")
        ds2 = pf.get_data(
            zone_keys=[zones[0]],
            init_time=[0, 1],
            max_prediction_timedelta=1440,
        )
        print(ds2)
        print()

    # --- Time range mode ---

    print("Time range mode: last 24h for all zones")
    now = datetime.now(timezone.utc)
    ds3 = pf.get_data(
        zone_keys=zones,
        start_time=now.replace(hour=0, minute=0, second=0, microsecond=0),
        end_time=now,
    )
    print(ds3)
    print()

    # --- Stitched day-ahead series (by init hour) ---
    if zones:
        print(f"Stitched day-ahead series for {zones[0]} Solar, runs at 09:00 UTC")
        ds4 = pf.get_day_ahead_timeseries(
            zone_keys=[zones[0]],
            psr_types=["Solar"],
            init_hour=9,  # e.g. D-1 09:00
            time_zone="UTC",
            max_init_times=10,  # stitch up to 10 matching days
        )
        print(ds4)


if __name__ == "__main__":
    main()
