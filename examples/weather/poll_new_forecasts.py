"""
Jua Weather SDK — Forecast availability monitor.

Uses the built-in ForecastWatcher to poll for changes to a model's
init_time. When a new model run is detected, the ``handle_new_forecast``
callback is invoked; replace its implementation with your own logic
(e.g. fetching data, triggering a downstream pipeline, or emitting a
notification).
"""

import os

from jua import JuaClient
from jua.weather import ForecastWatcher, Models
from jua.weather._types.query_response_types import LatestForecastInfo

MODELS_TO_WATCH = [Models.EPT2_HRRR]
POLL_INTERVAL_SECONDS = 60
MIN_PREDICTION_TIMEDELTA = 0


def handle_new_forecast(model_name: str, info: LatestForecastInfo) -> None:
    """Called whenever a new forecast run is detected."""
    print(f"  >>> NEW FORECAST for {model_name}")
    print(f"      init_time:            {info.init_time}")
    print(f"      prediction_timedelta: {info.prediction_timedelta}h")
    print("      (replace this with your own logic)")


def main() -> None:
    client = JuaClient()

    watcher = ForecastWatcher(
        client=client,
        models=MODELS_TO_WATCH,
        on_new_forecast=handle_new_forecast,
        interval_seconds=POLL_INTERVAL_SECONDS,
        min_prediction_timedelta=MIN_PREDICTION_TIMEDELTA,
    )

    max_cycles_env = os.environ.get("POLL_MAX_CYCLES")
    max_cycles = int(max_cycles_env) if max_cycles_env else None

    print(f"Polling every {POLL_INTERVAL_SECONDS}s — press Ctrl+C to stop.\n")
    watcher.watch(max_cycles=max_cycles)


if __name__ == "__main__":
    main()
