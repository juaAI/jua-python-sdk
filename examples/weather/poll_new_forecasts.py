"""
Jua Weather SDK – Detect when new forecasts become available.

Uses the built-in ForecastWatcher to poll for init_time changes.
When a new model run appears the ``handle_new_forecast`` callback
fires — replace its body with your own logic (fetch data, trigger a
pipeline, send a Slack message, …).

Get API credentials at https://athena.jua.ai/api-keys
"""

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

    print(f"Polling every {POLL_INTERVAL_SECONDS}s — press Ctrl+C to stop.\n")
    watcher.watch()


if __name__ == "__main__":
    main()
