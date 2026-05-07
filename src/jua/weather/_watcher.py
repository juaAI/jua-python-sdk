"""Polling-based watcher for detecting new forecast availability."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Callable

from pydantic import validate_call

from jua.client import JuaClient
from jua.logging import get_logger
from jua.weather._model import Model
from jua.weather._types.query_response_types import LatestForecastInfo
from jua.weather.models import Models

logger = get_logger(__name__)

OnNewForecast = Callable[[str, LatestForecastInfo], None]


class ForecastWatcher:
    """Polls for new forecast availability and triggers a callback.

    Args:
        client: An authenticated ``JuaClient``.
        models: One or more models to watch.
        on_new_forecast: Called with ``(model_name, LatestForecastInfo)``
            whenever a new init_time is detected.
        interval_seconds: Seconds between polling cycles (default 60).
        min_prediction_timedelta: Minimum forecast horizon in hours that
            the run must satisfy before it is considered "available"
            (default 0).

    Examples:
        >>> from jua import JuaClient
        >>> from jua.weather import Models, ForecastWatcher
        >>>
        >>> def handle(model_name, info):
        ...     print(f"New forecast for {model_name}: {info.init_time}")
        ...
        >>> client = JuaClient()
        >>> watcher = ForecastWatcher(
        ...     client=client,
        ...     models=[Models.EPT2_HRRR],
        ...     on_new_forecast=handle,
        ...     interval_seconds=120,
        ... )
        >>> watcher.watch()  # blocks until Ctrl-C or watcher.stop()
    """

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        client: JuaClient,
        models: list[Models],
        on_new_forecast: OnNewForecast,
        interval_seconds: int = 60,
        min_prediction_timedelta: int = 0,
    ) -> None:
        if not models:
            raise ValueError("At least one model must be specified.")
        if interval_seconds < 1:
            raise ValueError("interval_seconds must be >= 1.")

        self._client = client
        self._models: list[Models] = models
        self._on_new_forecast = on_new_forecast
        self._interval_seconds = interval_seconds
        self._min_prediction_timedelta = min_prediction_timedelta

        self._sdk_models: dict[str, Model] = {}
        for m in self._models:
            model = self._client.weather.get_model(m)
            self._sdk_models[model.name] = model

        self._latest_init: dict[str, datetime] = {}
        self._stop_event = threading.Event()

    @property
    def latest_init_times(self) -> dict[str, datetime]:
        """The last-seen init_time per model name (read-only snapshot)."""
        return dict(self._latest_init)

    def check_once(self) -> list[tuple[str, LatestForecastInfo]]:
        """Run a single polling cycle and return any new forecasts found.

        Returns:
            A list of ``(model_name, LatestForecastInfo)`` tuples for
            every model whose ``init_time`` advanced since the last
            check. The callback is also invoked for each.
        """
        new_forecasts: list[tuple[str, LatestForecastInfo]] = []
        now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

        for model_name, model in self._sdk_models.items():
            try:
                info = model.get_latest_init_time(
                    min_prediction_timedelta=self._min_prediction_timedelta,
                )
            except Exception as e:
                logger.warning(
                    "[%s] %s: error fetching latest init_time: %s",
                    now,
                    model_name,
                    e,
                )
                continue

            init_time = info.init_time
            prev = self._latest_init.get(model_name)

            if prev is None:
                self._latest_init[model_name] = init_time
                logger.info(
                    "[%s] %s: initial snapshot, init_time=%s",
                    now,
                    model_name,
                    init_time,
                )
            elif init_time > prev:
                self._latest_init[model_name] = init_time
                logger.info(
                    "[%s] %s: new init_time %s (was %s)",
                    now,
                    model_name,
                    init_time,
                    prev,
                )
                self._on_new_forecast(model_name, info)
                new_forecasts.append((model_name, info))
            else:
                logger.debug(
                    "[%s] %s: no change (latest: %s)",
                    now,
                    model_name,
                    prev,
                )

        return new_forecasts

    def watch(self, max_cycles: int | None = None) -> None:
        """Block and poll until stopped or ``max_cycles`` is reached.

        Args:
            max_cycles: Stop after this many polling cycles. ``None``
                means run indefinitely (stop via ``stop()`` or
                ``KeyboardInterrupt``).
        """
        model_names = ", ".join(self._sdk_models)
        logger.info(
            "Watching models: %s | interval=%ds | min_prediction_timedelta=%dh",
            model_names,
            self._interval_seconds,
            self._min_prediction_timedelta,
        )
        self._stop_event.clear()

        cycle = 0
        try:
            while not self._stop_event.is_set():
                if max_cycles is not None and cycle >= max_cycles:
                    break
                self.check_once()
                cycle += 1
                self._stop_event.wait(timeout=self._interval_seconds)
        except KeyboardInterrupt:
            logger.info("Stopped by KeyboardInterrupt.")

    def stop(self) -> None:
        """Signal the watcher to stop after the current cycle."""
        self._stop_event.set()
