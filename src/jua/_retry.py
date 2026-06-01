"""HTTP retry configuration for the Jua API clients.

This module builds a :class:`requests.Session` with a permissive but bounded
retry policy so that transient failures (connection errors and retryable status
codes such as ``502``/``503``/``504``) are retried at the HTTP level, below the
"retry the whole job" granularity.
"""

from logging import getLogger

import requests  # type: ignore[import-untyped]
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from jua.settings.jua_settings import JuaSettings

logger = getLogger(__name__)


class _BoundedRetry(Retry):
    """``urllib3`` retry policy that caps the honored ``Retry-After`` delay.

    Servers (e.g. Cloudflare on a ``504``) can return a large ``Retry-After``
    value such as 120 seconds. Honoring that verbatim on every retry would block
    the caller for minutes, so we cap it by ``retry_backoff_max`` while still
    respecting the server's request to back off.
    """

    def __init__(self, *args, retry_after_max: float | None = None, **kwargs):
        self._retry_after_max = retry_after_max
        super().__init__(*args, **kwargs)

    def new(self, **kwargs):
        # Preserve our custom attribute across the copies urllib3 makes
        # internally (it calls ``new`` on every increment).
        kwargs.setdefault("retry_after_max", self._retry_after_max)
        return super().new(**kwargs)

    def get_retry_after(self, response):
        retry_after = super().get_retry_after(response)
        if retry_after is None:
            return None
        if self._retry_after_max is not None:
            return min(retry_after, self._retry_after_max)
        return retry_after


def build_retry(settings: JuaSettings) -> Retry:
    """Build a :class:`urllib3.util.retry.Retry` from the client settings."""
    return _BoundedRetry(
        total=settings.max_retries,
        connect=settings.max_retries,
        read=settings.max_retries,
        status=settings.max_retries,
        backoff_factor=settings.retry_backoff_factor,
        backoff_max=settings.retry_backoff_max,
        status_forcelist=frozenset(settings.retry_status_codes),
        # The Jua endpoints are read-only queries even when issued as POST, so it
        # is safe to retry every HTTP method.
        allowed_methods=None,
        respect_retry_after_header=settings.respect_retry_after_header,
        retry_after_max=settings.retry_backoff_max,
        raise_on_status=False,
    )


def build_session(settings: JuaSettings) -> requests.Session:
    """Create a :class:`requests.Session` configured with the retry policy.

    Args:
        settings: Client settings controlling the retry behavior.

    Returns:
        A session whose ``http://`` and ``https://`` adapters retry transient
        failures according to ``settings``.
    """
    session = requests.Session()
    retry = build_retry(settings)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
