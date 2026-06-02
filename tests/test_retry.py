"""Tests for the HTTP-level retry policy of the Jua API clients."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from jua._retry import _BoundedRetry, build_retry, build_session
from jua.settings.jua_settings import DEFAULT_RETRY_STATUS_CODES, JuaSettings


class _FlakyHandler(BaseHTTPRequestHandler):
    """Request handler that fails a configurable number of times before 200."""

    # Class-level config set by the test harness.
    fail_times = 0
    fail_status = 504
    retry_after = None

    # Shared counter across requests.
    request_count = 0
    lock = threading.Lock()

    def _handle(self):
        with _FlakyHandler.lock:
            _FlakyHandler.request_count += 1
            count = _FlakyHandler.request_count

        if count <= _FlakyHandler.fail_times:
            self.send_response(_FlakyHandler.fail_status)
            if _FlakyHandler.retry_after is not None:
                self.send_header("Retry-After", str(_FlakyHandler.retry_after))
            self.end_headers()
            self.wfile.write(b"error")
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

    def do_GET(self):  # noqa: N802 (http.server API)
        self._handle()

    def do_POST(self):  # noqa: N802 (http.server API)
        # Drain the request body so the connection can be reused.
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length:
            self.rfile.read(length)
        self._handle()

    def log_message(self, *args, **kwargs):  # silence test server logging
        pass


@pytest.fixture
def flaky_server():
    """Start a local HTTP server and reset the flaky handler state."""
    _FlakyHandler.fail_times = 0
    _FlakyHandler.fail_status = 504
    _FlakyHandler.retry_after = None
    _FlakyHandler.request_count = 0

    server = ThreadingHTTPServer(("127.0.0.1", 0), _FlakyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()


def _fast_settings(**overrides) -> JuaSettings:
    """Settings with zero backoff so retries do not sleep during tests."""
    params = dict(
        max_retries=3,
        retry_backoff_factor=0.0,
        retry_backoff_max=0.0,
        respect_retry_after_header=False,
    )
    params.update(overrides)
    return JuaSettings(**params)


class TestBuildRetry:
    def test_maps_settings(self):
        settings = JuaSettings(
            max_retries=5,
            retry_backoff_factor=1.5,
            retry_backoff_max=42.0,
            retry_status_codes=[500, 503],
            respect_retry_after_header=False,
        )
        retry = build_retry(settings)

        assert retry.total == 5
        assert retry.connect == 5
        assert retry.read == 5
        assert retry.status == 5
        assert retry.backoff_factor == 1.5
        assert retry.backoff_max == 42.0
        assert set(retry.status_forcelist) == {500, 503}
        assert retry.respect_retry_after_header is False
        # All methods (including POST) should be retried.
        assert retry.allowed_methods is None
        # The error response is returned (and surfaced as a JuaError) rather than
        # raising a urllib3 MaxRetryError.
        assert retry.raise_on_status is False

    def test_default_status_codes(self):
        retry = build_retry(JuaSettings())
        assert set(retry.status_forcelist) == set(DEFAULT_RETRY_STATUS_CODES)

    def test_zero_retries(self):
        retry = build_retry(JuaSettings(max_retries=0))
        assert retry.total == 0


class TestBoundedRetry:
    def _response(self, retry_after):
        class _Resp:
            headers = {"Retry-After": str(retry_after)}

            def getheader(self, name, default=None):
                if name.lower() == "retry-after":
                    return str(retry_after)
                return default

        return _Resp()

    def test_caps_retry_after(self):
        retry = _BoundedRetry(respect_retry_after_header=True, retry_after_max=60.0)
        assert retry.get_retry_after(self._response(120)) == 60.0

    def test_below_cap_is_unchanged(self):
        retry = _BoundedRetry(respect_retry_after_header=True, retry_after_max=60.0)
        assert retry.get_retry_after(self._response(10)) == 10.0

    def test_cap_survives_new(self):
        retry = _BoundedRetry(
            respect_retry_after_header=True, retry_after_max=60.0
        ).new()
        assert retry.get_retry_after(self._response(120)) == 60.0


class TestSessionRetries:
    def test_session_mounts_adapters(self):
        session = build_session(JuaSettings())
        assert set(session.adapters) >= {"http://", "https://"}

    def test_pool_size_from_settings(self):
        session = build_session(JuaSettings(connection_pool_maxsize=32))
        for scheme in ("http://", "https://"):
            adapter = session.get_adapter(scheme)
            assert adapter._pool_maxsize == 32
            assert adapter._pool_connections == 32

    def test_retries_on_504_then_succeeds(self, flaky_server):
        _FlakyHandler.fail_times = 2
        _FlakyHandler.fail_status = 504

        session = build_session(_fast_settings(max_retries=3))
        response = session.get(flaky_server)

        assert response.status_code == 200
        assert _FlakyHandler.request_count == 3

    def test_retries_post_requests(self, flaky_server):
        _FlakyHandler.fail_times = 1
        _FlakyHandler.fail_status = 503

        session = build_session(_fast_settings(max_retries=3))
        response = session.post(flaky_server, json={"hello": "world"})

        assert response.status_code == 200
        assert _FlakyHandler.request_count == 2

    def test_gives_up_after_max_retries(self, flaky_server):
        _FlakyHandler.fail_times = 100  # always fail
        _FlakyHandler.fail_status = 504

        session = build_session(_fast_settings(max_retries=2))
        response = session.get(flaky_server)

        # raise_on_status is False, so the final 504 response is returned.
        assert response.status_code == 504
        assert _FlakyHandler.request_count == 3  # initial + 2 retries

    def test_zero_retries_disables_retrying(self, flaky_server):
        _FlakyHandler.fail_times = 100
        _FlakyHandler.fail_status = 504

        session = build_session(_fast_settings(max_retries=0))
        response = session.get(flaky_server)

        assert response.status_code == 504
        assert _FlakyHandler.request_count == 1

    def test_non_retryable_status_not_retried(self, flaky_server):
        _FlakyHandler.fail_times = 100
        _FlakyHandler.fail_status = 400  # not in the retry list

        session = build_session(_fast_settings(max_retries=3))
        response = session.get(flaky_server)

        assert response.status_code == 400
        assert _FlakyHandler.request_count == 1

    def test_concurrent_requests_share_session(self, flaky_server):
        """A shared session handles many concurrent requests without error."""
        _FlakyHandler.fail_times = 0  # all succeed

        session = build_session(_fast_settings(connection_pool_maxsize=8))
        results: list[int] = []
        results_lock = threading.Lock()

        def _worker():
            resp = session.get(flaky_server)
            with results_lock:
                results.append(resp.status_code)

        threads = [threading.Thread(target=_worker) for _ in range(24)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 24
        assert all(code == 200 for code in results)
        assert _FlakyHandler.request_count == 24
