import time

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import requests  # type: ignore[import-untyped]
from rich.progress import (
    Progress,
    TaskID,
    TextColumn,
)

from jua.errors.api_errors import (
    ConnectionBrokenError,
    RequestFailedError,
)

HTTP_OK = 200
BYTES_IN_KB = 1024
BYTES_IN_MB = 1024 * 1024
BYTES_IN_GB = 1024 * 1024 * 1024


def _format_bytes(bytes_count: int | float) -> str:
    """Format bytes as human readable string.

    Args:
        bytes_count: the number of bytes

    Returns:
        A string displaying the number of bytes with an appropriate unit.
    """
    if bytes_count < BYTES_IN_KB:
        return f"{bytes_count} B"
    elif bytes_count < BYTES_IN_MB:
        return f"{bytes_count / BYTES_IN_KB:.1f} KB"
    elif bytes_count < BYTES_IN_GB:
        return f"{bytes_count / BYTES_IN_MB:.1f} MB"
    else:
        return f"{bytes_count / BYTES_IN_GB:.1f} GB"


def process_arrow_streaming_response(
    response: requests.Response,
    print_progress: bool = True,
) -> pd.DataFrame:
    """Process a single continuous Arrow IPC stream and return a DataFrame.

    Args:
        response:

        print_progress: Whether to print the progress
    """
    response.raw.decode_content = True
    if print_progress:
        with Progress(
            TextColumn("Reading data...", justify="right"),
            "|",
            TextColumn("[bold cyan]{task.fields[size]}"),
            "|",
            TextColumn("[bold cyan]{task.fields[speed]}", justify="right"),
            transient=False,
        ) as progress:
            content_length = response.headers.get("content-length")
            task = progress.add_task(
                "reading",
                total=int(content_length) if content_length else None,
                size="0 B",
                speed="0 B/s",
            )
            return _read_stream(response, progress, task)

    return _read_stream(response)


def _read_stream(
    response: requests.Response,
    progress: Progress | None = None,
    task: TaskID | None = None,
) -> pd.DataFrame:
    """Read a table from an Arrow IPC stream.

    Args:
        response: The response from the server.

        progress: A progress bar that can be used to show the state of the download.

        task: If progress is not None, the task to update with the download progress.

    Returns:
        The received table as a pandas DataFrame

    Raises:
        ConnectionBrokenError if the stream fails to be read with a `connection broken`
        RequestFailedError: if the stream fails for any other reason
    """
    try:
        raw = response.raw
        if progress is not None and task is not None:
            raw = _RawProgressWrapper(response, progress, task)

        with pa_ipc.open_stream(raw) as reader:
            table = reader.read_all()

    except Exception as stream_err:
        # Fallback: buffer then parse
        try:
            buf = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if chunk:
                    buf.extend(chunk)
                    if progress and task:
                        progress.update(task, size=_format_bytes(len(buf)))
            table = pa_ipc.open_stream(pa.BufferReader(bytes(buf))).read_all()
        except Exception as fallback_err:
            if "connection broken" in str(stream_err).lower():
                raise ConnectionBrokenError(
                    "This likely happened because the request timed out. If this issue "
                    "persists, try making smaller requests and saving partial results."
                    "\n\nSee https://docs.jua.ai/python-sdk/weather/large-requests for "
                    "more information"
                )

            raise RequestFailedError(
                details=(
                    f"Failed to read data with error {stream_err}; fallback failed: "
                    f"{fallback_err}"
                )
            ) from fallback_err
    finally:
        if progress:
            progress.refresh()
            progress.stop()

    return table.to_pandas()


class _RawProgressWrapper:
    """Reader of raw content with a progress bar"""

    def __init__(
        self, response: requests.Response, progress: Progress, task_id: TaskID
    ):
        self._raw = response.raw
        self._progress = progress
        self._task_id = task_id
        self._start = time.time()
        self._bytes = 0
        self.closed = False

    def read(self, size: int = -1):
        chunk = self._raw.read(size)
        if chunk:
            self._bytes += len(chunk)
            elapsed = max(1e-9, time.time() - self._start)
            speed = self._bytes / elapsed
            self._progress.update(
                self._task_id,
                size=_format_bytes(self._bytes),
                speed=f"{_format_bytes(speed)}/s",
            )

        return chunk

    def readable(self):
        return True

    def flush(self):
        self._progress.refresh()

    def close(self):
        self._progress.refresh()
        if not self.closed:
            close_fn = getattr(self._raw, "close", None)
            if callable(close_fn):
                close_fn()
            self.closed = True

    def __getattr__(self, name):
        return getattr(self._raw, name)
