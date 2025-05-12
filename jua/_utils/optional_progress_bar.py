from dask.diagnostics import ProgressBar

from jua.settings.jua_settings import JuaSettings


class OptionalProgressBar:
    """Context manager for conditionally displaying a progress bar for dask operations.

    Wraps dask.diagnostics.ProgressBar to display progress only when specified
    by JuaSettings or explicit override.
    """

    def __init__(
        self, settings: JuaSettings, print_progress: bool | None = None, **kwargs
    ):
        """Initialize with settings that determine whether to show progress.

        Args:
            settings: JuaSettings instance to read default progress setting from.
            print_progress: Optional boolean to override the settings value.
            **kwargs: Additional arguments passed to dask.diagnostics.ProgressBar.
        """
        self._should_print_progress = settings.should_print_progress(print_progress)
        self._progress_bar = None

        if self._should_print_progress:
            self._progress_bar = ProgressBar(**kwargs)

    def __enter__(self):
        """Enter context manager, activating progress bar if enabled.

        Returns:
            Either the activated progress bar or self.
        """
        if self._progress_bar:
            return self._progress_bar.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context manager, deactivating progress bar if enabled.

        Returns:
            False to propagate exceptions.
        """
        if self._progress_bar:
            return self._progress_bar.__exit__(exc_type, exc_value, traceback)
        return False
