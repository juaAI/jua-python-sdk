from dask.diagnostics import ProgressBar

from jua.settings.jua_settings import JuaSettings


class OptionalProgressBar:
    def __init__(
        self, settings: JuaSettings, print_progress: bool | None = None, **kwargs
    ):
        self._should_print_progress = settings.should_print_progress(print_progress)
        self._progress_bar = None

        if self._should_print_progress:
            self._progress_bar = ProgressBar(**kwargs)

    def __enter__(self):
        if self._progress_bar:
            return self._progress_bar.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._progress_bar:
            return self._progress_bar.__exit__(exc_type, exc_value, traceback)
        return False
