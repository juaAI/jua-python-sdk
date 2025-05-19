import sys
import threading
import time


class Spinner:
    """
    A simple spinner class using classic rotating line animation.
    Can be used as a context manager to show progress while a task is running.
    """

    def __init__(
        self,
        message="Loading...",
        delay=0.1,
        size=1,
        enabled: bool = True,
    ):
        """
        Initialize the spinner.

        Args:
            message (str): Message to display alongside the spinner
            delay (float): Time between spinner updates in seconds
            size (int): Size multiplier for the spinner characters
            enabled (bool): Whether to enable the spinner
        """
        self.message = message
        self.delay = delay
        self.size = size
        self.chars = ["|", "/", "-", "\\"]
        self._spinner_thread = None
        self._stop_event = threading.Event()
        self._enabled = enabled

    def _spin(self):
        """Internal method that runs the spinner animation in a loop."""
        idx = 0
        while not self._stop_event.is_set():
            char = self.chars[idx % len(self.chars)]
            # Make the spinner larger by repeating the character
            spinner_display = char * self.size

            sys.stdout.write(f"\r{self.message} {spinner_display} ")
            sys.stdout.flush()
            time.sleep(self.delay)
            idx += 1

    def start(self):
        """Start the spinner animation in a separate thread."""
        if not self._enabled:
            return

        if self._spinner_thread is not None:
            return  # Already running

        self._stop_event.clear()
        self._spinner_thread = threading.Thread(target=self._spin)
        self._spinner_thread.daemon = True
        self._spinner_thread.start()
        return self

    def stop(self):
        """Stop the spinner animation and clear the line."""
        if self._spinner_thread is None:
            return  # Not running

        self._stop_event.set()
        self._spinner_thread.join()
        self._spinner_thread = None

        # Clear the spinner line
        sys.stdout.write(f"\r{' ' * (len(self.message) + 10)}\r")
        sys.stdout.flush()

    def update(self, message):
        """Update the spinner message."""
        self.message = message

    def __enter__(self):
        """Context manager entry - starts the spinner."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stops the spinner."""
        self.stop()
