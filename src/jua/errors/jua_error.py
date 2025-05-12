class JuaError(Exception):
    """Base exception class for all Jua-specific errors."""

    def __init__(self, message: str, details: str = ""):
        """Initialize error with message and optional details.

        Args:
            message: Main error message.
            details: Additional error details. Defaults to empty string.
        """
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self):
        msg = f"{self.__class__.__name__}: {self.message}"
        if self.details:
            msg += f"\nDetails: {self.details}"
        return msg
