from jua.errors.jua_error import JuaError


class NotAuthenticatedError(JuaError):
    """Error raised when API requests fail due to missing or invalid authentication."""

    def __init__(self, status_code: int | None = None):
        """Initialize with optional status code.

        Args:
            status_code: HTTP status code from the failed request.
        """
        super().__init__(
            "Not authenticated",
            details="Please check your API key and try again.",
        )
        self.status_code = status_code

    def __str__(self):
        msg = super().__str__()
        if self.status_code:
            msg += f"\nStatus code: {self.status_code}"
        return msg


class UnauthorizedError(JuaError):
    """Error raised when API requests are rejected due to insufficient permissions."""

    def __init__(self, status_code: int | None = None):
        """Initialize with optional status code.

        Args:
            status_code: HTTP status code from the failed request.
        """
        super().__init__(
            "Unauthorized",
            details="Please check your API key and try again.",
        )


class NotFoundError(JuaError):
    """Error raised when a requested resource does not exist."""

    def __init__(self, status_code: int | None = None):
        """Initialize with optional status code.

        Args:
            status_code: HTTP status code from the failed request.
        """
        super().__init__(
            "Not found",
            details="The requested resource was not found.",
        )
