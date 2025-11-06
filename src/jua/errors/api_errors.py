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


class RequestExceedsCreditLimitError(JuaError):
    """Error raised when API requests fail as the request exceeds the credit limit"""

    def __init__(self, message: str):
        """Initialize with optional status code.

        Args:
            message: The error message
        """
        super().__init__(
            message,
            details=(
                "Set the maximum number of credits consumed in a request with: \n"
                "  `client = JuaClient(request_credit_limit=X)`"
            ),
        )


class RequestFailedError(JuaError):
    """Error raised when streaming responses cannot be read."""

    def __init__(self, details: str | None = None):
        """Initialize with optional details."""

        super().__init__(
            "Unable to read the response from the API.",
            details=details if details else "",
        )


class ConnectionBrokenError(JuaError):
    """Error raised when streaming responses cannot be read."""

    def __init__(self, details: str | None = None):
        """Initialize with optional details."""

        super().__init__(
            "Unable to read the response from the API.",
            details=details if details else "",
        )
