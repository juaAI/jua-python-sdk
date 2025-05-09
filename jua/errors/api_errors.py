from jua.errors.jua_error import JuaError


class NotAuthenticatedError(JuaError):
    def __init__(self, status_code: int | None = None):
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
    def __init__(self, status_code: int | None = None):
        super().__init__(
            "Unauthorized",
            details="Please check your API key and try again.",
        )


class NotFoundError(JuaError):
    def __init__(self, status_code: int | None = None):
        super().__init__(
            "Not found",
            details="The requested resource was not found.",
        )
