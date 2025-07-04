import importlib.metadata

import requests  # type: ignore[import-untyped]

from jua.client import JuaClient
from jua.errors.api_errors import (
    NotAuthenticatedError,
    NotFoundError,
    UnauthorizedError,
)
from jua.errors.jua_error import JuaError


class API:
    """Internal HTTP client for Jua API communication.

    This class handles API requests, authentication, URL construction,
    and error handling. Not intended for direct use by SDK users.
    """

    def __init__(self, jua_client: JuaClient):
        """Initialize API client with Jua client reference.

        Args:
            jua_client: Client instance containing configuration settings.
        """
        self._jua_client = jua_client
        self._user_agent = _get_user_agent()

    def _get_headers(self, requires_auth: bool = True) -> dict:
        """Construct HTTP headers for API requests.

        Args:
            requires_auth: Whether to include authentication credentials.

        Returns:
            Dictionary of HTTP headers.
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self._user_agent,
        }
        if requires_auth:
            auth_settings = self._jua_client.settings.auth
            headers["X-API-Key"] = (
                f"{auth_settings.api_key_id}:{auth_settings.api_key_secret}"
            )
        return headers

    def _validate_response_status(self, response: requests.Response) -> None:
        """Check response status code and raise appropriate errors.

        Only validates the HTTP status code, not the response content.

        Args:
            response: HTTP response to validate.

        Raises:
            NotAuthenticatedError: For 401 responses.
            UnauthorizedError: For 403 responses.
            NotFoundError: For 404 responses.
            JuaError: For other error responses.
        """
        if response.ok:
            return

        # Throw not authenticated error
        if response.status_code == 401:
            raise NotAuthenticatedError(response.status_code)

        if response.status_code == 403:
            raise UnauthorizedError(response.status_code)

        if response.status_code == 404:
            raise NotFoundError(response.status_code)

        raise JuaError(
            f"Unexpected status code: {response.status_code}",
            details=response.text,
        )

    def _get_url(self, url: str) -> str:
        """Construct full API URL from endpoint path.

        Args:
            url: API endpoint path.

        Returns:
            Complete URL including API base URL and version.
        """
        return (
            f"{self._jua_client.settings.api_url}/"
            f"{self._jua_client.settings.api_version}/{url}"
        )

    def get(
        self, url: str, params: dict | None = None, requires_auth: bool = True
    ) -> requests.Response:
        """Perform HTTP GET request.

        Args:
            url: API endpoint path.
            params: Optional query parameters.
            requires_auth: Whether authentication is required.

        Returns:
            HTTP response object.

        Raises:
            NotAuthenticatedError: For 401 responses.
            UnauthorizedError: For 403 responses.
            NotFoundError: For 404 responses.
            JuaError: For other non-2xx responses.
        """
        headers = self._get_headers(requires_auth)
        response = requests.get(self._get_url(url), headers=headers, params=params)
        self._validate_response_status(response)
        return response

    def post(
        self,
        url: str,
        data: dict | None = None,
        query_params: dict | None = None,
        requires_auth: bool = True,
    ) -> requests.Response:
        """Perform HTTP POST request.

        Args:
            url: API endpoint path.
            data: Optional JSON payload.
            query_params: Optional query parameters.
            requires_auth: Whether authentication is required.

        Returns:
            HTTP response object.

        Raises:
            NotAuthenticatedError: For 401 responses.
            UnauthorizedError: For 403 responses.
            NotFoundError: For 404 responses.
            JuaError: For other non-2xx responses.
        """
        headers = self._get_headers(requires_auth)
        response = requests.post(
            self._get_url(url),
            headers=headers,
            json=data,
            params=query_params,
        )
        self._validate_response_status(response)
        return response

    def put(
        self, url: str, data: dict | None = None, requires_auth: bool = True
    ) -> requests.Response:
        """Perform HTTP PUT request.

        Args:
            url: API endpoint path.
            data: Optional JSON payload.
            requires_auth: Whether authentication is required.

        Returns:
            HTTP response object.

        Raises:
            NotAuthenticatedError: For 401 responses.
            UnauthorizedError: For 403 responses.
            NotFoundError: For 404 responses.
            JuaError: For other non-2xx responses.
        """
        headers = self._get_headers(requires_auth)
        response = requests.put(self._get_url(url), headers=headers, json=data)
        self._validate_response_status(response)
        return response

    def delete(self, url: str, requires_auth: bool = True) -> requests.Response:
        """Perform HTTP DELETE request.

        Args:
            url: API endpoint path.
            requires_auth: Whether authentication is required.

        Returns:
            HTTP response object.

        Raises:
            NotAuthenticatedError: For 401 responses.
            UnauthorizedError: For 403 responses.
            NotFoundError: For 404 responses.
            JuaError: For other non-2xx responses.
        """
        headers = self._get_headers(requires_auth)
        response = requests.delete(self._get_url(url), headers=headers)
        self._validate_response_status(response)
        return response

    def patch(
        self, url: str, data: dict | None = None, requires_auth: bool = True
    ) -> requests.Response:
        """Perform HTTP PATCH request.

        Args:
            url: API endpoint path.
            data: Optional JSON payload.
            requires_auth: Whether authentication is required.

        Returns:
            HTTP response object.

        Raises:
            NotAuthenticatedError: For 401 responses.
            UnauthorizedError: For 403 responses.
            NotFoundError: For 404 responses.
            JuaError: For other non-2xx responses.
        """
        headers = self._get_headers(requires_auth)
        response = requests.patch(self._get_url(url), headers=headers, json=data)
        self._validate_response_status(response)
        return response


def _get_user_agent() -> str:
    try:
        jua_version = importlib.metadata.version("jua") or "unknown"
    except importlib.metadata.PackageNotFoundError:
        jua_version = "unknown"
    return f"jua-python-sdk/{jua_version}"
