import requests  # type: ignore[import-untyped]

from jua.client import JuaClient
from jua.errors.api_errors import (
    NotAuthenticatedError,
    NotFoundError,
    UnauthorizedError,
)
from jua.errors.jua_error import JuaError


class API:
    def __init__(self, jua_client: JuaClient):
        self._jua_client = jua_client

    def _get_headers(self, requires_auth: bool = True) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if requires_auth:
            auth_settings = self._jua_client.settings.auth
            headers["X-API-Key"] = (
                f"{auth_settings.api_key_id}:{auth_settings.api_key_secret}"
            )
        return headers

    def _validate_response_status(self, response: requests.Response) -> None:
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
        return (
            f"{self._jua_client.settings.api_url}/"
            f"{self._jua_client.settings.api_version}/{url}"
        )

    def get(
        self, url: str, params: dict | None = None, requires_auth: bool = True
    ) -> requests.Response:
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
        headers = self._get_headers(requires_auth)
        response = requests.put(self._get_url(url), headers=headers, json=data)
        self._validate_response_status(response)
        return response

    def delete(self, url: str, requires_auth: bool = True) -> requests.Response:
        headers = self._get_headers(requires_auth)
        response = requests.delete(self._get_url(url), headers=headers)
        self._validate_response_status(response)
        return response

    def patch(
        self, url: str, data: dict | None = None, requires_auth: bool = True
    ) -> requests.Response:
        headers = self._get_headers(requires_auth)
        response = requests.patch(self._get_url(url), headers=headers, json=data)
        self._validate_response_status(response)
        return response
