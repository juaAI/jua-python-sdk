import argparse
import asyncio
import json
from datetime import datetime

from aiohttp import web

from jua.client import JuaClient
from jua.settings.authentication import ApiKey, AuthenticationSettings
from jua.settings.jua_settings import JuaSettings

_AUTH_FRONTEND_ENDPOINT = "cli-auth"
_LOCAL_URL = "localhost"
_LOCAL_PORT = 8080
_LOCAL_ENDPOINT = "auth"


def get_auth_parser(subparsers: argparse._SubParsersAction):
    """Set up the auth command parser"""
    parser = subparsers.add_parser("auth", help="Authentication commands")
    parser.add_argument(
        "--environment", type=str, default="default", help="Environment to use"
    )
    parser.add_argument(
        "--api-key-path", type=str, default=None, help="Path to the API key file"
    )
    return parser


def _get_local_auth_url() -> str:
    return rf"http://{_LOCAL_URL}:{_LOCAL_PORT}/{_LOCAL_ENDPOINT}"


def _get_frontend_url(client: JuaClient) -> str:
    return f"{client.settings.frontend_url}/{_AUTH_FRONTEND_ENDPOINT}"


class AuthResponseServer:
    def __init__(self):
        self.app = None
        self.runner = None
        self.site = None
        self.credentials_event = asyncio.Event()
        self.api_key: ApiKey | None = None

    async def __aenter__(self):
        self.app = web.Application()
        self.app.router.add_get(f"/{_LOCAL_ENDPOINT}", self.handle_auth)
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, _LOCAL_URL, _LOCAL_PORT)
        await self.site.start()
        print(f"Server started at {_get_local_auth_url()}")
        print("Waiting for authentication...")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        # shutdown the server
        if not self.runner:
            return
        await self.runner.cleanup()

    def stop_event(self):
        self.credentials_event.set()

    def _generate_html_response(self, title, message, is_success=True):
        """Generate a styled HTML response for the authentication page.

        Args:
            title: Title text to display
            message: Main message content
            is_success: Whether this is a success or error message

        Returns:
            HTML content as a string
        """
        # Green for success, red for error
        status_color = "#10B981" if is_success else "#EF4444"
        icon = "✓" if is_success else "✗"

        footer_text = "" if is_success else "Please try again"

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Jua Authentication</title>
            <style>
                body {{
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                        'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    background-color: #f9fafb;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    color: #374151;
                }}
                .container {{
                    background-color: white;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05),
                               0 10px 15px rgba(0, 0, 0, 0.03);
                    padding: 40px;
                    text-align: center;
                    max-width: 500px;
                    width: 90%;
                }}
                .logo {{
                    font-size: 32px;
                    font-weight: 800;
                    color: #0F172A;
                    margin-bottom: 20px;
                }}
                .status-icon {{
                    background-color: {status_color};
                    color: white;
                    width: 60px;
                    height: 60px;
                    border-radius: 50%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    margin: 0 auto 24px auto;
                    font-size: 30px;
                }}
                h1 {{
                    color: #0F172A;
                    font-size: 24px;
                    font-weight: 700;
                    margin-bottom: 12px;
                }}
                p {{
                    color: #6B7280;
                    font-size: 16px;
                    line-height: 1.5;
                    margin-bottom: 24px;
                }}
                .highlight {{
                    color: {status_color};
                    font-weight: 600;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="logo">Jua</div>
                <div class="status-icon">{icon}</div>
                <h1>{title}</h1>
                <p>{message}</p>
                <p class="highlight">{footer_text}</p>
            </div>
        </body>
        </html>
        """

    async def handle_auth(self, request: web.Request):
        try:
            self.api_key = ApiKey(**request.query)
        except Exception as e:
            print(f"Error parsing API key: {e}")
            self.stop_event()

            html_content = self._generate_html_response(
                "Authentication Failed",
                f"There was an error processing your API key: {e}",
                is_success=False,
            )

            return web.Response(
                status=400,
                text=html_content,
                content_type="text/html",
            )

        self.stop_event()

        html_content = self._generate_html_response(
            "Authentication Successful!",
            "Your Jua API key has been successfully created and saved. "
            "You can now close this window and return to the command line.",
        )

        return web.Response(
            text=html_content,
            content_type="text/html",
        )

    async def wait_for_credentials(self, timeout: float = 300) -> ApiKey:
        """
        Wait for credentials to be received.

        Args:
            timeout: Maximum time to wait in seconds (default: 5 minutes)

        Returns:
            ApiKey object

        Raises:
            TimeoutError: If credentials are not received within the timeout period
        """
        try:
            await asyncio.wait_for(self.credentials_event.wait(), timeout)
            if self.api_key:
                return self.api_key
            raise ValueError("Credentials received but incomplete")
        except asyncio.TimeoutError:
            print("Timed out waiting for authentication")
            raise TimeoutError("Authentication timed out")


def launch_frontend(client: JuaClient):
    """Launch the authentication frontend"""
    import webbrowser

    now = datetime.now().isoformat()
    key_name = f"jua-cli-{now}"

    callback_url = _get_local_auth_url()
    full_url = (
        f"{_get_frontend_url(client)}?callback={callback_url}&api_key_name={key_name}"
    )
    print(f"Opening browser: {full_url}")
    webbrowser.open(full_url)


async def main_async(args=None):
    """
    Handle authentication with Jua API

    This function guides the user through the authentication process
    by helping them set up their API key.
    """
    print("Jua API Authentication")
    print("======================")
    print("Please complete authentication in your browser.")

    jua_settings = JuaSettings(
        auth=AuthenticationSettings(
            environment=args.environment, api_key_path=args.api_key_path
        )
    )
    client = JuaClient(jua_settings)
    launch_frontend(client)
    async with AuthResponseServer() as auth_server:
        try:
            api_key = await auth_server.wait_for_credentials()
            print("\nAuthentication successful!")

            auth_settings = jua_settings.auth
            secrets_path = auth_settings.secrets_file_path
            secrets_path.parent.mkdir(parents=True, exist_ok=True)
            secrets_path.write_text(json.dumps(api_key.model_dump(), indent=4))

            return None
        except (TimeoutError, ValueError) as e:
            print(f"Authentication failed: {e}")
            return None


def main(args=None):
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jua API Authentication")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    get_auth_parser(subparsers)
    args = parser.parse_args()

    main(args)
