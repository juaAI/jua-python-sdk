import argparse


def get_auth_parser(subparsers: argparse._SubParsersAction):
    """Set up the auth command parser"""
    parser = subparsers.add_parser("auth", help="Authentication commands")
    return parser


def main(args=None):
    """
    Handle authentication with Jua API

    This function guides the user through the authentication process
    by helping them set up their API key.
    """
    print("Jua API Authentication")
    print("======================")
    print("Comming soon...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jua API Authentication")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    get_auth_parser(subparsers)
    args = parser.parse_args()
    main(args)
