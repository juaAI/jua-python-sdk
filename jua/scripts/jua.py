import argparse
import sys

from . import auth


def main():
    parser = argparse.ArgumentParser(
        description="Jua Python SDK command line interface"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Add command parsers
    auth.get_auth_parser(subparsers)

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    command = args.command
    if command == "auth":
        auth.main(args)
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
