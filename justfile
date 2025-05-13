default:
    just --list

lint:
    uv run pre-commit run --all

build:
    uv build

test:
    uv run pytest

check-commit: lint test
