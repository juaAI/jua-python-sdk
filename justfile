default:
    just --list

lint:
    uv run pre-commit run --all

build:
    uv build

test:
    uv run pytest

check-commit: lint test

push-to-pypi:
    uv publish

clean-before-publish:
    rm -rf dist

publish: lint test clean-before-publish build push-to-pypi
