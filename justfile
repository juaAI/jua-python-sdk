default:
    just --list

lint:
    uv run pre-commit run --all

build:
    uv build

test:
    uv run pytest

test-ci:
    uv run pytest --junitxml=pytest-report.xml

check-commit: lint test

push-to-pypi:
    uv publish

publish-ci:
    rm -rf dist
    uv build
    uv publish

clean-before-publish:
    rm -rf dist

publish: lint test clean-before-publish build push-to-pypi

docs-clean:
    rm -rf docs/build

docs-md:
    uv run sphinx-build -M markdown docs/source docs/build

docs-html:
    uv run sphinx-build -M html docs/source docs/build

docs: docs-clean docs-md docs-html
