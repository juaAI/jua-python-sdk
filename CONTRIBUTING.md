# Contributing

Thank you for your interest in contributing to the MCP Python SDK! This document provides guidelines and instructions for contributing.

## Development Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Make sure you have Python 3.11+ installed
3. Fork the repository
4. Clone your fork: `git clone https://github.com/YOUR-USERNAME/jua-python-sdk.git`
5. Install dependencies:

```bash
uv sync --frozen --all-extras --dev
```

## Development Workflow

1. Create a new branch from `main`

2. Make your changes

3. Commit your changes

   - We are using [commitizen](https://commitizen-tools.github.io/commitizen/) / [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
   - You may use `cz commit` for an interactive way of creating the commit message

4. Ensure linting & unittests pass:

```bash
just check-commit
```

5. Submit a pull request to the same branch you branched from

## Code Style

- We use `ruff` for linting and formatting
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings for public classes & functions

## Pull Request Process

1. Update documentation as needed
2. Add tests for new functionality (use [pytest](https://docs.pytest.org/en/stable/getting-started.html))
3. Ensure CI passes
4. Maintainers will review your code
5. Address review feedback

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
