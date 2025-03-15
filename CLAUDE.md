# CLAUDE.md - Siren Project Guidelines

## Build & Run Commands
- Install dependencies: `uv sync --frozen`
- Run server: `uv run fastapi run server.py`
- Run tests: `uv run pytest`
- Run single test: `uv run pytest test_server.py::test_function_name -v`
- Docker GPU: `docker compose up -d siren-gpu`
- Docker CPU: `docker compose up -d siren-cpu`

## Code Style
- **Format**: No specific formatter, follow existing 4-space indent style
- **Imports**: Group standard library, third-party, and local imports with blank lines between
- **Type Hints**: Use Python 3.11 syntax with Union types as `X | Y` instead of `Union[X, Y]`
- **Error Handling**: Use try/except blocks with specific exceptions and informative error messages
- **Logging**: Use the `logger` from the root module
- **Authentication**: Use Bearer token authentication with `verify_token` dependency
- **API Schema**: Follow OpenAI-compatible endpoint structure
- **Tests**: Use pytest fixtures and mocks, clearly name tests with pattern `test_function_what_condition`