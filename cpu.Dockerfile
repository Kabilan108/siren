FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ADD . /app
WORKDIR /app
RUN uv sync --frozen
ENV PATH="/app/.venv/bin:$PATH"
RUN fastapi run server.py --host 0.0.0.0 --port 8000
