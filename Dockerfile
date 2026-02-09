ARG MODE=gpu

FROM debian:bookworm-slim AS base-cpu
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04 AS base-gpu

FROM base-${MODE} AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
  --mount=type=bind,source=uv.lock,target=uv.lock \
  --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
  --mount=type=bind,source=.python-version,target=.python-version \
  uv sync --frozen --no-install-project

ADD . /app

RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

ENTRYPOINT ["uvicorn", "siren:app", "--host", "0.0.0.0", "--port", "8000", "--ws", "wsproto"]
