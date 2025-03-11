FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ADD . /app
WORKDIR /app
RUN uv sync --frozen
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
ENTRYPOINT ["fastapi", "run", "server.py", "--host", "0.0.0.0", "--port", "8000"]
