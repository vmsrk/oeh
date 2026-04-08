FROM python:3.11-slim

RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

# Debug: list files to confirm README.md is present
RUN ls -la

RUN uv sync --frozen --no-dev

COPY email_triage_env/ ./email_triage_env/
COPY server/ ./server/
COPY inference.py openenv.yaml ./

EXPOSE 7860

CMD ["uv", "run", "server"]