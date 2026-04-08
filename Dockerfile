FROM python:3.11-slim

RUN pip install uv

WORKDIR /app

# Copy dependency files AND README.md (required by hatchling)
COPY pyproject.toml uv.lock README.md ./

# Now run uv sync (README.md is present)
RUN uv sync --frozen --no-dev

# Copy the rest of the source code
COPY email_triage_env/ ./email_triage_env/
COPY server/ ./server/
COPY inference.py openenv.yaml ./

EXPOSE 7860

CMD ["uv", "run", "server"]