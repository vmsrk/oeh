FROM python:3.11-slim

RUN pip install uv

WORKDIR /app

# Copy all source code first (so hatchling can see the packages)
COPY clinical_triage_env/ ./clinical_triage_env/
COPY server/ ./server/
COPY pyproject.toml uv.lock README.md ./

# Now install dependencies and build the package
RUN uv sync --frozen --no-dev

# Copy remaining files
COPY inference.py openenv.yaml ./

EXPOSE 7860
CMD ["uv", "run", "server"]