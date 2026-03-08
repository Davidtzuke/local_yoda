FROM python:3.12-slim AS base

WORKDIR /app

# System deps for chromadb and other native packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || pip install --no-cache-dir .

# Copy source
COPY yoda/ yoda/
COPY tests/ tests/

# Data directory
RUN mkdir -p /root/.yoda

# Default: run the CLI
ENTRYPOINT ["python", "-m", "yoda.cli"]

# MCP server mode
# docker run yoda --mcp --port 8765
EXPOSE 8765
