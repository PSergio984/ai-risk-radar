# ==============================================================================
# AI Investment Risk Radar: Minimal Production Containerfile
#
# Features:
#   1. Python 3.12 slim Debian Bookworm base image.
#   2. Non-root user execution (appuser, UID 1000) for security hardening.
#   3. Writable mount targets for /app/data, /app/artifacts, and /app/logs.
#   4. Multi-mode entrypoint: dashboard (Streamlit on 8501), latest, test, research.
# ==============================================================================

FROM python:3.12-slim-bookworm AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

WORKDIR /app

# Install minimal system dependencies required for compilation and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user and application directories
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -m -s /bin/bash appuser && \
    mkdir -p /app/data /app/artifacts /app/artifacts/models /app/logs && \
    chown -R appuser:appgroup /app

# Install Python package dependencies
COPY requirements.txt pyproject.toml /app/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application source code, evaluation artifacts, and test suite
COPY ai_risk_radar/ /app/ai_risk_radar/
COPY artifacts/ /app/artifacts/
COPY tests/ /app/tests/
COPY app.py config.json /app/
COPY scripts/ /app/scripts/
COPY docker/entrypoint.sh /app/docker/entrypoint.sh

# Install local package in editable/module mode and ensure appuser ownership
RUN pip install --no-deps -e . && \
    chown -R appuser:appgroup /app && \
    chmod +x /app/docker/entrypoint.sh /app/scripts/*.sh

# Switch to unprivileged runtime user
USER appuser

# Expose Streamlit dashboard default port
EXPOSE 8501

# Declare volume mount targets for data and artifact persistence across container runs
VOLUME ["/app/data", "/app/artifacts", "/app/logs"]

# Healthcheck for Streamlit dashboard
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["dashboard"]
