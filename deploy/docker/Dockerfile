# =============================================================================
# CARBON-AWARE CONTROLLER — DOCKERFILE
# =============================================================================
# Multi-stage build for the Kubernetes Pod controller.
# Produces a minimal, secure, production-ready image.
#
# Build (CPU):
#   docker build -f deploy/docker/Dockerfile -t carbon-aware-controller:latest .
#
# Build (CUDA 12.1):
#   docker build -f deploy/docker/Dockerfile \
#       --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
#       -t carbon-aware-controller:latest-cuda .
#
# Run:
#   docker run --rm -it \
#       --env-file .env \
#       -v $(pwd)/watttime_cache:/app/watttime_cache \
#       -v $(pwd)/checkpoints:/app/checkpoints:ro \
#       carbon-aware-controller:latest
#
# Author: Carbon-Aware Deep Reinforcement Learning Platform
# =============================================================================


# -----------------------------------------------------------------------------
# STAGE 1: Base with Python
# -----------------------------------------------------------------------------
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

# Build arguments
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG APP_VERSION=1.0.0
ARG BUILD_DATE
ARG VCS_REF

# OCI image labels (https://github.com/opencontainers/image-spec)
LABEL org.opencontainers.image.title="Carbon-Aware Controller"
LABEL org.opencontainers.image.description="Kubernetes Pod controller for carbon-aware scheduling"
LABEL org.opencontainers.image.vendor="Carbon-Aware Deep Reinforcement Learning Platform"
LABEL org.opencontainers.image.version="${APP_VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.source="https://github.com/your-org/carbon-aware-deep-reinforcement-learning-platform"
LABEL org.opencontainers.image.licenses="MIT"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_HOME=/app \
    PYTHONPATH=/app

# Install system dependencies (minimal set)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r -g 1000 carbon-aware \
    && useradd -r -u 1000 -g carbon-aware -d /app -s /bin/false carbon-aware

# Set working directory
WORKDIR ${APP_HOME}


# -----------------------------------------------------------------------------
# STAGE 2: Dependencies
# -----------------------------------------------------------------------------
FROM base AS dependencies

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
# PyTorch is installed separately with the specified index URL
RUN pip install --no-cache-dir \
        torch --index-url ${PYTORCH_INDEX_URL} \
    && pip install --no-cache-dir -r requirements.txt


# -----------------------------------------------------------------------------
# STAGE 3: Application
# -----------------------------------------------------------------------------
FROM dependencies AS application

# Copy source code
COPY src/ ./src/

# Copy configuration files
COPY config/ ./config/

# Copy entrypoint script
COPY deploy/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Create required directories with correct permissions
RUN mkdir -p /app/watttime_cache \
             /app/checkpoints \
             /app/logs \
             /app/outputs \
    && chown -R carbon-aware:carbon-aware /app

# Switch to non-root user
USER carbon-aware

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8000 8080

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["tini", "--"]

# Default command: run the controller
CMD ["/usr/local/bin/entrypoint.sh", "controller"]