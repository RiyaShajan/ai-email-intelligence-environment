# ── AI Email Intelligence Environment — Dockerfile ────────────────────────
# Multi-stage build: keeps the final image lean.

# ── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (cache layer)
COPY requirements.txt .

# Install dependencies into a prefix directory
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Metadata
LABEL maintainer="AI Email Intelligence Team"
LABEL description="OpenEnv-compliant AI Email Intelligence Environment"
LABEL version="1.0.0"

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy project source
COPY . .

# Install the package in editable mode so imports work
RUN pip install --no-cache-dir -e .

# Set environment defaults
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# The HF_TOKEN must be provided at runtime via -e flag
# Example: docker run --rm -e HF_TOKEN=sk-xxx ai-email-env
ENV HF_TOKEN=""

# Default command: run the baseline inference
CMD ["python", "email_env/inference.py"]

# ── Alternative commands ───────────────────────────────────────────────────
# Run demo:       docker run --rm ai-email-env python email_env/main.py
# Run tests:      docker run --rm ai-email-env pytest tests/
# Open shell:     docker run --rm -it ai-email-env bash
