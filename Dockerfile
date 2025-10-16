# Build stage
# FROM python:3.11-slim AS builder
# WORKDIR /app
# COPY requirements.txt .
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gcc \
#     g++ \
#     libgomp1 \
#     poppler-utils \
#     && rm -rf /var/lib/apt/lists/*
# RUN pip install --no-cache-dir -r requirements.txt

# Final stage
# FROM python:3.11-slim
# WORKDIR /app
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libgomp1 \
#     poppler-utils \
#     curl \
#     && rm -rf /var/lib/apt/lists/*
# COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
# COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
# COPY . .
# EXPOSE 8000
# HEALTHCHECK --interval=30s --timeout=3s \
#     CMD curl -f http://localhost:8000/health || exit 1
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "info"]


# ---------- Build stage ----------
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Minimal build tools (only if needed for wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 poppler-utils \
 && rm -rf /var/lib/apt/lists/*

# Create venv and install deps in layers to leverage caching
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

# 1) Base first
COPY requirements.base.txt .
RUN pip install -r requirements.base.txt

# 2) CPU torch
COPY requirements.torch.txt .
RUN pip install -r requirements.torch.txt

# 3) ML stack
COPY requirements.ml.txt .
RUN pip install -r requirements.ml.txt

# Copy source last (benefits from .dockerignore)
COPY . .

# ---------- Final stage ----------
FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # optional: keep HF cache small / inside container
    HF_HOME="/tmp/hf"

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 poppler-utils curl \
 && rm -rf /var/lib/apt/lists/*

# Bring in venv & app
COPY --from=builder /opt/venv /opt/venv
COPY . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -fsS http://localhost:${PORT:-8000}/health || exit 1

# 1 worker is safer on basic-xxs; bump later if you scale the plan
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","${PORT}","--workers","2","--log-level","info"]
