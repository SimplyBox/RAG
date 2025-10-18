# ---------- Build stage ----------
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Optional cache-buster to invalidate flaky remote cached layers
ARG CACHE_BUSTER=0
RUN echo "cache-buster=${CACHE_BUSTER}"

# Minimal build tools (only if needed for wheels)
ARG APT_SIG=0
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 poppler-utils ca-certificates \
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

RUN python -c "print('deps installed OK')"
# Prepare model dirs
RUN mkdir -p /models/hf-cache /models/sbert

# Copy the fetch script (no heredocs; safe on DO builder)
COPY scripts/fetch_model.py /tmp/fetch_model.py

# Optional: pin a specific commit for deterministic builds (set via build arg)
ARG SBERT_REV=main
ENV SBERT_REV=${SBERT_REV}

# Cache hub data between builds; verbose download; fail-fast if empty
RUN --mount=type=cache,target=/root/.cache/huggingface \
    HUGGINGFACE_HUB_VERBOSITY=debug \
    python -u /tmp/fetch_model.py && \
    ls -lah /models/sbert/all-MiniLM-L6-v2 | sed -n '1,80p'

# Copy source last (benefits from .dockerignore)
COPY . .

# ---------- Final stage ----------
FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # optional: keep HF cache small / inside container
    HF_HOME="/models/hf-cache" \
    # force offline HF so no network calls at runtime
    HF_HUB_OFFLINE=1 \
    # app will read this to load embeddings by PATH
    EMBED_MODEL_LOCAL_PATH="/models/sbert/all-MiniLM-L6-v2"

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 poppler-utils curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Bring venv, vendored model, and app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /models/sbert /models/sbert
COPY --from=builder /app /app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -fsS http://localhost:8000/health || exit 1

# 1 worker is safer on basic-xxs; bump later if you scale the plan
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000","--workers","1","--log-level","info"]
