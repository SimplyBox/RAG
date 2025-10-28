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
    gcc g++ libgomp1 poppler-utils ca-certificates git \
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

# 4) Install CLIP library from GitHub
RUN pip install git+https://github.com/openai/CLIP.git

RUN python -c "print('deps installed OK')"

# Prepare model dirs (Ubah sbert ke clip)
RUN mkdir -p /models/hf-cache /models/clip

# Copy the fetch script
COPY scripts/fetch_model.py /tmp/fetch_model.py

# Pin model (Ubah SBERT ke CLIP)
ARG CLIP_REV=main
ENV CLIP_REV=${CLIP_REV}
ENV CLIP_REPO="sentence-transformers/clip-ViT-B-32"
ENV CLIP_DEST="/models/clip/clip-ViT-B-32"

# Cache hub data between builds; verbose download; fail-fast if empty
RUN --mount=type=cache,id=s/35a544df-5187-48e2-9b81-6d9e5ad6e0e1-/root/.cache/huggingface,target=/root/.cache/huggingface \
    HUGGINGFACE_HUB_VERBOSITY=debug \
    python -u /tmp/fetch_model.py && \
    ls -lah /models/clip/clip-ViT-B-32 | sed -n '1,80p' # Cek path baru

# Copy source last (benefits from .dockerignore)
COPY . .

# ---------- Final stage ----------
FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME="/models/hf-cache" \
    HF_HUB_OFFLINE=1 \
    EMBED_MODEL_LOCAL_PATH="/models/clip/clip-ViT-B-32"

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 poppler-utils curl ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

# Bring venv, vendored model, and app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /models/clip /models/clip
COPY --from=builder /app /app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -fsS http://localhost:8000/health || exit 1

# 1 worker is safer on basic-xxs; bump later if you scale the plan
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000","--workers","1","--log-level","info"]