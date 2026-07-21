# Track 1 - Hybrid Token-Efficient Routing Agent
# Judging VM is linux/amd64. Build for that platform explicitly (see README).
#
# The local model is Gemma 4 E4B, whose GGUF declares architecture "gemma4"
# (launched Apr 2026). That is NOT supported by older llama-cpp-python builds,
# which is the usual cause of a "pulled but crashed" RUNTIME_ERROR. To avoid
# that entirely we build llama.cpp's `llama-server` from recent source and call
# it over HTTP, so gemma4 support depends only on the llama.cpp build here.

# ---------- Stage 1: build llama-server from source ----------
FROM python:3.11-slim AS llama-builder

# Pin/override with:  --build-arg LLAMA_CPP_REF=b8900
ARG LLAMA_CPP_REF=master

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch "${LLAMA_CPP_REF}" \
        https://github.com/ggml-org/llama.cpp.git . \
    || git clone https://github.com/ggml-org/llama.cpp.git . \
       && git checkout "${LLAMA_CPP_REF}" || true

# Portable CPU build (GGML_NATIVE=OFF => runs on any amd64 host). CURL off to
# avoid a runtime libcurl dependency. We do NOT disable examples/tools, since on
# some llama.cpp versions the llama-server target lives behind those options.
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DLLAMA_CURL=OFF \
    && cmake --build build --target llama-server -j "$(nproc)" \
    && mkdir -p /opt/llama/bin /opt/llama/lib \
    && (cmake --install build --prefix /opt/llama 2>/dev/null || true) \
    && (find build -name 'llama-server' -type f -exec cp {} /opt/llama/bin/ \; ) \
    && (find build -name '*.so*' -exec cp {} /opt/llama/lib/ \; 2>/dev/null || true) \
    && strip --strip-unneeded /opt/llama/bin/* /opt/llama/lib/* 2>/dev/null || true \
    && test -x /opt/llama/bin/llama-server

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/models \
    PATH=/opt/llama/bin:$PATH \
    LD_LIBRARY_PATH=/opt/llama/lib:/opt/llama/lib64

# OpenMP runtime is required by the llama.cpp CPU backend.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Bring in the compiled server + shared libs.
COPY --from=llama-builder /opt/llama /opt/llama

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache

# Bundle the model. Copy your chosen GGUF into ./models before building.
COPY models/ /app/models/
COPY src/ /app/src/

# Fail fast at build time if no GGUF was bundled, and confirm the server exists.
RUN test -n "$(ls -A /app/models/*.gguf 2>/dev/null)" \
        || (echo 'ERROR: no *.gguf in ./models - copy your Gemma E4B GGUF there before building' && exit 1) \
    && (command -v llama-server >/dev/null || (echo 'ERROR: llama-server not built' && exit 1)) \
    && find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

ENV FIREWORKS_MODE=escalate \
    CONFIDENCE_THRESHOLD=0.45 \
    LLAMA_CTX=2048 \
    N_GPU_LAYERS=999 \
    INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json

ENTRYPOINT ["python", "-m", "src.main"]
