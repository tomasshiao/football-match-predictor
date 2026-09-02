# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – builder
#
# Compiles all C-extension wheels (PyMC / PyTensor, XGBoost, netCDF4, …) in
# an isolated layer so the final image contains only the installed packages,
# not the build toolchain.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS builder

# Build dependencies needed to compile PyTensor (C backend), netCDF4 (HDF5),
# and other wheels that ship without a pre-built manylinux binary for 3.12.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        make \
        libhdf5-dev \
        libnetcdf-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Use a dedicated virtual environment so the final COPY is a clean directory
# with no system-site contamination.
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy only the dependency manifest first to exploit Docker layer caching:
# if pyproject.toml is unchanged, this expensive layer is not rebuilt.
WORKDIR /build
COPY pyproject.toml .

# Install *runtime* dependencies only — no dev or notebook extras. The
# notebook extra (ipykernel, ipywidgets, PyQt6 — a full Qt GUI toolkit) was
# previously installed here despite this comment saying otherwise; nothing
# in predictor/ or main.py imports any of it, and Qt in particular carries a
# long CVE history, so it's very likely a meaningful chunk of your
# container-scan findings. Dropping it also shrinks the image noticeably.
# --no-cache-dir keeps the image layer small.
RUN pip install --upgrade pip \
    && pip install --no-cache-dir "."

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS runtime

# Runtime-only native libraries required by the installed wheels:
#   libgomp1     – OpenMP (XGBoost parallel tree building)
#   libhdf5-103  – HDF5 shared library (netCDF4 / ArviZ .nc I/O)
#   libnetcdf19  – netCDF4 shared library
#   g++          – PyTensor (PyMC's backend) JIT-compiles small C extensions
#                  for graph execution at *sampling* time. Without a
#                  compiler present it silently falls back to a much slower
#                  pure-Python execution path for every op lacking one,
#                  which matters here since the default config runs 8 NUTS
#                  chains on every /predict call. g++ (not the full
#                  build-essential meta-package) is enough for PyTensor's
#                  C backend and keeps the added image size small.
#   gosu         – used by docker-entrypoint.sh to drop from root to the
#                  unprivileged `predictor` user after fixing bind-mount
#                  ownership (see that file). Purpose-built for exactly
#                  this: unlike `su`/`sudo`, it doesn't need a TTY and
#                  correctly forwards signals to the process it execs, so
#                  `docker stop` still reaches uvicorn directly instead of
#                  being swallowed by an intermediary shell.
# These are *not* dev headers; they are significantly smaller than -dev packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libhdf5-103 \
        libnetcdf19 \
        g++ \
        gosu \
    && rm -rf /var/lib/apt/lists/*

# Activate the virtual environment from the builder stage.
ENV VIRTUAL_ENV=/opt/venv
COPY --from=builder "$VIRTUAL_ENV" "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# ── Application ───────────────────────────────────────────────────────────────
WORKDIR /app

# Copy source tree.  Directories are ordered from least-changed to most-changed
# so that Docker can reuse cache layers for the stable parts of the codebase.
COPY predictor/ ./predictor/
COPY main.py .
COPY static/ ./static/

# Data directory: intentionally NOT copied in (see line below) — the app
# fetches results.csv / shootouts.csv over HTTPS on first run instead of
# using a baked-in cache, and the container needs outbound internet access
# for that. /app/data is created lazily by the app at runtime (see the
# chown below) or should be volume-mounted, per docker-compose.yml.
# To bake a local CSV cache into the image instead, uncomment the next line
# (and it must exist in the build context — an empty dir is fine).
# COPY data/ ./data/
# models/ is intentionally excluded here: mount it at runtime via a bind mount
# or a named volume so artefacts can be updated without rebuilding the image.
# To pre-bake models into the image, uncomment the line below:
# COPY models/ ./models/

# Non-root user: principle of least privilege.
# WORKDIR creates /app as root:root, mode 755. Pre-create every directory
# the app writes to at runtime — data/, models/, figures/, .pytensor_cache/,
# .cache/matplotlib/ — and chown the whole tree in one step. None of these
# exist yet otherwise: COPY data/ and COPY models/ are commented out above,
# and figures/ is normally created fresh per request with a dynamic path.
#
# This matters beyond just "the directory exists": data/, models/, and
# figures/ are named Docker volumes in docker-compose.yml (see that file's
# comments for why they're not bind mounts). Docker seeds a fresh named
# volume's ownership from whatever's already at that path in the image —
# with nothing here to seed from, a new volume defaults to root:root and
# `predictor` can't write to it, which is what previously produced
# "PermissionError: [Errno 13] ... '/app/data/results_cache.csv'". Same
# story for .pytensor_cache (also a named volume) and .cache/matplotlib
# (not a volume at all, just needs to exist before predictor's first write).
RUN addgroup --system predictor && adduser --system --ingroup predictor predictor \
    && mkdir -p /app/data /app/models /app/figures /app/.cache/matplotlib /app/.pytensor_cache \
    && chown -R predictor:predictor /app

# ── Runtime configuration ─────────────────────────────────────────────────────
# PyMC / PyTensor compilation cache: kept inside /app (rather than the
# default $HOME/.pytensor) so a read-only root filesystem can be used if
# required. Writable because of the chown above.
ENV PYTENSOR_FLAGS="base_compiledir=/app/.pytensor_cache"

# Force the non-interactive rendering backend. /predict now generates real
# chart images (score heatmap, outcome bars, etc.) on every request via
# predictor.evaluation.visualisation — without this, matplotlib may try to
# find a display in a headless container and fail or silently misbehave.
ENV MPLBACKEND=Agg

# The `predictor` user has no home directory (adduser --system defaults to
# /nonexistent), so matplotlib's default config-dir lookup fails and it
# falls back to a fresh /tmp directory on every restart — harmless, but
# noisy, and wasteful now that charts are a real per-request feature rather
# than an unused import. Writable because of the chown above.
ENV MPLCONFIGDIR=/app/.cache/matplotlib

# Suppress PyMC progress bars in a container log stream (tqdm writes ANSI).
ENV PYMC_PROGRESS_DISABLE=1

# Optuna verbosity (already set in code, but explicit here for robustness).
ENV OPTUNA_LOG_LEVEL=WARNING

# Expose the default uvicorn port.
EXPOSE 8000

# Health check: hits the FastAPI docs endpoint (always available if the app
# starts) every 30 s with a 10 s timeout and a 60 s start period to allow
# PyMC/PyTensor compilation on first startup.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/docs')"

# ── Startup command ───────────────────────────────────────────────────────────
# This is what docker-entrypoint.sh receives as "$@" and execs via gosu.
# --workers 1: the Bayesian NUTS sampler is already multi-threaded internally;
#   multiple workers would compete for CPU cores and shared model state.
# --timeout-keep-alive 75: slightly above the 60 s NUTS startup period.
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "75", \
     "--log-level", "info"]