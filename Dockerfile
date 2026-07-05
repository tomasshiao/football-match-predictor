# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – builder
#
# Compiles all C-extension wheels (PyMC / PyTensor, XGBoost, netCDF4, …) in
# an isolated layer so the final image contains only the installed packages,
# not the build toolchain.
#
# Base image is pinned to the "-bookworm" suffix, not just "python:3.12-slim".
# The unsuffixed "slim" tag floats to whatever Debian release is currently
# "stable" and moved from bookworm to trixie, which renamed the versioned
# runtime packages below (libhdf5-103 -> libhdf5-310, libnetcdf19 ->
# libnetcdf22) and broke this build. Pinning the Debian release stops that
# drift; bump it deliberately (and update the package names in the runtime
# stage to match) when you're ready to move to a newer Debian release rather
# than having it happen silently on a routine rebuild.
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
# These are *not* dev headers; they are significantly smaller than -dev packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libhdf5-103 \
        libnetcdf19 \
        g++ \
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
# WORKDIR creates /app as root:root, mode 755. The app writes at runtime to
# several subdirectories under /app — data/, models/, figures/<date>/<match>/,
# .pytensor_cache/ — some of which don't exist at build time at all now that
# COPY data/ is commented out above, and figures/ is created fresh per
# request with a dynamic path. Chown the whole tree rather than trying to
# enumerate every writable subdirectory individually.
RUN addgroup --system predictor && adduser --system --ingroup predictor predictor \
    && chown -R predictor:predictor /app
USER predictor

# ── Runtime configuration ─────────────────────────────────────────────────────
# PyMC / PyTensor compilation cache: kept inside /app (rather than the
# default $HOME/.pytensor) so a read-only root filesystem can be used if
# required. Writable because of the chown above.
ENV PYTENSOR_FLAGS="base_compiledir=/app/.pytensor_cache"

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

# ── Entrypoint ────────────────────────────────────────────────────────────────
# --workers 1: the Bayesian NUTS sampler is already multi-threaded internally;
#   multiple workers would compete for CPU cores and shared model state.
# --timeout-keep-alive 75: slightly above the 60 s NUTS startup period.
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "75", \
     "--log-level", "info"]