# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – builder
#
# Compiles all C-extension wheels (PyMC / PyTensor, XGBoost, netCDF4, …) in
# an isolated layer so the final image contains only the installed packages,
# not the build toolchain.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

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

# Install *runtime* extras only (no dev / notebook groups).
# --no-cache-dir keeps the image layer small.
RUN pip install --upgrade pip \
    && pip install --no-cache-dir ".[notebook]"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Runtime-only native libraries required by the installed wheels:
#   libgomp1     – OpenMP (XGBoost parallel tree building)
#   libhdf5-103  – HDF5 shared library (netCDF4 / ArviZ .nc I/O)
#   libnetcdf19  – netCDF4 shared library
# These are *not* dev headers; they are significantly smaller than -dev packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libhdf5-103 \
        libnetcdf19 \
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

# Data directories: pre-populated CSV caches are baked in so the service can
# start without a network round-trip.  The models/ directory is expected to
# contain serialised production artefacts (*.joblib, *.nc).  Both are mounted
# as volumes in production so updates don't require a rebuild.
COPY data/ ./data/
# models/ is intentionally excluded here: mount it at runtime via a bind mount
# or a named volume so artefacts can be updated without rebuilding the image.
# To pre-bake models into the image, uncomment the line below:
# COPY models/ ./models/

# Non-root user: principle of least privilege.
RUN addgroup --system predictor && adduser --system --ingroup predictor predictor
USER predictor

# ── Runtime configuration ─────────────────────────────────────────────────────
# PyMC / PyTensor compilation cache: writing to $HOME/.pytensor is safe for
# the non-root user.  Set PYTENSOR_FLAGS to keep the cache inside /app so
# that a read-only filesystem can be used if required.
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