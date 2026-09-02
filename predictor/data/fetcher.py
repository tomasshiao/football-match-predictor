import io
import pathlib
import time
import urllib.request

import polars as pl

# Retries + linear backoff for transient failures fetching the source CSVs
# (DNS blips, brief GitHub raw-content-CDN hiccups, etc.) — see
# _fetch_url_bytes for the rationale.
_MAX_FETCH_ATTEMPTS: int = 3
_FETCH_RETRY_BACKOFF_SECONDS: float = 2.0


def _fetch_url_bytes(source_url: str, timeout: int = 60) -> bytes:
    """Download a URL's raw bytes, retrying on transient network failures.

    Observed in practice: two back-to-back requests to the same hostname
    (``raw.githubusercontent.com``) from inside a container — one for
    ``results.csv``, immediately followed by one for ``shootouts.csv`` —
    where the first succeeds and the second fails with a DNS resolution
    error (``socket.gaierror``). That pattern indicates a one-off blip
    (in the container's resolver, the CDN, or the network path) rather
    than a systemic outage, since a genuinely broken resolver would fail
    both requests identically. A short retry with linear backoff turns
    that class of failure into a slower but successful fetch, without
    masking a truly persistent problem — this still raises after
    ``_MAX_FETCH_ATTEMPTS`` attempts, same as a single unretried failure
    would have.

    Args:
        source_url: URL to fetch.
        timeout: Per-attempt socket timeout, in seconds.

    Returns:
        The response body as raw bytes.

    Raises:
        Exception: Whatever the final attempt raised (e.g.
            ``urllib.error.URLError``), re-raised as-is so callers can wrap
            it with their own context.
    """
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_FETCH_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(source_url, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < _MAX_FETCH_ATTEMPTS:
                wait_s = _FETCH_RETRY_BACKOFF_SECONDS * attempt
                print(
                    f"  ↳ Attempt {attempt}/{_MAX_FETCH_ATTEMPTS} failed "
                    f"({exc}); retrying in {wait_s:.0f}s …"
                )
                time.sleep(wait_s)
    assert last_exc is not None  # the loop always sets this before exiting
    raise last_exc

def fetch_results_csv(
    source_url: str,
    cache_path: pathlib.Path | None,
) -> pl.DataFrame:
    """Fetch the international results CSV and return a typed Polars DataFrame.

    If ``cache_path`` is provided and the file already exists on disk, it
    is read from disk.  Otherwise the file is downloaded from
    ``source_url`` and, if ``cache_path`` is set, written to disk for
    future runs.

    The ``date`` column is parsed directly to ``pl.Date`` during
    ingestion.  Missing scores (unplayed fixtures) are represented as
    ``null`` via ``null_values=['NA']``.

    Args:
        source_url: Direct URL to the raw ``results.csv`` file.
        cache_path: Optional local path for caching.

    Returns:
        Typed Polars DataFrame with columns matching ``EXPECTED_COLUMNS``.

    Raises:
        RuntimeError: If the download fails or the resulting frame is empty.
    """
    if cache_path is not None and cache_path.exists():
        print(f"  ↳ Loading from cache: {cache_path}")
        raw_bytes = cache_path.read_bytes()
    else:
        print(f"  ↳ Downloading from: {source_url}")
        try:
            raw_bytes = _fetch_url_bytes(source_url)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to download results CSV: {exc}") from exc

        if cache_path is not None:
            cache_path.write_bytes(raw_bytes)
            print(f"  ↳ Saved to cache: {cache_path}")

    df = pl.read_csv(
        io.BytesIO(raw_bytes),
        null_values=["NA"],
        schema_overrides={
            "home_score": pl.Int64,
            "away_score": pl.Int64,
            "neutral":    pl.Boolean,
        },
        try_parse_dates=False,
    )

    # Parse date column explicitly to pl.Date
    df = df.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))

    if df.is_empty():
        raise RuntimeError("Fetched CSV resulted in an empty DataFrame.")

    return df

def fetch_shootout_csv(
    source_url: str,
    cache_path: pathlib.Path | None,
) -> pl.DataFrame:
    """Fetch the penalty-shootout results CSV.

    The file contains one row per match that was decided by a penalty
    shootout.  Columns: date (pl.Date), home_team (pl.Utf8),
    away_team (pl.Utf8), winner (pl.Utf8 — the winning team's name).

    Args:
        source_url: Direct URL to ``shootouts.csv``.
        cache_path: Optional local cache path (sibling of results_cache).

    Returns:
        Polars DataFrame with columns ['date', 'home_team', 'away_team', 'winner'].

    Raises:
        RuntimeError: If the download fails or the file is empty.
    """
    _shootout_cache = (
        cache_path.parent / "shootout_cache.csv"
        if cache_path is not None else None
    )
    if _shootout_cache is not None and _shootout_cache.exists():
        print(f"  ↳ Loading shootout from cache: {_shootout_cache}")
        raw_bytes = _shootout_cache.read_bytes()
    else:
        print(f"  ↳ Downloading shootout from: {source_url}")
        try:
            raw_bytes = _fetch_url_bytes(source_url)
        except Exception as exc:
            raise RuntimeError(f"Failed to download shootout CSV: {exc}") from exc
        if _shootout_cache is not None:
            _shootout_cache.write_bytes(raw_bytes)
            print(f"  ↳ Saved shootout to cache: {_shootout_cache}")

    df = pl.read_csv(
        io.BytesIO(raw_bytes),
        try_parse_dates=False,
    ).with_columns(pl.col("date").str.to_date("%Y-%m-%d"))

    _expected_cols = {"date", "home_team", "away_team", "winner"}
    _missing = _expected_cols - set(df.columns)
    if _missing:
        raise RuntimeError(
            f"shootouts.csv is missing expected columns: {_missing}"
        )
    if df.is_empty():
        raise RuntimeError("Fetched shootout CSV resulted in an empty DataFrame.")
    return df
