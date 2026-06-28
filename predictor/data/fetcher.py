import io
import pathlib
import polars as pl

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
        import urllib.request
        print(f"  ↳ Downloading from: {source_url}")
        try:
            with urllib.request.urlopen(source_url, timeout=60) as resp:
                raw_bytes = resp.read()
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
        import urllib.request
        print(f"  ↳ Downloading shootout from: {source_url}")
        try:
            with urllib.request.urlopen(source_url, timeout=60) as resp:
                raw_bytes = resp.read()
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
