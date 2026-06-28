import datetime
import numpy as np
import polars as pl

from predictor.config.pipeline_config import WeightingConfig

def compute_time_decay_weight(
    dates: pl.Series,
    reference_date: datetime.date,
    half_life_days: int,
) -> pl.Series:
    """Compute exponential time-decay weights for matches.
    
    Matches on or after the reference date receive weight 1.0. Matches
    before the reference date are exponentially downweighted with the given
    half-life. A match played exactly ``half_life_days`` before the
    reference date receives weight 0.5.
    
    **Mathematical form:**
    $$w_t = \exp\left( -\frac{\ln(2)}{T_{1/2}} \cdot (d_{\text{ref}} - d_t) \right)$$
    
    where $d_t$ is the match date, $d_{\text{ref}}$ is the reference date,
    $T_{1/2}$ is the half-life in days, and $w_t$ is the weight.
    
    Args:
        dates: Polars Date series of match dates.
        reference_date: Reference date (typically max training date in backtest,
            or today in production).
        half_life_days: Half-life in days; must be positive.
    
    Returns:
        Polars Float64 series of weights, same length as ``dates``.
        
    Raises:
        AssertionError: If ``half_life_days <= 0``.
    """
    assert half_life_days > 0, "half_life_days must be positive"
    
    # Subtracting pl.lit(date) from a pl.Series returns a Polars Expr rather
    # than a Series, so .dt.days() lands on ExprDateTimeNameSpace (no .days())
    _ref_epoch: int = (reference_date - datetime.date(1970, 1, 1)).days
    _date_epoch: np.ndarray = dates.to_physical().to_numpy()   # int32 epoch days
    _days_before = np.clip(_ref_epoch - _date_epoch, 0.0, None).astype(np.float64)

    # Exponential decay: w = exp(-ln(2) / T½ · Δdays)
    decay_factor = np.log(2) / half_life_days
    _weight_arr = np.exp(-decay_factor * _days_before)

    return pl.Series(values=_weight_arr, dtype=pl.Float64)

def compute_tournament_weight(
    tournaments: pl.Series,
    config: WeightingConfig,
) -> tuple[pl.Series, list[str]]:
    """Compute tournament-importance weights via exact-string lookup.
    
    Each tournament name is looked up in ``config.tournament_weight_table``.
    Unrecognised tournaments are assigned ``config.default_tournament_weight``
    and a warning is emitted **once per unique unrecognised tournament**,
    listing the name and the number of matches it affects.
    
    Args:
        tournaments: Polars Utf8 series of tournament names.
        config: WeightingConfig containing the lookup table and defaults.
    
    Returns:
        Tuple of (weights_series, warnings_list). ``weights_series`` is a
        Float64 series of the same length as ``tournaments``. ``warnings_list``
        is a list of warning messages (one per unrecognised tournament).
    
    Raises:
        AssertionError: If any required tournament key is missing from the
            config table (checked in WeightingConfig.__post_init__).
    """
    # Build lookup dict from config
    lookup = config.tournament_weight_table
    default_weight = config.default_tournament_weight
    
    # Identify unrecognised tournaments
    unique_tournaments = tournaments.unique().to_list()
    unrecognised = [t for t in unique_tournaments if t not in lookup]
    
    warnings: list[str] = []
    for tourney in unrecognised:
        count = (tournaments == tourney).sum()
        msg = (f"Unrecognised tournament '{tourney}' ({count} matches); "
               f"using default weight {default_weight}")
        warnings.append(msg)
    
    # Map each tournament to its weight
    def _get_weight(t: str) -> float:
        return lookup.get(t, default_weight)
    
    weights = tournaments.map_elements(
        _get_weight,
        return_dtype=pl.Float64,
    )
    
    return weights, warnings

def compute_match_weights(
    df: pl.DataFrame,
    reference_date: datetime.date,
    config: WeightingConfig,
) -> pl.DataFrame:
    """Attach time-decay, tournament-importance, and combined match weights.
    
    Computes three new columns:
    
    1. ``time_weight`` — exponential decay based on days before reference_date
    2. ``tournament_weight`` — exact-key lookup from tournament name
    3. ``match_weight`` — product of the above two, then mean-normalized to 1.0
    
    **Normalization:** ``match_weight`` is normalized so its mean equals 1.0,
    which prevents the loss scale from drifting as weighting schemes change.
    
    Args:
        df: Match DataFrame (typically ``TRAIN_DF`` or ``CORE_MATCH_DF``) with
            at minimum ``date`` and ``tournament`` columns.
        reference_date: Reference date for time-decay (typically max training
            date in backtest pass, or today in production).
        config: WeightingConfig with half-life and tournament table.
    
    Returns:
        DataFrame enriched with ``time_weight``, ``tournament_weight``, and
        ``match_weight`` columns. Original columns are preserved.
    
    Raises:
        AssertionError: (via compute_tournament_weight) if required
            tournaments are missing from the config table.
    """
    df = df.with_columns(
        time_weight=compute_time_decay_weight(
            df["date"], reference_date, config.half_life_days
        )
    )
    
    # Compute tournament weights and emit warnings
    tour_weights, warnings = compute_tournament_weight(
        df["tournament"], config
    )
    for warning_msg in warnings:
        print(f"  ⚠ {warning_msg}")
    
    df = df.with_columns(
        tournament_weight=tour_weights
    )
    
    # Combined weight (product), then normalize to mean 1.0
    df = df.with_columns(
        _combined=(pl.col("time_weight") * pl.col("tournament_weight"))
    )
    
    mean_combined = df["_combined"].mean()
    assert mean_combined > 0, "Mean combined weight is not positive; check inputs"
    
    df = df.with_columns(
        match_weight=(pl.col("_combined") / mean_combined)
    ).drop("_combined")
    
    return df