import polars as pl

from predictor.config.pipeline_config import PipelineConfig
from predictor.constants.constants import SEP
from predictor.models.dixon_coles import DixonColesRatings

def _validate_feature_inputs(
    core_match_df: pl.DataFrame,
    train_weighted_df: pl.DataFrame,
    dc_ratings: DixonColesRatings,
    cfg: PipelineConfig,
    home_team: str,
    away_team: str,
) -> None:
    """Pre-implementation guard: verify all feature inputs are leak-free."""
    print("── Validation Phase ─────────────────────────────────────────────")

    # 1. CORE_MATCH_DF must have no null scores (only played matches)
    _null_scores = core_match_df.filter(
        pl.col("home_score").is_null() | pl.col("away_score").is_null()
    ).height
    assert _null_scores == 0, (
        f"CORE_MATCH_DF has {_null_scores} rows with null scores. "
        "Feature engineering must operate on played matches only."
    )
    print("  ✓ CORE_MATCH_DF: no null scores")

    # 2. CORE_MATCH_DF must be sorted chronologically (rolling requires this)
    _dates = core_match_df["date"].to_list()
    assert _dates == sorted(_dates), (
        "CORE_MATCH_DF is not sorted chronologically. "
        "Call .sort('date') before feature engineering."
    )
    print("  ✓ CORE_MATCH_DF: chronological order verified")

    # 3. match_id must be present (pivot/join anchor for rolling form)
    assert "match_id" in core_match_df.columns, (
        "CORE_MATCH_DF is missing 'match_id'. "
        "Call filter_from_date() before feature engineering."
    )
    print("  ✓ CORE_MATCH_DF: 'match_id' column present")

    # 4. TRAIN_WEIGHTED_DF max date must be strictly < cutoff (backtest safety)
    _train_max = train_weighted_df["date"].max()
    assert _train_max < cfg.split.cutoff_date, (
        f"TRAIN_WEIGHTED_DF max date {_train_max} is not strictly "
        f"before cutoff {cfg.split.cutoff_date}. "
        "This would allow test-period information into the backtest reference date."
    )
    print(f"  ✓ TRAIN_WEIGHTED_DF max date ({_train_max}) < cutoff ({cfg.split.cutoff_date})")

    # 5. DC_RATINGS_BACKTEST must contain both fixture teams
    for team, role in [
        (home_team, "home"),
        (away_team, "away"),
    ]:
        assert team in dc_ratings.attack, (
            f"DC_RATINGS_BACKTEST missing attack rating for {role} team '{team}'. "
            "Check that both fixture teams are in CORE_TEAMS."
        )
    print("  ✓ DC_RATINGS_BACKTEST contains both fixture teams")

    # 6. FeatureConfig.rolling_window >= 3 (enforced in __post_init__ but re-check here)
    assert cfg.features.rolling_window >= 3, (
        f"rolling_window={cfg.features.rolling_window} is too small; minimum is 3."
    )
    print(f"  ✓ rolling_window={cfg.features.rolling_window} ≥ 3")

    print("── Validation passed ─────────────────────────────────────────────\n")

def add_rolling_form_features(df: pl.DataFrame, window: int) -> pl.DataFrame:
    """Compute leak-free rolling form stats and join them onto every match row.

    Algorithm
    ---------
    1.  Pivot the match frame to a long "team perspective" frame: each match
        produces two rows, one for the home side and one for the away side.
        Each row carries goals_for, goals_against, and points (3/1/0) from
        that team's perspective.

    2.  Sort by ('team', 'date', 'match_id') — strictly chronological per team,
        with match_id as the tiebreaker for same-day fixtures.

    3.  For each team, shift every statistic column by 1 (shift(1)) BEFORE
        computing the rolling mean.  This guarantees that a match's own result
        NEVER enters its own feature — the single most important leakage guard.

    4.  Compute rolling_mean(window, min_samples=1) on the shifted columns.
        min_samples=1 means a team with fewer than `window` past matches still
        gets a feature value (based on fewer games), rather than a null.

    5.  Pivot back to a wide format (one row per match) by joining the home
        and away form columns onto the original frame via match_id.

    6.  Return df with six new columns added:
          home_form_goals_for, home_form_goals_against, home_form_points
          away_form_goals_for, away_form_goals_against, away_form_points

    Critical leakage note
    ---------------------
    This function MUST be called on CORE_MATCH_DF (train + test combined), not
    on TRAIN_DF or TEST_DF separately.  A test match that is the first
    post-cutoff match for a team would have no rolling history at all if we
    computed rolling form on the test set alone.  The shift(1) ensures no
    future information ever enters a row's feature, so calling the function on
    the combined frame is safe — the re-split by date happens afterward.

    Args:
        df:     Played match DataFrame with columns:
                  match_id (UInt32), date (Date), home_team, away_team,
                  home_score (Int64), away_score (Int64)
                Must be sorted by date ascending before calling.
        window: Rolling window size (number of preceding matches per team).

    Returns:
        df with six additional Float64 columns for home and away rolling form.
    """
    assert window >= 1, "window must be at least 1"
    assert "match_id" in df.columns, "df must have a 'match_id' column"
    assert df["date"].to_list() == sorted(df["date"].to_list()), (
        "df must be sorted chronologically before add_rolling_form_features"
    )

    # ── Step 1: build long team-perspective frame ──────────────────────────
    # Home side rows
    home_rows = df.select([
        pl.col("match_id"),
        pl.col("date"),
        pl.col("home_team").alias("team"),
        pl.col("home_score").cast(pl.Float64).alias("goals_for"),
        pl.col("away_score").cast(pl.Float64).alias("goals_against"),
        # Points: 3 for win, 1 for draw, 0 for loss
        pl.when(pl.col("home_score") > pl.col("away_score")).then(pl.lit(3.0))
          .when(pl.col("home_score") == pl.col("away_score")).then(pl.lit(1.0))
          .otherwise(pl.lit(0.0))
          .alias("points"),
        pl.lit("home").alias("side"),
    ])

    # Away side rows
    away_rows = df.select([
        pl.col("match_id"),
        pl.col("date"),
        pl.col("away_team").alias("team"),
        pl.col("away_score").cast(pl.Float64).alias("goals_for"),
        pl.col("home_score").cast(pl.Float64).alias("goals_against"),
        pl.when(pl.col("away_score") > pl.col("home_score")).then(pl.lit(3.0))
          .when(pl.col("away_score") == pl.col("home_score")).then(pl.lit(1.0))
          .otherwise(pl.lit(0.0))
          .alias("points"),
        pl.lit("away").alias("side"),
    ])

    long_df = pl.concat([home_rows, away_rows])

    # ── Step 2: sort chronologically per team ─────────────────────────────
    long_df = long_df.sort(["team", "date", "match_id"])

    # ── Steps 3 & 4: shift(1) then rolling_mean — THE leak guard ──────────
    # shift(1) over "team" excludes the current match from its own window.
    # min_samples=1 (renamed from min_periods in Polars 1.21) gives a partial
    # mean for teams with fewer than `window` historical matches rather than null.
    long_df = long_df.with_columns([
        pl.col("goals_for")
          .shift(1)
          .rolling_mean(window_size=window, min_samples=1)
          .over("team")
          .alias("form_goals_for"),
        pl.col("goals_against")
          .shift(1)
          .rolling_mean(window_size=window, min_samples=1)
          .over("team")
          .alias("form_goals_against"),
        pl.col("points")
          .shift(1)
          .rolling_mean(window_size=window, min_samples=1)
          .over("team")
          .alias("form_points"),
    ])

    # ── Step 5: pivot back to wide format ─────────────────────────────────
    # Extract home-side and away-side form columns separately and join.
    home_form = (
        long_df
        .filter(pl.col("side") == "home")
        .select([
            pl.col("match_id"),
            pl.col("form_goals_for").alias("home_form_goals_for"),
            pl.col("form_goals_against").alias("home_form_goals_against"),
            pl.col("form_points").alias("home_form_points"),
        ])
    )

    away_form = (
        long_df
        .filter(pl.col("side") == "away")
        .select([
            pl.col("match_id"),
            pl.col("form_goals_for").alias("away_form_goals_for"),
            pl.col("form_goals_against").alias("away_form_goals_against"),
            pl.col("form_points").alias("away_form_points"),
        ])
    )

    # ── Step 6: join back onto original frame ──────────────────────────────
    result = (
        df
        .join(home_form, on="match_id", how="left")
        .join(away_form,  on="match_id", how="left")
    )

    return result

def run_rolling_form_leakage_spot_check(
    form_df: pl.DataFrame,
    team: str,
    window: int,
) -> None:
    """Manually verify that rolling form excludes the current match (shift(1)).

    Selects the `team`'s last (window + 2) matches from form_df, extracts their
    raw home/away goals_for sequence, and asserts that the penultimate match's
    form_goals_for equals the rolling mean of the goals_for values for the
    preceding `window` matches — NOT including the match itself.

    Args:
        form_df: Output of add_rolling_form_features(CORE_MATCH_DF, window).
        team:    Team to spot-check (e.g. FIXTURE_HOME_TEAM).
        window:  Rolling window size used in add_rolling_form_features.

    Raises:
        AssertionError: if leakage is detected or the manual calculation
                        does not match the feature column value.
    """
    print(f"── Leakage spot-check for '{team}' (window={window}) ────────────")

    # Collect all matches for this team (as home OR away) in date order
    team_home = (
        form_df
        .filter(pl.col("home_team") == team)
        .select([
            "match_id", "date",
            pl.col("home_score").cast(pl.Float64).alias("goals_for"),
            pl.col("away_score").cast(pl.Float64).alias("goals_against"),
            pl.col("home_form_goals_for").alias("feature_goals_for"),
            pl.lit("home").alias("side"),
        ])
    )
    team_away = (
        form_df
        .filter(pl.col("away_team") == team)
        .select([
            "match_id", "date",
            pl.col("away_score").cast(pl.Float64).alias("goals_for"),
            pl.col("home_score").cast(pl.Float64).alias("goals_against"),
            pl.col("away_form_goals_for").alias("feature_goals_for"),
            pl.lit("away").alias("side"),
        ])
    )
    team_matches = (
        pl.concat([team_home, team_away])
        .sort(["date", "match_id"])
    )

    n_matches = team_matches.height
    assert n_matches >= window + 2, (
        f"Team '{team}' has only {n_matches} matches; need at least {window + 2} "
        "to spot-check a full rolling window. Choose a team with more history."
    )

    # Take the last (window + 2) matches so we have a full window to verify
    check_slice = team_matches.tail(window + 2)
    goals_for_seq = check_slice["goals_for"].to_list()
    feature_seq   = check_slice["feature_goals_for"].to_list()

    # For the LAST match in the slice:
    #   - Its feature should be the mean of goals_for[-(window+1) : -1]
    #   - i.e. the window matches BEFORE it, NOT including it
    target_feature = feature_seq[-1]
    preceding_goals = goals_for_seq[-(window + 1) : -1]   # exactly `window` values

    assert len(preceding_goals) == window, (
        f"Expected {window} preceding values, got {len(preceding_goals)}."
    )

    manual_mean = sum(preceding_goals) / window

    # Check: feature must NOT equal the mean that includes the current match
    mean_with_current = sum(goals_for_seq[-window:]) / window
    if abs(manual_mean - mean_with_current) > 1e-6:
        # Only meaningful to run this if the two differ (otherwise shift is undetectable)
        assert abs(target_feature - mean_with_current) > 1e-6, (
            "LEAKAGE DETECTED: the feature value equals the rolling mean "
            "INCLUDING the current match's own goals_for.  "
            "shift(1) is not working correctly."
        )

    # Primary assertion: feature equals the mean of the PREVIOUS window matches
    assert abs(target_feature - manual_mean) < 1e-4, (
        f"Spot-check FAILED for '{team}' last match:\n"
        f"  Expected (mean of {window} preceding goals_for): {manual_mean:.4f}\n"
        f"  Actual feature value:                            {target_feature:.4f}\n"
        f"  Difference: {abs(target_feature - manual_mean):.6f}\n"
        f"  Preceding goals_for seq: {preceding_goals}\n"
        "  This indicates either shift(1) is missing or wrong sort order."
    )

    print(f"  Team matches available: {n_matches}")
    print("  Checking last match in slice:")
    print(f"    Preceding {window} goals_for: {[round(g, 1) for g in preceding_goals]}")
    print(f"    Manual mean:         {manual_mean:.4f}")
    print(f"    Feature column val:  {target_feature:.4f}")

def add_dixon_coles_features(
    df: pl.DataFrame,
    ratings: "DixonColesRatings",
) -> pl.DataFrame:
    """Join the four DC rating columns and is_neutral onto a match frame.

    Four new Float64 columns are added:
      home_attack_dc, home_defense_dc, away_attack_dc, away_defense_dc

    Plus one Int8 column:
      is_neutral  (cast from the boolean `neutral` column)

    Teams not present in ratings.attack receive null (not 0.0) so that
    downstream null-auditing can detect unrated teams rather than silently
    imputing a misleading zero.

    Args:
        df:      Match DataFrame (must have home_team, away_team, neutral cols).
        ratings: DixonColesRatings (train-only for backtest, full for production).

    Returns:
        df with five additional columns.
    """
    attack_map  = ratings.attack
    defense_map = ratings.defense

    # Build replacement series via map_elements with default=null for unknowns
    result = df.with_columns([
        pl.col("home_team")
          .map_elements(lambda t: attack_map.get(t, None), return_dtype=pl.Float64)
          .alias("home_attack_dc"),
        pl.col("home_team")
          .map_elements(lambda t: defense_map.get(t, None), return_dtype=pl.Float64)
          .alias("home_defense_dc"),
        pl.col("away_team")
          .map_elements(lambda t: attack_map.get(t, None), return_dtype=pl.Float64)
          .alias("away_attack_dc"),
        pl.col("away_team")
          .map_elements(lambda t: defense_map.get(t, None), return_dtype=pl.Float64)
          .alias("away_defense_dc"),
        pl.col("neutral").cast(pl.Int8).alias("is_neutral"),
    ])

    return result

def assemble_feature_df(
    core_match_df:      pl.DataFrame,
    train_weighted_df:  pl.DataFrame,
    dc_ratings_backtest:"DixonColesRatings",
    cfg:                "PipelineConfig",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build TRAIN_FEATURES and TEST_FEATURES from the full match history.

    Steps
    -----
    1. ``add_rolling_form_features`` on ``core_match_df`` (train + test
       combined) — leak-free by construction (shift before rolling).
    2. ``add_dixon_coles_features`` with ``dc_ratings_backtest``.
    3. Re-split by date: TRAIN / TEST.
    4. Join ``match_weight`` from ``train_weighted_df`` onto TRAIN_FEATURES.
    5. Null audit: assert zero nulls in training feature columns; warn on
       any nulls in test feature columns.

    Args:
        core_match_df:       Full played-match history filtered to core teams,
                             sorted by date ascending.  Must contain
                             ``match_id``.
        train_weighted_df:   Training frame with ``match_id`` and
                             ``match_weight`` columns.
        dc_ratings_backtest: Train-only Dixon-Coles ratings used for the four
                             static DC feature columns.
        cfg:                 Pipeline configuration; uses
                             ``cfg.features.rolling_window``,
                             ``cfg.split.cutoff_date``, and
                             ``cfg.features.feature_columns``.

    Returns:
        Tuple of ``(TRAIN_FEATURES, TEST_FEATURES)`` Polars DataFrames.

    Raises:
        AssertionError: If ``train_weighted_df`` contains rows with null
            ``match_weight`` after joining, or if any training feature column
            contains nulls.
    """
    # Step 1: rolling form (on combined history — leak-free by construction)
    form_df = add_rolling_form_features(core_match_df, cfg.features.rolling_window)

    # Step 2: DC features
    feature_df = add_dixon_coles_features(form_df, dc_ratings_backtest)

    # Step 3: re-split by date
    train_features = feature_df.filter(pl.col("date") < cfg.split.cutoff_date)
    test_features  = feature_df.filter(pl.col("date") >= cfg.split.cutoff_date)

    # Step 4: join match_weight onto training features
    weight_cols    = train_weighted_df.select(["match_id", "match_weight"])
    train_features = train_features.join(weight_cols, on="match_id", how="left")

    _n_null_weight = train_features["match_weight"].null_count()
    assert _n_null_weight == 0, (
        f"TRAIN_FEATURES has {_n_null_weight} rows with null match_weight "
        "after joining TRAIN_WEIGHTED_DF. "
        "Ensure TRAIN_WEIGHTED_DF covers all training match_ids."
    )

    # Drop any training rows where DC features are null (teams that fell below
    # the core-team threshold between production and backtest fits).  Log the
    # count so the caller can surface it — never drop silently.
    _n_before = train_features.height
    train_features = train_features.drop_nulls(subset=cfg.features.feature_columns)
    _n_dropped = _n_before - train_features.height
    if _n_dropped > 0:
        print(
            f"  ⚠ TRAIN_FEATURES: dropped {_n_dropped} rows with null feature "
            "values (teams absent from DC_RATINGS_BACKTEST)."
        )

    # Step 5: null audit — hard failure on training nulls, warning on test nulls
    _all_feature_cols  = cfg.features.feature_columns
    _train_null_total  = sum(
        train_features[c].null_count()
        for c in _all_feature_cols
        if c in train_features.columns
    )
    assert _train_null_total == 0, (
        f"TRAIN_FEATURES has {_train_null_total} total nulls across feature "
        "columns after drop_nulls.  XGBoost cannot receive null inputs."
    )

    _test_null_total = sum(
        test_features[c].null_count()
        for c in _all_feature_cols
        if c in test_features.columns
    )
    if _test_null_total > 0:
        print(
            f"  ⚠ TEST_FEATURES: {_test_null_total} nulls across feature "
            "columns (teams absent from DC backtest ratings). "
            "These rows will be excluded from evaluation."
        )

    return train_features, test_features

def compute_current_form(df: pl.DataFrame, window: int) -> pl.DataFrame:
    """Compute each team's form snapshot over their most recent `window` matches.

    This is the UNSHIFTED version of rolling form, used ONLY in Section 15 to
    build the feature row for the headline (not-yet-played) fixture.

    It must NEVER be used for backtest training or test feature rows — doing so
    would include the current match's own result in the feature, which is
    leakage.  The docstring distinguishes it from add_rolling_form_features
    by naming this "compute_current_form" and marking it as production-only.

    Args:
        df:     CORE_MATCH_DF (or any played-match frame, sorted by date).
        window: Number of recent matches to average over.

    Returns:
        One-row-per-team pl.DataFrame with columns:
          team, form_goals_for, form_goals_against, form_points
        Only teams that appear in df are included.
    """
    # Build long frame (same pivot as add_rolling_form_features)
    home_rows = df.select([
        pl.col("home_team").alias("team"),
        pl.col("date"),
        pl.col("match_id"),
        pl.col("home_score").cast(pl.Float64).alias("goals_for"),
        pl.col("away_score").cast(pl.Float64).alias("goals_against"),
        pl.when(pl.col("home_score") > pl.col("away_score")).then(pl.lit(3.0))
          .when(pl.col("home_score") == pl.col("away_score")).then(pl.lit(1.0))
          .otherwise(pl.lit(0.0))
          .alias("points"),
    ])
    away_rows = df.select([
        pl.col("away_team").alias("team"),
        pl.col("date"),
        pl.col("match_id"),
        pl.col("away_score").cast(pl.Float64).alias("goals_for"),
        pl.col("home_score").cast(pl.Float64).alias("goals_against"),
        pl.when(pl.col("away_score") > pl.col("home_score")).then(pl.lit(3.0))
          .when(pl.col("away_score") == pl.col("home_score")).then(pl.lit(1.0))
          .otherwise(pl.lit(0.0))
          .alias("points"),
    ])
    long_df = pl.concat([home_rows, away_rows]).sort(["team", "date", "match_id"])

    # Tail `window` rows per team, then aggregate — no shift, because we WANT
    # all recent matches (no future match to exclude)
    form = (
        long_df
        .group_by("team")
        .map_groups(lambda grp: grp.tail(window))
        .group_by("team")
        .agg([
            pl.col("goals_for").mean().alias("form_goals_for"),
            pl.col("goals_against").mean().alias("form_goals_against"),
            pl.col("points").mean().alias("form_points"),
        ])
        .sort("team")
    )

    return form

def run_feature_engineering(
    core_match_df:        pl.DataFrame,
    train_weighted_df:    pl.DataFrame,
    dc_ratings_backtest:  "DixonColesRatings",
    cfg:                  "PipelineConfig",
    fixture_home_team:    str,
    fixture_away_team:    str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Run the full Section 7 feature engineering pipeline.

    Orchestrates validation → rolling form → leakage check → DC features →
    re-split → weight join → null audit.

    Args:
        core_match_df:        Full played-match history (train + test),
                              sorted by date, filtered to core teams.
        train_weighted_df:    Training frame with ``match_weight``.
        dc_ratings_backtest:  Train-only Dixon-Coles ratings.
        cfg:                  Pipeline configuration.
        fixture_home_team:    Dataset name of the headline fixture home team
                              (used in leakage spot-check).
        fixture_away_team:    Dataset name of the headline fixture away team
                              (used in leakage spot-check).

    Returns:
        ``(TRAIN_FEATURES, TEST_FEATURES, FORM_DF)``

        ``FORM_DF`` is the output of ``add_rolling_form_features`` on
        ``core_match_df``.  It is returned so the caller can run additional
        leakage spot-checks without recomputing the expensive pivot.
    """
    # ── Pre-implementation validation ──────────────────────────────────────
    _validate_feature_inputs(
        core_match_df, train_weighted_df, dc_ratings_backtest, cfg,
        fixture_home_team, fixture_away_team
    )

    # ── Compute FORM_DF once; reuse it for both the spot-check and assembly ─
    # IMPORTANT: assemble_feature_df internally calls add_rolling_form_features
    # again, so we pass form_df directly to avoid computing the expensive pivot
    # twice.  The leakage check uses the SAME form_df that will become
    # TRAIN_FEATURES, preserving the invariant that the check covers the
    # exact frame the model trains on.
    form_df = add_rolling_form_features(core_match_df, cfg.features.rolling_window)

    # ── 7.2 Leakage spot-check (mandatory before any split) ───────────────
    for team in [fixture_home_team, fixture_away_team]:
        run_rolling_form_leakage_spot_check(form_df, team, cfg.features.rolling_window)

    # ── 7.3 + 7.4  DC features, re-split, weight join, null audit ─────────
    # Build feature_df from the already-computed form_df (no second pivot).
    feature_df     = add_dixon_coles_features(form_df, dc_ratings_backtest)
    train_features = feature_df.filter(pl.col("date") < cfg.split.cutoff_date)
    test_features  = feature_df.filter(pl.col("date") >= cfg.split.cutoff_date)

    weight_cols    = train_weighted_df.select(["match_id", "match_weight"])
    train_features = train_features.join(weight_cols, on="match_id", how="left")

    _n_null_weight = train_features["match_weight"].null_count()
    assert _n_null_weight == 0, (
        f"match_weight has {_n_null_weight} nulls after join. "
        "TRAIN_WEIGHTED_DF match_ids do not align with TRAIN_FEATURES."
    )

    # Drop rows with null feature values and report the count.
    _n_before = train_features.height
    train_features = train_features.drop_nulls(subset=cfg.features.feature_columns)
    _n_dropped = _n_before - train_features.height
    if _n_dropped > 0:
        print(
            f"  ⚠ TRAIN_FEATURES: dropped {_n_dropped} rows with null DC "
            "features (teams absent from DC_RATINGS_BACKTEST)."
        )

    # ── Summary banner ──────────────────────────────────────────────────────
    print("Section 7 — Feature Engineering")
    print(SEP)
    print(f"  FORM_DF:        {form_df.height:,} rows")
    print(f"  FEATURE_DF:     {feature_df.height:,} rows")
    print(f"  TRAIN_FEATURES: {train_features.height:,} rows")
    print(f"  TEST_FEATURES:  {test_features.height:,} rows")
    print(f"  match_weight:   {_n_null_weight} nulls ✓")

    # ── Null audit ─────────────────────────────────────────────────────────
    print("\n  Null audit — feature columns:")
    print(f"    {'Feature':<30} {'Train':>8} {'Test':>8}")
    print("    " + "─" * 50)
    _train_null_total = 0
    for col in cfg.features.feature_columns:
        tn = train_features[col].null_count() if col in train_features.columns else -1
        tt = test_features[col].null_count()  if col in test_features.columns  else -1
        flag = " ⚠" if tt > 0 else ("  ✓" if tn == 0 else "")
        print(f"    {col:<30} {tn:>8,} {tt:>8,}{flag}")
        _train_null_total += max(tn, 0)

    assert _train_null_total == 0, (
        f"TRAIN_FEATURES has {_train_null_total} total nulls across feature "
        "columns. Cannot pass null inputs to XGBoost."
    )

    print("  Section 7 complete\n")

    return train_features, test_features, form_df

def build_fixture_feature_row(
    home_team:    str,
    away_team:    str,
    is_neutral:   bool,
    ratings:      DixonColesRatings,
    current_form: pl.DataFrame,
    feature_cols: list[str],
) -> pl.DataFrame:
    """Assemble the single-row feature frame for the headline fixture.

    Pulls production DC attack/defense ratings and the current (unshifted)
    form snapshot for both teams, then packages them into a one-row Polars
    DataFrame with exactly the columns listed in ``feature_cols``.

    This function must only be called with PRODUCTION-fitted ratings
    (DC_RATINGS_PROD) and the unshifted form snapshot (CURRENT_FORM_DF).
    Using backtest ratings or shifted form here would produce an inconsistent
    production prediction.

    Args:
        home_team:    Dataset name of the home team.
        away_team:    Dataset name of the away team.
        is_neutral:   True if the match is at a neutral venue (zeroes
                      home-advantage in the DC/Bayesian models; sets
                      is_neutral=1 in the XGBoost feature).
        ratings:      Production DixonColesRatings (DC_RATINGS_PROD).
        current_form: Output of compute_current_form (CURRENT_FORM_DF).
        feature_cols: Ordered list from CFG.features.feature_columns.
                      The returned DataFrame has exactly these columns,
                      in this order.

    Returns:
        Single-row pl.DataFrame with exactly ``len(feature_cols)`` columns.

    Raises:
        KeyError:  If either team is absent from ``ratings`` or
                   ``current_form``.
        AssertionError: If the returned frame does not have exactly one row
                        or the column set does not match ``feature_cols``.
    """
    # Pull DC ratings
    if home_team not in ratings.attack:
        raise KeyError(
            f"'{home_team}' has no production DC rating. "
            "Ensure DC_RATINGS_PROD covers this team."
        )
    if away_team not in ratings.attack:
        raise KeyError(
            f"'{away_team}' has no production DC rating. "
            "Ensure DC_RATINGS_PROD covers this team."
        )

    home_attack_dc  = ratings.attack[home_team]
    home_defense_dc = ratings.defense[home_team]
    away_attack_dc  = ratings.attack[away_team]
    away_defense_dc = ratings.defense[away_team]

    # Pull current form for home team
    _home_form_row = current_form.filter(pl.col("team") == home_team)
    if _home_form_row.height == 0:
        raise KeyError(
            f"'{home_team}' has no entry in current_form. "
            "Ensure CORE_MATCH_DF contains matches for this team."
        )
    _hf = _home_form_row.row(0, named=True)

    # Pull current form for away team
    _away_form_row = current_form.filter(pl.col("team") == away_team)
    if _away_form_row.height == 0:
        raise KeyError(
            f"'{away_team}' has no entry in current_form. "
            "Ensure CORE_MATCH_DF contains matches for this team."
        )
    _af = _away_form_row.row(0, named=True)

    # Build the feature dict using the canonical column names.
    # int(is_neutral) maps True->1, False->0, consistent with Section 7.3
    # which uses pl.col("neutral").cast(pl.Int8).alias("is_neutral").
    _feature_dict: dict[str, float] = {
        "home_attack_dc":          float(home_attack_dc),
        "home_defense_dc":         float(home_defense_dc),
        "away_attack_dc":          float(away_attack_dc),
        "away_defense_dc":         float(away_defense_dc),
        "home_form_goals_for":     float(_hf["form_goals_for"]),
        "home_form_goals_against": float(_hf["form_goals_against"]),
        "home_form_points":        float(_hf["form_points"]),
        "away_form_goals_for":     float(_af["form_goals_for"]),
        "away_form_goals_against": float(_af["form_goals_against"]),
        "away_form_points":        float(_af["form_points"]),
        "is_neutral":              float(int(is_neutral)),
    }

    # Select only the columns listed in feature_cols, in their canonical order.
    # This guards against column-order drift between the training frame and the
    # prediction frame — XGBoost is order-sensitive.
    missing = [c for c in feature_cols if c not in _feature_dict]
    if missing:
        raise ValueError(
            f"build_fixture_feature_row: feature_cols contains column(s) not "
            f"populated by this function: {missing}. "
            "Update _feature_dict or FeatureConfig.feature_columns."
        )

    ordered_row = {c: [_feature_dict[c]] for c in feature_cols}
    row_df = pl.DataFrame(ordered_row)

    assert row_df.height == 1, "build_fixture_feature_row must return exactly one row."
    assert list(row_df.columns) == list(feature_cols), (
        "Column order mismatch between returned frame and feature_cols."
    )

    return row_df