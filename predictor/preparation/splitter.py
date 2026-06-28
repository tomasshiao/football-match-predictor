import datetime
import polars as pl

def split_played_and_upcoming(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a match DataFrame into played and upcoming fixtures.

    Played matches have non-null scores in *both* ``home_score`` and
    ``away_score``.  Fixtures with at least one null score are treated
    as upcoming (i.e. not yet played).

    Args:
        df: Raw match DataFrame (all rows, including future fixtures).

    Returns:
        Tuple of (played_df, upcoming_df).  Their row counts sum to
        ``len(df)``.

    Raises:
        AssertionError: If the union of row counts does not equal
            ``len(df)``.
    """
    played = df.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    upcoming = df.filter(
        pl.col("home_score").is_null() | pl.col("away_score").is_null()
    )
    assert played.height + upcoming.height == df.height, (
        "split_played_and_upcoming: played + upcoming != total rows. "
        "Check for unexpected null patterns."
    )
    return played, upcoming

def filter_from_date(
    df: pl.DataFrame,
    start_date: datetime.date,
) -> pl.DataFrame:
    """Restrict a match DataFrame to rows on or after start_date.

    A sequential ``match_id`` index column is attached after filtering
    so that rolling-form logic in Section 7 can use it as a stable,
    monotonically increasing row identifier.

    Args:
        df: Played match DataFrame (scores must be non-null).
        start_date: Earliest date to include (inclusive).

    Returns:
        Filtered DataFrame with an additional ``match_id`` column
        (``UInt32``) starting at 0.
    """
    filtered = df.filter(pl.col("date") >= start_date)
    filtered = filtered.with_row_index("match_id")
    return filtered

def train_test_split_by_date(
    df: pl.DataFrame,
    cutoff_date: datetime.date,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a played-match DataFrame at a chronological boundary.

    Every match in the returned test set has ``date >= cutoff_date``;
    every match in the training set has ``date < cutoff_date``.
    No row appears in both sets.

    Args:
        df: Played match DataFrame, sorted by date (ascending).
        cutoff_date: Exclusive upper bound for the training set.

    Returns:
        Tuple of (train_df, test_df).

    Raises:
        AssertionError: If the maximum training date is not strictly
            before ``cutoff_date``, or if the minimum test date is
            not ``>= cutoff_date``.
    """
    train = df.filter(pl.col("date") < cutoff_date)
    test  = df.filter(pl.col("date") >= cutoff_date)

    if train.height > 0 and test.height > 0:
        _train_max: datetime.date = train["date"].max()
        _test_min:  datetime.date = test["date"].min()
        assert _train_max < cutoff_date, (
            f"train_test_split_by_date: max train date {_train_max} "
            f">= cutoff_date {cutoff_date}"
        )
        assert _test_min >= cutoff_date, (
            f"train_test_split_by_date: min test date {_test_min} "
            f"< cutoff_date {cutoff_date}"
        )
    return train, test
