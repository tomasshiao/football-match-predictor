import polars as pl

def load_fifa_code_mapping(source: dict[str, str]) -> dict[str, str]:
    """Wrap a FIFA-code to dataset-team-name dictionary in a typed copy.

    Args:
        source: The canonical ``FIFA_TO_DATASET_TEAM`` dict (or any
            equivalent mapping).

    Returns:
        A new ``dict[str, str]`` containing the same key-value pairs.
    """
    return dict(source)


def resolve_team_name(fifa_code: str, mapping: dict[str, str]) -> str:
    """Resolve a FIFA three-letter code to the dataset team name string.

    Args:
        fifa_code: Upper-case three-letter FIFA code (e.g. ``'ARG'``).
        mapping: Code to team-name dictionary (from ``load_fifa_code_mapping``).

    Returns:
        The exact team name string used in ``results.csv``.

    Raises:
        KeyError: If ``fifa_code`` is not in ``mapping``, with a
            descriptive message listing the code.
    """
    try:
        return mapping[fifa_code]
    except KeyError:
        raise KeyError(
            f"FIFA code {fifa_code!r} not found in the code mapping. "
            f"Check fifa_country_codes.py for the correct code."
        ) from None

def select_core_teams(df: pl.DataFrame, min_matches: int) -> list[str]:
    """Return teams with at least min_matches appearances in df.

    Appearance counts are the sum of home and away occurrences combined.
    This function must be called on the **training** frame only to
    prevent test-set match frequencies from influencing which teams
    are in the model universe.

    Args:
        df: Match DataFrame (training window only).
        min_matches: Minimum number of matches required for inclusion.

    Returns:
        Sorted list of team name strings meeting the threshold.
    """
    home_counts = (
        df.group_by("home_team")
        .agg(pl.len().alias("n"))
        .rename({"home_team": "team"})
    )
    away_counts = (
        df.group_by("away_team")
        .agg(pl.len().alias("n"))
        .rename({"away_team": "team"})
    )
    total_counts = (
        pl.concat([home_counts, away_counts])
        .group_by("team")
        .agg(pl.col("n").sum())
    )
    core = (
        total_counts
        .filter(pl.col("n") >= min_matches)
        ["team"]
        .to_list()
    )
    return sorted(core)

def filter_to_core_teams(df: pl.DataFrame, core_teams: list[str]) -> pl.DataFrame:
    """Filter a match DataFrame to rows where both teams are in core_teams.

    Args:
        df: Match DataFrame (played matches only).
        core_teams: List of team name strings in the model universe.

    Returns:
        Filtered DataFrame; the original row order is preserved.
    """
    _core_set = set(core_teams)
    return df.filter(
        pl.col("home_team").is_in(_core_set)
        & pl.col("away_team").is_in(_core_set)
    )