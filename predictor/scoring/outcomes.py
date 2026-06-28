from dataclasses import dataclass
import datetime
import numpy as np
import polars as pl

@dataclass
class OutcomeProbabilities:
    """Collapsed 1X2 outcome probabilities derived from a scoreline matrix.

    All three fields are non-negative and sum to 1.0 (up to floating-point
    rounding).  Computed by ``score_matrix_to_outcome_probs``; never
    constructed manually.

    Attributes:
        p_home_win: Probability that the home team wins (home goals > away goals).
        p_draw:     Probability of a draw (home goals == away goals).
        p_away_win: Probability that the away team wins (away goals > home goals).
    """

    p_home_win: float
    p_draw:     float
    p_away_win: float

    def __str__(self) -> str:
        return (
            f"Home win: {self.p_home_win:.1%}  |  "
            f"Draw: {self.p_draw:.1%}  |  "
            f"Away win: {self.p_away_win:.1%}"
        )

@dataclass
class PlayoffOutcomeProbabilities:
    """Four-way outcome probabilities for a playoff / knockout fixture.

    Replaces the three-way (home win / draw / away win) output when
    ``CFG.fixture.is_playoff`` is True.  A draw after 90 minutes leads
    to a penalty shootout; there is no 'draw' final result.

    Attributes:
        p_home_win:             P(home team wins within 90 min).
        p_home_win_penalties:   P(draw after 90 min AND home team wins shootout).
        p_away_win_penalties:   P(draw after 90 min AND away team wins shootout).
        p_away_win:             P(away team wins within 90 min).

    Note:
        The four probabilities sum to 1.0.
        p_home_win_penalties + p_away_win_penalties == p_draw_90min
        (the probability of a 90-minute draw from the Poisson matrix).
    """

    p_home_win:           float
    p_home_win_penalties: float
    p_away_win_penalties: float
    p_away_win:           float

    def __str__(self) -> str:
        return (
            f"Home win: {self.p_home_win:.1%}  |  "
            f"Home win (pen): {self.p_home_win_penalties:.1%}  |  "
            f"Away win (pen): {self.p_away_win_penalties:.1%}  |  "
            f"Away win: {self.p_away_win:.1%}"
        )

@dataclass
class FixturePrediction:
    """Complete prediction output for the headline fixture.

    Bundles per-model matrices, the ensemble matrix, derived outcome
    probabilities, top scorelines, and diagnostic values into one object
    for downstream consumption by the visualization cells (Sections 16-18)
    and the summary cell (Section 20).

    Attributes:
        home_team:          Dataset name of the home team.
        away_team:          Dataset name of the away team.
        match_date:         Date the fixture is scheduled.
        per_model_matrices: Dict mapping model name to its (K+1, K+1) matrix.
        ensemble_matrix:    Weight-averaged combination matrix.
        ensemble_weights:   Per-model weights used in the combination.
        outcome:            Collapsed 1X2 outcome probabilities.
        playoff_outcome:   Playoff outcome probabilities (if applicable).
        top_scorelines:     List of ((home, away), prob) tuples, sorted desc.
        entropy:            Shannon entropy of the ensemble matrix (nats).
    """

    home_team:           str
    away_team:           str
    match_date:          datetime.date
    per_model_matrices:  dict[str, np.ndarray]
    ensemble_matrix:     np.ndarray
    ensemble_weights:    dict[str, float]
    outcome:             OutcomeProbabilities
    playoff_outcome:     PlayoffOutcomeProbabilities | None
    top_scorelines:      list[tuple[tuple[int, int], float]]
    entropy:             float

def score_matrix_to_outcome_probs(matrix: np.ndarray) -> OutcomeProbabilities:
    """Collapse a scoreline matrix into 1X2 outcome probabilities.

    Home win  = sum of cells where home goals > away goals (upper triangle).
    Draw      = sum of diagonal cells.
    Away win  = sum of cells where away goals > home goals (lower triangle).

    Args:
        matrix: 2-D ``np.ndarray`` of shape ``(K+1, K+1)`` summing to 1.0.

    Returns:
        ``OutcomeProbabilities`` with ``p_home_win + p_draw + p_away_win`` ≈ 1.0.
    """
    k = matrix.shape[0]
    p_home = float(np.sum(np.tril(matrix, k=-1)))   # home goals > away goals
    p_draw = float(np.trace(matrix))
    p_away = float(np.sum(np.triu(matrix, k=1)))    # away goals > home goals
    return OutcomeProbabilities(p_home_win=p_home, p_draw=p_draw, p_away_win=p_away)


def top_n_scorelines(
    matrix: np.ndarray,
    n: int,
) -> list[tuple[tuple[int, int], float]]:
    """Return the n most probable exact scorelines, sorted descending by probability.

    Args:
        matrix: 2-D ``np.ndarray`` of shape ``(K+1, K+1)`` summing to 1.0.
                Row index is home goals; column index is away goals.
        n:      Number of scorelines to return.

    Returns:
        List of ``((home_goals, away_goals), probability)`` tuples,
        sorted highest-probability first.
    """
    flat_idx = np.argsort(matrix, axis=None)[::-1][:n]
    rows, cols = np.unravel_index(flat_idx, matrix.shape)
    return [((int(r), int(c)), float(matrix[r, c])) for r, c in zip(rows, cols)]

# --- Playoffs helper functions -------------------------------------------------------------
def estimate_shootout_win_prob(
    home_team:    str,
    away_team:    str,
    shootout_df:  pl.DataFrame,
    min_samples:  int = 3,
    prior:        float = 0.5,
) -> float:
    """Estimate P(home team wins a penalty shootout) against a given opponent.

    Uses a Laplace-smoothed empirical estimate from historical shootout
    data.  When the two specific teams have met in a shootout fewer than
    ``min_samples`` times, falls back to a global home-team shootout win
    rate computed across all historical shootouts, and ultimately to
    ``prior`` (default 0.50) when there is no usable data at all.

    The fallback chain is:
      1. Head-to-head shootout record (if >= min_samples meetings).
      2. Global historical home-team shootout win rate (if any data).
      3. ``prior`` (theoretical coin-flip default).

    Args:
        home_team:   Dataset team name of the nominally home team.
        away_team:   Dataset team name of the nominally away team.
        shootout_df: DataFrame with columns ['date','home_team','away_team','winner'].
        min_samples: Minimum head-to-head shootout meetings required before
                     trusting the pair-specific empirical rate.
        prior:       Fallback probability used when no data exists.

    Returns:
        Estimated probability in [0, 1] that the home team wins the shootout.
    """
    if shootout_df is None or shootout_df.is_empty():
        return prior

    # ── Global home-team win rate (all historical shootouts) ───────────
    _n_total = shootout_df.height
    _n_home_win_global = shootout_df.filter(
        pl.col("winner") == pl.col("home_team")
    ).height
    global_rate = _n_home_win_global / _n_total if _n_total > 0 else prior

    # ── Head-to-head: both match orientations count ────────────────────
    # (home_team vs away_team) OR (away_team vs home_team, then invert)
    _h2h = shootout_df.filter(
        (
            (pl.col("home_team") == home_team) & (pl.col("away_team") == away_team)
        ) | (
            (pl.col("home_team") == away_team) & (pl.col("away_team") == home_team)
        )
    )
    if _h2h.height < min_samples:
        return global_rate  # fall back to global rate

    # How many times did `home_team` win a shootout against `away_team`?
    # Count wins when home_team was the actual home side.
    _wins_as_home = _h2h.filter(
        (pl.col("home_team") == home_team) & (pl.col("winner") == home_team)
    ).height
    # Count wins when home_team was the actual away side (invert perspective).
    _wins_as_away = _h2h.filter(
        (pl.col("home_team") == away_team) & (pl.col("winner") == home_team)
    ).height
    _total_h2h = _h2h.height
    _home_team_wins = _wins_as_home + _wins_as_away

    # Laplace smoothing to avoid 0 or 1 probabilities from tiny samples
    return (_home_team_wins + 1) / (_total_h2h + 2)


def score_matrix_to_playoff_outcome_probs(
    matrix:           np.ndarray,
    home_team:        str,
    away_team:        str,
    shootout_df:      pl.DataFrame | None,
) -> PlayoffOutcomeProbabilities:
    """Collapse a scoreline matrix into four playoff outcome probabilities.

    Computes the standard three-way 1X2 probabilities from the matrix,
    then splits the draw probability into two penalty-shootout branches
    using ``estimate_shootout_win_prob``.

    Args:
        matrix:      (K+1, K+1) scoreline probability matrix summing to 1.
        home_team:   Dataset team name of the home team.
        away_team:   Dataset team name of the away team.
        shootout_df: Loaded shootout DataFrame, or None (uses prior 0.5).

    Returns:
        ``PlayoffOutcomeProbabilities`` with the four fields populated.
        The four values sum to 1.0.
    """
    k = matrix.shape[0]
    home_mask = np.tril(np.ones((k, k), dtype=bool), k=-1)
    draw_mask  = np.eye(k, dtype=bool)
    away_mask  = np.triu(np.ones((k, k), dtype=bool), k=1)

    p_home_win = float(matrix[home_mask].sum())
    p_draw_90  = float(matrix[draw_mask].sum())
    p_away_win = float(matrix[away_mask].sum())

    p_home_shootout = estimate_shootout_win_prob(
        home_team, away_team, shootout_df if shootout_df is not None else pl.DataFrame()
    )

    return PlayoffOutcomeProbabilities(
        p_home_win           = p_home_win,
        p_home_win_penalties = p_draw_90 * p_home_shootout,
        p_away_win_penalties = p_draw_90 * (1.0 - p_home_shootout),
        p_away_win           = p_away_win,
    )