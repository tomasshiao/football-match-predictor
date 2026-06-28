import datetime
import pathlib
from dataclasses import dataclass
from common.tournament_weights import TOURNAMENT_WEIGHTS

# --- DataConfig -------------------------------------------
@dataclass(frozen=True)
class DataConfig:
    """Configuration for raw data acquisition.

    Args:
        source_url: Direct URL to the raw results CSV file.
        cache_path: Local path for caching the downloaded CSV. ``None``
            disables caching and forces a fresh download on every run.
        start_date: Earliest match date to include in the analysis.
            Matches played before this date are discarded after ingestion.
    """

    source_url: str
    shootout_url: str
    cache_path: pathlib.Path | None
    start_date: datetime.date

# --- TeamFilterConfig --------------------------------
@dataclass(frozen=True)
class TeamFilterConfig:
    """Configuration for the core-team universe.

    Args:
        min_matches: Minimum number of matches (home + away combined) a team
            must have played in the *training window* to be included in model
            fitting and backtest evaluation.  Teams below this threshold are
            excluded silently only after the exclusion count has been printed.
    """

    min_matches: int

# --- WeightingConfig --------------------------------
def get_tournament_weight_table() -> dict[str, float]:
    """Return the canonical tournament-importance weight table.

    Keys are **exact** strings as they appear in the ``results.csv``
    ``tournament`` column (including accented characters). Weights are
    multiplicative scalars applied before the time-decay weight.

    Returns:
        Mapping from tournament name to importance weight.

    Note:
        An exact-string match is intentional: substring or
        accent-normalised matching introduced two confirmed bugs in an
        earlier draft (silent Copa America misweighting and World Cup
        qualifier/final conflation).
    """
    return TOURNAMENT_WEIGHTS


@dataclass(frozen=True)
class WeightingConfig:
    """Configuration for per-match sample weighting.

    Args:
        half_life_days: Number of days before the reference date at which
            a match receives exactly half the base weight.  Exponential
            decay: ``weight = exp(-log(2) / half_life_days * delta_days)``.
        tournament_weight_table: Exact-string mapping from tournament name
            to a multiplicative importance scalar.  Keys must match the
            ``tournament`` column strings in the dataset verbatim.
        default_tournament_weight: Fallback weight for tournaments not
            found in ``tournament_weight_table``.  A warning is emitted
            for every unrecognised tournament name.
    """

    half_life_days: int
    tournament_weight_table: dict[str, float]
    default_tournament_weight: float

    def __post_init__(self) -> None:
        _required = {
            "Copa América", "FIFA World Cup", "UEFA Euro",
            "African Cup of Nations", "FIFA World Cup qualification", "Friendly",
        }
        _missing = _required - self.tournament_weight_table.keys()
        assert not _missing, (
            f"WeightingConfig.tournament_weight_table is missing required keys: {_missing}"
        )
        assert self.half_life_days > 0, "half_life_days must be positive"
        assert self.default_tournament_weight > 0, "default_tournament_weight must be positive"

# --- SplitConfig --------------------------------
@dataclass(frozen=True)
class SplitConfig:
    """Chronological train/test split configuration.

    Args:
        cutoff_date: Matches with ``date < cutoff_date`` form the training
            set; matches with ``date >= cutoff_date`` form the holdout.
            Using a named config value (rather than a hardcoded literal)
            allows walk-forward sensitivity analysis without editing any
            function bodies.
    """

    cutoff_date: datetime.date
