from pathlib import Path
import polars as pl

SAVE_DIR: Path = Path(__file__).resolve().parents[2]

RESULTS_URL: str = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SHOOTOUT_URL: str = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"

BAYESIAN_SEED: int = 42
OPTUNA_SEED: int = 42

REQUIRED_FEATURE_COLUMNS: list[str] = [
    "home_attack_dc",
    "home_defense_dc",
    "away_attack_dc",
    "away_defense_dc",
    "home_form_goals_for",
    "home_form_goals_against",
    "home_form_points",
    "away_form_goals_for",
    "away_form_goals_against",
    "away_form_points",
    "is_neutral",
]

REQUIRED_METRICS: list[str] = [
    "exact_score_accuracy",
    "outcome_accuracy",
    "log_loss",
    "brier_score",
    "home_goals_mae",
    "away_goals_mae",
    "home_goals_rmse",
    "away_goals_rmse",
]

EXPECTED_COLUMNS: dict[str, type[pl.DataType]] = {
    "date":       pl.Date,
    "home_team":  pl.Utf8,
    "away_team":  pl.Utf8,
    "home_score": pl.Int64,
    "away_score": pl.Int64,
    "tournament": pl.Utf8,
    "city":       pl.Utf8,
    "country":    pl.Utf8,
    "neutral":    pl.Boolean,
}

SEP = "─" * 60

COLOURS = {
    "bg":         "#0d1117",
    "surface":    "#161b22",
    "border":     "#30363d",
    "home":       "#2f81f7",
    "home_pens":  "#6aabfc",
    "draw":       "#8b949e",
    "away":       "#f78166",
    "away_pens":  "#f7a898",
    "accent":     "#3fb950",
    "text":       "#e6edf3",
    "text_muted": "#8b949e",
}