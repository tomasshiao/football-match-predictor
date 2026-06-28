BAYESIAN_SEED: int = 42
OPTUNA_SEED: int = 42

_REQUIRED_FEATURE_COLUMNS: list[str] = [
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

_REQUIRED_METRICS: list[str] = [
    "exact_score_accuracy",
    "outcome_accuracy",
    "log_loss",
    "brier_score",
    "home_goals_mae",
    "away_goals_mae",
    "home_goals_rmse",
    "away_goals_rmse",
]