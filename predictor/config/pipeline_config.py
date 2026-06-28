import datetime
from dataclasses import dataclass
import pathlib
from ..constants.config_constants import _REQUIRED_FEATURE_COLUMNS, _REQUIRED_METRICS, BAYESIAN_SEED, OPTUNA_SEED
from ..constants.global_constants import SAVE_DIR
from .data_config import DataConfig, TeamFilterConfig, WeightingConfig, SplitConfig, get_tournament_weight_table
from .model_configs import DixonColesConfig, BayesianConfig, FeatureConfig, OptunaConfig, XGBoostConfig, EnsembleConfig

# --- FixtureConfig --------------------------------
@dataclass(frozen=True)
class FixtureConfig:
    """The single match being predicted.

    Teams are stored as FIFA three-letter codes here and resolved to
    dataset team-name strings in Section 3 (Team Registry).

    Args:
        home_team_fifa_code: FIFA code of the nominally "home" team
            (used for feature assignment even at neutral venues).
        away_team_fifa_code: FIFA code of the "away" team.
        match_date: Date on which the match will be / was played.
        is_neutral_venue: ``True`` when the match is played at a
            neutral location (home-advantage term is zeroed out).
    """

    home_team_fifa_code: str
    away_team_fifa_code: str
    match_date: datetime.date
    is_neutral_venue: bool
    is_playoff: bool = False

    def __post_init__(self) -> None:
        assert len(self.home_team_fifa_code) == 3, (
            f"home_team_fifa_code must be 3 characters, got {self.home_team_fifa_code!r}"
        )
        assert len(self.away_team_fifa_code) == 3, (
            f"away_team_fifa_code must be 3 characters, got {self.away_team_fifa_code!r}"
        )
        assert self.home_team_fifa_code != self.away_team_fifa_code, (
            "home and away FIFA codes must differ"
        )

# --- EvaluationConfig --------------------------------
@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for backtest evaluation.

    Args:
        max_goals: Maximum number of goals per team represented in the
            scoreline probability matrix.  Scorelines beyond this bound
            are collapsed into the boundary bin during evaluation.
        metrics: List of metric names to compute.  Must include all
            entries in ``_REQUIRED_METRICS``.
    """

    max_goals: int
    metrics: list[str]

    def __post_init__(self) -> None:
        assert self.max_goals >= 5, "max_goals must be at least 5"
        _missing = set(_REQUIRED_METRICS) - set(self.metrics)
        assert not _missing, (
            f"EvaluationConfig.metrics is missing required entries: {_missing}"
        )

# --- PathConfig --------------------------------
@dataclass(frozen=True)
class PathConfig:
    """
    Configuration for save paths.
    
    Args:
        root_dir: Root directory for all output files.
        results_cache: Path to the cached results CSV file.
        xgb_home_model_prod: Path to save the production XGBoost home model.
        xgb_away_model_prod: Path to save the production XGBoost away model.
        bayes_idata_prod: Path to save the production Bayesian inference data.
    """
    root_dir: pathlib.Path
    results_cache: pathlib.Path
    xgb_home_model_prod: pathlib.Path
    xgb_away_model_prod: pathlib.Path
    bayes_idata_prod: pathlib.Path
    
    def __post_init__(self) -> None:
        assert self.root_dir.exists(), f"Root directory {self.root_dir} does not exist."
        assert self.results_cache.parent.exists(), f"Directory for results_cache {self.results_cache.parent} does not exist."
        assert self.xgb_home_model_prod.parent.exists(), f"Directory for xgb_home_model_prod {self.xgb_home_model_prod.parent} does not exist."
        assert self.xgb_away_model_prod.parent.exists(), f"Directory for xgb_away_model_prod {self.xgb_away_model_prod.parent} does not exist."
        assert self.bayes_idata_prod.parent.exists(), f"Directory for bayes_idata_prod {self.bayes_idata_prod.parent} does not exist."

# --- PipelineConfig --------------------------------
@dataclass(frozen=True)
class PipelineConfig:
    """Top-level frozen configuration for the entire prediction pipeline.

    Composed of twelve nested sub-config objects.  This is the **only**
    object passed to orchestration functions in subsequent sections.
    No global mutable state is used anywhere in the notebook.

    Args:
        data: Raw data acquisition settings.
        team_filter: Core-team universe selection.
        weighting: Per-match time-decay and tournament-importance weights.
        split: Chronological train/test boundary.
        dixon_coles: Dixon-Coles MAP optimizer settings.
        bayesian: PyMC / NUTS hierarchical model settings.
        features: XGBoost feature engineering settings.
        optuna: Hyperparameter search settings.
        xgboost: XGBoost model settings.
        ensemble: Scoreline matrix combination settings.
        fixture: The headline fixture to predict.
        evaluation: Backtest evaluation settings.
    """

    prod_ref_date: datetime.date
    data: DataConfig
    team_filter: TeamFilterConfig
    weighting: WeightingConfig
    split: SplitConfig
    dixon_coles: DixonColesConfig
    bayesian: BayesianConfig
    features: FeatureConfig
    optuna: OptunaConfig
    xgboost: XGBoostConfig
    ensemble: EnsembleConfig
    fixture: FixtureConfig
    evaluation: EvaluationConfig
    paths: PathConfig


def build_default_pipeline_config() -> PipelineConfig:
    """Construct the default PipelineConfig with documented defaults.

    All tunable values are set here.  No magic numbers appear anywhere
    else in the notebook; modify this function to change pipeline behaviour.

    Returns:
        A fully validated, frozen PipelineConfig instance.
    """
    _weight_table = get_tournament_weight_table()

    return PipelineConfig(
        prod_ref_date=datetime.date.today(),
        data=DataConfig(
            source_url=(
                "https://raw.githubusercontent.com/martj42/"
                "international_results/master/results.csv"
            ),
            shootout_url=(
                "https://raw.githubusercontent.com/martj42/"
                "international_results/master/shootouts.csv"
            ),
            cache_path=pathlib.Path(f"{SAVE_DIR}/data/results_cache.csv"),
            start_date=datetime.date(2018, 1, 1),
        ),
        team_filter=TeamFilterConfig(min_matches=10),
        weighting=WeightingConfig(
            half_life_days=365 * 3,   # 3-year half-life
            tournament_weight_table=_weight_table,
            default_tournament_weight=1.0,
        ),
        split=SplitConfig(cutoff_date=datetime.date(2024, 1, 1)),
        dixon_coles=DixonColesConfig(
            optimizer_method="L-BFGS-B",
            max_iterations=2000,
            prior_scale=1.0,
        ),
        bayesian=BayesianConfig(
            draws=1000,
            tune=1000,
            chains=8,
            target_accept=0.9,
            attack_prior_sigma=0.5,
            defense_prior_sigma=0.5,
            home_advantage_prior_mu=0.1,
            home_advantage_prior_sigma=0.3,
            random_seed=BAYESIAN_SEED,
            rhat_threshold=1.05,
            min_ess=200,
        ),
        features=FeatureConfig(
            rolling_window=10,
            feature_columns=list(_REQUIRED_FEATURE_COLUMNS),
        ),
        optuna=OptunaConfig(
            n_trials=60,
            timeout_seconds=300,
            cv_folds=4,
            min_train_fraction=0.3,
            sampler_seed=OPTUNA_SEED,
        ),
        xgboost=XGBoostConfig(
            fixed_params={
                "objective": "count:poisson",
                "random_state": OPTUNA_SEED,
                "tree_method": "hist",
                "eval_metric": "poisson-nloglik",
            },
            search_space={
                "n_estimators": (100, 600),
                "max_depth": (3, 8),
                "learning_rate": (0.01, 0.3),
                "subsample": (0.6, 1.0),
                "colsample_bytree": (0.6, 1.0),
                "reg_alpha": (0.0, 5.0),
                "reg_lambda": (0.5, 5.0),
            },
        ),
        ensemble=EnsembleConfig(temperature=1.0),
        fixture=FixtureConfig(
            home_team_fifa_code=HOME_TEAM_FIFA_CODE,
            away_team_fifa_code=AWAY_TEAM_FIFA_CODE,
            match_date=MATCH_DATE,
            is_neutral_venue=IS_NEUTRAL_VENUE,
            is_playoff=IS_PLAYOFF
        ),
        evaluation=EvaluationConfig(
            max_goals=13,
            metrics=list(_REQUIRED_METRICS),
        ),
        paths=PathConfig(
            root_dir=SAVE_DIR,
            results_cache=pathlib.Path(f"{SAVE_DIR}/data/results_cache.csv"),
            xgb_home_model_prod=pathlib.Path(f"{SAVE_DIR}/models/xgb_home_model_prod.joblib"),
            xgb_away_model_prod=pathlib.Path(f"{SAVE_DIR}/models/xgb_away_model_prod.joblib"),
            bayes_idata_prod=pathlib.Path(f"{SAVE_DIR}/models/bayes_idata_prod.nc")
        ),
    )
