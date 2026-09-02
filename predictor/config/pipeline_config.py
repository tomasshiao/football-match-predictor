import datetime
from dataclasses import dataclass
import os
import pathlib
from ..constants.constants import REQUIRED_FEATURE_COLUMNS, REQUIRED_METRICS, BAYESIAN_SEED, OPTUNA_SEED, RESULTS_URL, SAVE_DIR, SHOOTOUT_URL
from .data_config import DataConfig, TeamFilterConfig, WeightingConfig, SplitConfig, get_tournament_weight_table
from .model_configs import DixonColesConfig, BayesianConfig, FeatureConfig, OptunaConfig, XGBoostConfig, EnsembleConfig


def _default_bayesian_cores(chains: int, memory_safe_cap: int = 2) -> int:
    """Pick a ``pm.sample()`` ``cores`` value for the current machine.

    Takes the smallest of three *independent* constraints:

    - ``chains``: no benefit running more worker processes than there are
      chains to sample.
    - The host's visible CPU count: oversubscribing CPUs doesn't sample
      faster, it just adds context-switching overhead for no benefit.
    - ``memory_safe_cap``: an explicit ceiling addressing the actual cause
      of the OOM kill this was built to prevent (see
      ``BayesianConfig.cores``'s docstring) — each concurrent chain holds
      its own compiled PyTensor graph plus ``draws + tune`` samples in
      memory, and CPU count says nothing about available RAM. A machine
      can easily have 8 cores and 4 GB of RAM; auto-detecting CPU count
      alone would have reproduced the original bug on exactly that kind
      of host.
    
    Args:
        chains: Total number of MCMC chains that will be run.
        memory_safe_cap: Hard ceiling independent of CPU count. Raise this
            explicitly once you've confirmed your target host handles it;
            lower it (down to 1, fully sequential) if OOM kills persist.

    Returns:
        A ``cores`` value in ``[1, chains]``.
    """
    configured_cores = os.getenv("BAYESIAN_CORES")
    if configured_cores is not None:
        try:
            return max(1, min(int(configured_cores), chains))
        except ValueError:
            pass

    try:
        available_cpus = len(os.sched_getaffinity(0))  # Linux: cpuset-aware
    except AttributeError:  # sched_getaffinity doesn't exist on macOS/Windows
        available_cpus = os.cpu_count() or 1
    return max(1, min(available_cpus, chains, memory_safe_cap))

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
        _missing = set(REQUIRED_METRICS) - set(self.metrics)
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
        fig_dir: Directory to save generated figures.   
        xgb_home_model_prod: Path to save the production XGBoost home model.
        xgb_away_model_prod: Path to save the production XGBoost away model.
        bayes_idata_prod: Path to save the production Bayesian inference data.
    """
    root_dir: pathlib.Path
    results_cache: pathlib.Path
    fig_dir: pathlib.Path
    xgb_home_model_prod: pathlib.Path
    xgb_away_model_prod: pathlib.Path
    bayes_idata_prod: pathlib.Path
    
    def __post_init__(self) -> None:
        self.ensure_paths_exist()
        assert self.root_dir.exists(), f"Root directory {self.root_dir} does not exist."
        assert self.results_cache.parent.exists(), f"Directory for results_cache {self.results_cache.parent} does not exist."
        assert self.fig_dir.exists(), f"Directory for fig_dir {self.fig_dir} does not exist."
        assert self.xgb_home_model_prod.parent.exists(), f"Directory for xgb_home_model_prod {self.xgb_home_model_prod.parent} does not exist."
        assert self.xgb_away_model_prod.parent.exists(), f"Directory for xgb_away_model_prod {self.xgb_away_model_prod.parent} does not exist."
        assert self.bayes_idata_prod.parent.exists(), f"Directory for bayes_idata_prod {self.bayes_idata_prod.parent} does not exist."
    
    def ensure_paths_exist(self) -> None:
        """
        Ensure that all directories in the path configuration exist.
        If any directory does not exist, it will be created.
        """
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.results_cache.parent.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.xgb_home_model_prod.parent.mkdir(parents=True, exist_ok=True)
        self.xgb_away_model_prod.parent.mkdir(parents=True, exist_ok=True)
        self.bayes_idata_prod.parent.mkdir(parents=True, exist_ok=True)

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


def build_default_pipeline_config(
        home_team_fifa_code: str,
        away_team_fifa_code: str,
        match_date: datetime.date,
        is_neutral_venue: bool,
        is_playoff: bool = False
    ) -> PipelineConfig:
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
            source_url=RESULTS_URL,
            shootout_url=SHOOTOUT_URL,
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
            # min(visible CPUs, chains, memory_safe_cap) — see
            # _default_bayesian_cores's docstring. The memory_safe_cap=2
            # default is the one actually preventing another OOM kill; CPU
            # detection alone wouldn't have caught it. Pass an explicit
            # memory_safe_cap= here instead if 2 is too conservative (more
            # RAM available) or still too high (OOM persists at 2).
            cores=_default_bayesian_cores(chains=8),
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
            feature_columns=list(REQUIRED_FEATURE_COLUMNS),
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
                "n_jobs": 1,
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
            home_team_fifa_code=home_team_fifa_code,
            away_team_fifa_code=away_team_fifa_code,
            match_date=match_date,
            is_neutral_venue=is_neutral_venue,
            is_playoff=is_playoff
        ),
        evaluation=EvaluationConfig(
            max_goals=13,
            metrics=list(REQUIRED_METRICS),
        ),
        paths=PathConfig(
            root_dir=SAVE_DIR,
            results_cache=pathlib.Path(f"{SAVE_DIR}/data/results_cache.csv"),
            fig_dir=pathlib.Path(f"{SAVE_DIR}/figures/{match_date.isoformat()}/{home_team_fifa_code}vs{away_team_fifa_code}"),
            xgb_home_model_prod=pathlib.Path(f"{SAVE_DIR}/models/xgb_home_model_prod.joblib"),
            xgb_away_model_prod=pathlib.Path(f"{SAVE_DIR}/models/xgb_away_model_prod.joblib"),
            bayes_idata_prod=pathlib.Path(f"{SAVE_DIR}/models/bayes_idata_prod.nc")
        ),
    )
