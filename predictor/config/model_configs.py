from dataclasses import dataclass, field
from ..constants.config_constants import _REQUIRED_FEATURE_COLUMNS

# --- DixonColesConfig --------------------------------
@dataclass(frozen=True)
class DixonColesConfig:
    """Configuration for the Dixon-Coles MAP optimizer.

    Args:
        optimizer_method: SciPy minimization method.  Must be ``'L-BFGS-B'``
            to support box constraints and an analytic gradient.
        max_iterations: Maximum iterations passed to the L-BFGS-B solver.
        prior_scale: Standard deviation of the zero-mean Gaussian prior
            placed on each attack and defense parameter.  Acts as an L2
            regularizer; larger values allow more extreme estimates for
            data-sparse teams.
    """

    optimizer_method: str
    max_iterations: int
    prior_scale: float

    def __post_init__(self) -> None:
        assert self.optimizer_method == "L-BFGS-B", (
            f"optimizer_method must be 'L-BFGS-B', got {self.optimizer_method!r}"
        )
        assert self.max_iterations > 0, "max_iterations must be positive"
        assert self.prior_scale > 0, "prior_scale must be positive"


# --- BayesianConfig --------------------------------
@dataclass(frozen=True)
class BayesianConfig:
    """Configuration for the PyMC / NUTS hierarchical model.

    Args:
        draws: Number of post-tuning NUTS samples per chain.
        tune: Number of tuning (warm-up) steps per chain.
        chains: Number of independent Markov chains.
        target_accept: Target Metropolis-Hastings acceptance rate for the
            dual-averaging step-size adaptor (NUTS).
        attack_prior_sigma: Scale of the ZeroSumNormal prior on attack
            parameters.
        defense_prior_sigma: Scale of the ZeroSumNormal prior on defense
            parameters.
        home_advantage_prior_mu: Prior mean for the home-advantage term.
        home_advantage_prior_sigma: Prior standard deviation for the
            home-advantage term.
        random_seed: Seed passed to ``pm.sample`` for reproducibility.
        rhat_threshold: Maximum acceptable R-hat value for convergence.
            Typically 1.01 (strict) or 1.05 (lenient).
        min_ess: Minimum acceptable effective sample size for convergence.
    """

    draws: int
    tune: int
    chains: int
    target_accept: float
    attack_prior_sigma: float
    defense_prior_sigma: float
    home_advantage_prior_mu: float
    home_advantage_prior_sigma: float
    random_seed: int
    rhat_threshold: float
    min_ess: int

    def __post_init__(self) -> None:
        assert self.draws > 0, "draws must be positive"
        assert self.tune > 0, "tune must be positive"
        assert 4 <= self.chains <= 8, "chains must be in [4, 8]: R-hat is undefined with 1 chain and unreliable with 2; the standard minimum is 4."
        assert 0 < self.target_accept < 1, "target_accept must be in (0, 1)"
        assert self.attack_prior_sigma > 0
        assert self.defense_prior_sigma > 0
        assert self.home_advantage_prior_sigma > 0
        assert 1.0 < self.rhat_threshold <= 1.1, "rhat_threshold must be in (1.0, 1.1]"
        assert self.min_ess > 0

# --- FeatureConfig --------------------------------
@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for XGBoost feature engineering.

    Args:
        rolling_window: Number of preceding matches used to compute
            each team's rolling form statistics.
        feature_columns: Ordered list of column names fed to XGBoost.
            This is the **single source of truth** used in both training
            and at prediction time; it must never be manually re-typed
            at a call site.
    """

    rolling_window: int
    feature_columns: list[str]

    def __post_init__(self) -> None:
        assert self.rolling_window >= 3, "rolling_window must be at least 3"
        _missing = set(_REQUIRED_FEATURE_COLUMNS) - set(self.feature_columns)
        assert not _missing, (
            f"FeatureConfig.feature_columns is missing required features: {_missing}"
        )

# --- OptunaConfig --------------------------------
@dataclass(frozen=True)
class OptunaConfig:
    """Configuration for the Optuna hyperparameter search.

    Args:
        n_trials: Total number of Optuna trials.
        timeout_seconds: Wall-clock time limit in seconds.  ``None``
            means no time limit; the search runs for exactly ``n_trials``.
        cv_folds: Number of expanding-window folds used inside each trial.
        min_train_fraction: Folds whose training fraction is below this
            threshold are skipped to prevent very small training windows.
        sampler_seed: Seed for the TPE sampler, ensuring reproducible
            trial sequences.
    """

    n_trials: int
    timeout_seconds: int | None
    cv_folds: int
    min_train_fraction: float
    sampler_seed: int

    def __post_init__(self) -> None:
        assert self.n_trials > 0, "n_trials must be positive"
        assert self.cv_folds >= 2, "cv_folds must be at least 2"
        assert 0.0 < self.min_train_fraction < 1.0, (
            "min_train_fraction must be in (0, 1)"
        )
        if self.timeout_seconds is not None:
            assert self.timeout_seconds > 0, "timeout_seconds must be positive"

# --- XGBoostConfig --------------------------------
@dataclass(frozen=True)
class XGBoostConfig:
    """Configuration for XGBoost Poisson regressors.

    Args:
        fixed_params: Hyperparameters that are *not* tuned by Optuna.
            Must include ``objective='count:poisson'`` and
            ``random_state``.
        search_space: Optuna search bounds for tunable hyperparameters.
            Each value is a ``(low, high)`` tuple of the same dtype as
            the parameter (``float`` for continuous, ``int`` for discrete).
    """

    fixed_params: dict[str, float | int | str]
    search_space: dict[str, tuple[float, float] | tuple[int, int]]

    def __post_init__(self) -> None:
        assert self.fixed_params.get("objective") == "count:poisson", (
            "fixed_params must include objective='count:poisson'"
        )
        assert "random_state" in self.fixed_params, (
            "fixed_params must include random_state"
        )
        _required_space = {
            "n_estimators", "max_depth", "learning_rate",
            "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
        }
        _missing = _required_space - self.search_space.keys()
        assert not _missing, (
            f"XGBoostConfig.search_space is missing required keys: {_missing}"
        )

# --- EnsembleConfig --------------------------------
@dataclass(frozen=True)
class EnsembleConfig:
    """Configuration for the ensemble matrix combination.

    Args:
        temperature: Softmax temperature applied to negative log-loss
            values when deriving model weights from backtest performance.
            Lower temperature concentrates weight on the best model;
            higher temperature gives a more uniform blend.
        weights: Pre-computed model weights keyed by model name.  Starts
            as ``None`` and is populated by ``compute_ensemble_weights``
            after the backtest.  The frozen dataclass is replaced with
            an updated instance at that point.
    """

    temperature: float
    weights: dict[str, float] | None = field(default=None)

    def __post_init__(self) -> None:
        assert self.temperature > 0, "temperature must be positive"
        if self.weights is not None:
            _total = sum(self.weights.values())
            assert abs(_total - 1.0) < 1e-6, (
                f"EnsembleConfig.weights must sum to 1.0, got {_total}"
            )
