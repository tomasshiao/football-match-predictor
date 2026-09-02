import arviz as az
from dataclasses import dataclass
import numpy as np
import pymc as pm
import polars as pl

from predictor.config.pipeline_config import BayesianConfig

def validate_bayesian_inputs(
    train_df: pl.DataFrame,
    core_teams: list[str],
) -> None:
    """Verify that the Bayesian model inputs are valid before fitting.

    Checks that all required columns are present, that every core team appears
    at least once in the training data (so all attack / defense effects are
    estimable), and confirms the identifiability constraint provided by
    ``pm.ZeroSumNormal``.

    Args:
        train_df:   Training DataFrame (``TRAIN_WEIGHTED_DF``).  Must contain
                    ``home_team``, ``away_team``, ``home_score``,
                    ``away_score``, ``match_weight``, and ``neutral``.
        core_teams: Sorted list of team names in the model universe.

    Raises:
        AssertionError: If required columns are missing or if any core team
            has no appearances in ``train_df``.
    """
    print("── Validation Phase ─────────────────────────────────────────────")

    # 1. Verify all required columns are present
    required_cols = {
        "home_team", "away_team", "home_score", "away_score",
        "match_weight", "neutral",
    }
    missing = required_cols - set(train_df.columns)
    assert not missing, f"Missing required columns for Bayesian model: {missing}"
    print("  ✓ Feature availability verified (all required columns present).")

    # 2. Verify all core teams appear in training data (otherwise attack /
    #    defense effects cannot be estimated)
    home_teams = set(train_df["home_team"].unique().to_list())
    away_teams = set(train_df["away_team"].unique().to_list())
    _unseen_core = set(core_teams) - (home_teams | away_teams)
    assert not _unseen_core, (
        f"Cannot estimate effects: core teams missing from training data: "
        f"{_unseen_core}"
    )
    print("  ✓ Attack and defense effects estimable (all core teams observed).")

    # 3. Identifiability is enforced structurally via pm.ZeroSumNormal priors
    print("  ✓ Priors identifiability guaranteed via pm.ZeroSumNormal constraint.")
    print("── Validation passed ─────────────────────────────────────────────\n")

@dataclass(frozen=True)
class ConvergenceReport:
    """Diagnostic report for PyMC model convergence.

    Attributes:
        max_rhat:      Maximum R-hat statistic across all monitored parameters.
                       Values close to 1.0 indicate chain mixing.
        min_ess_bulk:  Minimum bulk effective sample size across parameters.
                       Reflects sampling efficiency in the bulk of the posterior.
        min_ess_tail:  Minimum tail effective sample size across parameters.
                       Reflects sampling efficiency in the tails.
        n_divergences: Number of divergent transitions during sampling.
                       Any value > 0 signals geometry problems and warrants
                       investigation.
        min_bfmi:      Minimum Bayesian Fraction of Missing Information across
                       chains.  Values below 0.2 suggest the sampler is
                       struggling with the posterior geometry.
        n_chains:      Number of independent MCMC chains that were run.
        passed:        ``True`` when all diagnostics are within the thresholds
                       defined in ``BayesianConfig``.
    """

    max_rhat: float
    min_ess_bulk: float
    min_ess_tail: float
    n_divergences: int
    min_bfmi: float
    n_chains: int
    passed: bool


@dataclass(frozen=True)
class BayesianPosterior:
    """Aggregated posterior means and standard deviations per team.

    Attributes:
        teams:                Ordered list of team names (length K).
        team_index:           Mapping from team name to its position in the
                              posterior arrays.
        attack_mean:          Posterior mean attack strength per team.
        attack_sd:            Posterior standard deviation of attack per team.
        defense_mean:         Posterior mean defense strength per team.
        defense_sd:           Posterior standard deviation of defense per team.
        home_advantage_mean:  Posterior mean of the global home-advantage term.
        intercept_mean:       Posterior mean of the global log-rate intercept.
        convergence:          Convergence diagnostics from ``check_convergence``.
    """

    teams: list[str]
    team_index: dict[str, int]
    attack_mean: dict[str, float]
    attack_sd: dict[str, float]
    defense_mean: dict[str, float]
    defense_sd: dict[str, float]
    home_advantage_mean: float
    intercept_mean: float
    convergence: ConvergenceReport

def build_bayesian_model(
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    is_neutral: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    config: BayesianConfig,
) -> pm.Model:
    """Build the PyMC hierarchical Poisson model graph (no sampling).

    The model mirrors the Dixon-Coles log-rate structure:

        log λ_i = ι + δ·(1 − neutral_i) + a_{h_i} − d_{a_i}
        log μ_i = ι + a_{a_i} − d_{h_i}

    where ``a`` and ``d`` are per-team attack and defense effects constrained
    to sum to zero via ``pm.ZeroSumNormal`` priors (identifiability without
    fixing a reference team).  Weighted likelihood is injected via
    ``pm.Potential`` because PyMC distributions do not natively accept
    per-observation weights.

    Args:
        home_idx:   Integer array of home-team indices (length n_matches).
        away_idx:   Integer array of away-team indices (length n_matches).
        home_goals: Observed home goals (length n_matches).
        away_goals: Observed away goals (length n_matches).
        is_neutral: Boolean array; ``True`` zeroes the home-advantage term.
        weights:    Per-match importance weights (length n_matches).
        n_teams:    Total number of teams in the model universe.
        config:     ``BayesianConfig`` supplying all prior hyperparameters.

    Returns:
        Unsampled ``pm.Model`` object ready to be passed to
        ``fit_bayesian_model``.
    """
    with pm.Model() as model:
        # ── Priors (ZeroSumNormal enforces the sum-to-zero constraint) ────
        attack = pm.ZeroSumNormal(
            "attack", sigma=config.attack_prior_sigma, shape=n_teams
        )
        defense = pm.ZeroSumNormal(
            "defense", sigma=config.defense_prior_sigma, shape=n_teams
        )
        home_advantage = pm.Normal(
            "home_advantage",
            mu=config.home_advantage_prior_mu,
            sigma=config.home_advantage_prior_sigma,
        )
        intercept = pm.Normal("intercept", mu=0.0, sigma=0.5)

        # ── Home-advantage mask (0 at neutral venues) ─────────────────────
        ha_multiplier = 1.0 - is_neutral.astype(np.float64)

        # ── Log goal rates ─────────────────────────────────────────────────
        log_lambda = (
            intercept
            + home_advantage * ha_multiplier
            + attack[home_idx]
            - defense[away_idx]
        )
        log_mu = intercept + attack[away_idx] - defense[home_idx]

        # Clip to prevent NaN gradients from extreme parameter values
        log_lambda = pm.math.clip(log_lambda, -10.0, 10.0)
        log_mu     = pm.math.clip(log_mu,     -10.0, 10.0)

        # ── Deterministics (needed for manual posterior predictive checks) ─
        lam = pm.Deterministic("lambda_obs", pm.math.exp(log_lambda))
        mu  = pm.Deterministic("mu_obs",     pm.math.exp(log_mu))

        # ── Weighted likelihood via Potential ──────────────────────────────
        home_ll = pm.logp(pm.Poisson.dist(mu=lam), home_goals)
        away_ll = pm.logp(pm.Poisson.dist(mu=mu),  away_goals)
        pm.Potential(
            "weighted_likelihood",
            pm.math.sum(weights * (home_ll + away_ll)),
        )

    return model

def fit_bayesian_model(
    model: pm.Model,
    config: BayesianConfig,
) -> az.InferenceData:
    """Sample the PyMC model using NUTS.

    Runs ``pm.sample`` with the parameters defined in ``config``. All
    ``config.chains`` chains are sampled — full statistical validity,
    R-hat/ESS computed the same way regardless — but only ``config.cores``
    of them run simultaneously as separate OS processes, in batches. Lower
    ``config.cores`` trades wall-clock time for peak memory; see
    ``BayesianConfig.cores``'s docstring for why running all chains at
    once by default is a real problem, not just a minor inefficiency.

    NUTS (No-U-Turn Sampler) adapts the step size and trajectory length
    automatically, making it well-suited for the high-dimensional hierarchical
    structure of this model.

    Args:
        model:  Compiled ``pm.Model`` from ``build_bayesian_model``.
        config: ``BayesianConfig`` supplying ``draws``, ``tune``, ``chains``,
                ``cores``, ``target_accept``, and ``random_seed``.

    Returns:
        ``az.InferenceData`` containing posterior samples, sample statistics,
        and log-likelihood values.
    """
    # cores is derived from config here — never from a module-level default
    # argument that would capture the global CFG at definition time.
    with model:
        idata = pm.sample(
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            cores=config.cores,
            target_accept=config.target_accept,
            random_seed=config.random_seed,
            progressbar=True,
        )
    return idata

def check_convergence(
    idata: az.InferenceData,
    config: BayesianConfig,
) -> ConvergenceReport:
    """Compute MCMC convergence diagnostics and return a structured report.

    Evaluates R-hat, bulk and tail ESS, divergence count, and BFMI for the
    four key variable groups (``attack``, ``defense``, ``home_advantage``,
    ``intercept``).  Pass/fail thresholds are read from ``config`` so they
    can be adjusted without editing function bodies.

    Args:
        idata:  ``az.InferenceData`` returned by ``fit_bayesian_model``.
        config: ``BayesianConfig`` supplying ``rhat_threshold`` and
                ``min_ess`` thresholds.

    Returns:
        ``ConvergenceReport`` with all diagnostic values and a single
        ``passed`` flag that is the authoritative convergence verdict.
    """
    summary = az.summary(
        idata,
        var_names=["attack", "defense", "home_advantage", "intercept"],
    )

    max_rhat:     float = float(summary["r_hat"].max())
    min_ess_bulk: float = float(summary["ess_bulk"].min())
    min_ess_tail: float = float(summary["ess_tail"].min())

    n_chains:     int = idata.posterior.dims["chain"]
    n_divergences: int = int(idata.sample_stats["diverging"].sum().item())

    # Energy diagnostics (Bayesian Fraction of Missing Information)
    bfmi_values = az.bfmi(idata)
    min_bfmi:   float = float(bfmi_values.ds["energy"].min().item())

    # Thresholds come from config — never from hardcoded literals
    passed = (
        (max_rhat     <= config.rhat_threshold)
        and (min_ess_bulk >= config.min_ess)
        and (min_ess_tail >= config.min_ess)
        and (n_divergences == 0)
    )

    return ConvergenceReport(
        max_rhat=max_rhat,
        min_ess_bulk=min_ess_bulk,
        min_ess_tail=min_ess_tail,
        n_divergences=n_divergences,
        min_bfmi=min_bfmi,
        n_chains=n_chains,
        passed=passed,
    )

def posterior_predictive_check(
    idata: az.InferenceData,
    observed_home: np.ndarray,
    observed_away: np.ndarray,
    config: BayesianConfig,
) -> dict[str, float]:
    """Simulate goals from posterior rates and compare to observed statistics.

    Draws Poisson samples from the posterior predictive distribution
    (using ``lambda_obs`` and ``mu_obs`` from the InferenceData posterior)
    and compares them to the observed training-set statistics.  A
    well-fitting model should reproduce the observed mean goals per match and
    the proportion of 0-0 draws within a reasonable tolerance.

    Args:
        idata:         ``az.InferenceData`` from ``fit_bayesian_model``.
        observed_home: Array of observed home goals (length n_matches).
        observed_away: Array of observed away goals (length n_matches).
        config:        ``BayesianConfig``; ``random_seed`` is used for
                       the Poisson draw so results are reproducible.

    Returns:
        Dictionary with keys:
          - ``observed_mean_home`` / ``predicted_mean_home``
          - ``observed_mean_away`` / ``predicted_mean_away``
          - ``observed_prop_00``  / ``predicted_prop_00``
    """
    # Extract flattened rate samples shape: (n_draws * n_chains, n_matches)
    lam_samples = az.extract(idata, var_names=["lambda_obs"]).values.T
    mu_samples  = az.extract(idata, var_names=["mu_obs"]).values.T

    # Draw simulated goals — seed from config, never a bare literal
    _rng      = np.random.default_rng(seed=config.random_seed)
    home_sim  = _rng.poisson(lam_samples)
    away_sim  = _rng.poisson(mu_samples)

    obs_00 = np.mean((observed_home == 0) & (observed_away == 0))
    sim_00 = np.mean((home_sim == 0) & (away_sim == 0))

    return {
        "observed_mean_home":  float(observed_home.mean()),
        "predicted_mean_home": float(home_sim.mean()),
        "observed_mean_away":  float(observed_away.mean()),
        "predicted_mean_away": float(away_sim.mean()),
        "observed_prop_00":    float(obs_00),
        "predicted_prop_00":   float(sim_00),
    }

def extract_posterior_means(
    idata: az.InferenceData, 
    teams: list[str], 
    report: ConvergenceReport
) -> BayesianPosterior:
    """
    Collapse posterior samples into means and standard deviations.
    
    This function computes the mean and standard deviation of the posterior distributions for attack and defense strengths, as well as home advantage and intercept terms. It organizes these statistics into a structured dataclass for easy access and further analysis.
    
    Args:
        idata: az.InferenceData - The ArviZ InferenceData object containing posterior samples.
        teams: list[str] - A list of team names.
        report: ConvergenceReport - The convergence report.
    
    Returns:
        BayesianPosterior - A dataclass containing the posterior means and standard deviations.
    """
    # Collapse dimensions over (chain, draw)
    post = idata.posterior
    
    att_mean = post["attack"].mean(dim=("chain", "draw")).values
    att_sd = post["attack"].std(dim=("chain", "draw")).values
    def_mean = post["defense"].mean(dim=("chain", "draw")).values
    def_sd = post["defense"].std(dim=("chain", "draw")).values
    
    ha_mean = float(post["home_advantage"].mean().values)
    int_mean = float(post["intercept"].mean().values)
    
    team_idx = {team: i for i, team in enumerate(teams)}
    
    return BayesianPosterior(
        teams=teams,
        team_index=team_idx,
        attack_mean={t: float(att_mean[i]) for t, i in team_idx.items()},
        attack_sd={t: float(att_sd[i]) for t, i in team_idx.items()},
        defense_mean={t: float(def_mean[i]) for t, i in team_idx.items()},
        defense_sd={t: float(def_sd[i]) for t, i in team_idx.items()},
        home_advantage_mean=ha_mean,
        intercept_mean=int_mean,
        convergence=report
    )
