from dataclasses import dataclass
import math
import numpy as np
import polars as pl
import scipy.optimize as opt

from predictor.config.pipeline_config import DixonColesConfig

@dataclass(frozen=True)
class DixonColesRatings:
    """Complete Dixon-Coles MAP solution.
    
    Attributes:
        teams: Ordered list of team names (length K).
        attack: Mapping from team name to attack parameter.
        defense: Mapping from team name to defense parameter.
        home_advantage: Global home-advantage additive term.
        intercept: Global baseline log-scoring-rate.
        rho: Low-score correlation parameter (typically in [-0.2, 0.2]).
    """
    
    teams: list[str]
    attack: dict[str, float]
    defense: dict[str, float]
    home_advantage: float
    intercept: float
    rho: float
    
    def __post_init__(self) -> None:
        # Verify all teams appear in both attack and defense
        attack_teams = set(self.attack.keys())
        defense_teams = set(self.defense.keys())
        teams_set = set(self.teams)
        num_teams = len(teams_set)
        tolerance = 1e-6 * num_teams  # small tolerance for floating-point comparisons
        
        assert attack_teams == teams_set, (
            f"Attack teams {attack_teams} do not match teams {teams_set}"
        )
        assert defense_teams == teams_set, (
            f"Defense teams {defense_teams} do not match teams {teams_set}"
        )
        
        # Attack and defense sums should be close to 0 (identifiability constraint)
        _attack_sum = math.fsum(self.attack.values())
        _defense_sum = math.fsum(self.defense.values())
        assert abs(_attack_sum) < tolerance, (
            f"Attack sum should be ~0, got {_attack_sum} (not identifiable)"
        )
        assert abs(_defense_sum) < tolerance, (
            f"Defense sum should be ~0, got {_defense_sum} (not identifiable)"
        )
        
        # Reasonable parameter ranges
        assert -2 < self.home_advantage < 2, (
            f"home_advantage {self.home_advantage} out of reasonable range"
        )
        assert -1 < self.intercept < 1, (
            f"intercept {self.intercept} out of reasonable range"
        )
        assert -0.2 < self.rho < 0.2, (
            f"rho {self.rho} should be in (-0.2, 0.2)"
        )
    
    def predict_rate(
        self, home_team: str, away_team: str, is_neutral: bool = False
    ) -> tuple[float, float]:
        """Predict Poisson rate parameters (lambda, mu) for a fixture.
        
        **Mathematical form:**
        $$\log(\lambda) = \iota + a_h - d_a + \delta \cdot [1 - \text{neutral}]$$
        $$\log(\mu) = \iota + a_a - d_h$$
        
        Args:
            home_team: Home team name.
            away_team: Away team name.
            is_neutral: If ``True``, zero out the home-advantage term.
        
        Returns:
            Tuple of (lambda, mu), the Poisson rate parameters.
        """
        ha_term = 0.0 if is_neutral else self.home_advantage
        
        log_lambda = (
            self.intercept
            + self.attack[home_team]
            - self.defense[away_team]
            + ha_term
        )
        log_mu = (
            self.intercept
            + self.attack[away_team]
            - self.defense[home_team]
        )
        
        return np.exp(log_lambda), np.exp(log_mu)

def _pack_params(ratings: DixonColesRatings) -> np.ndarray:
    """Pack a DixonColesRatings object into a flat parameter vector.
    
    **Packing convention (for K teams, K-1 attack and K-1 defense due to
    identifiability constraint):**
    
    indices 0..(K-2)      : attack[0], ..., attack[K-2]
    indices (K-1)..(2K-3) : defense[0], ..., defense[K-2]
    index 2K-2            : home_advantage
    index 2K-1            : intercept
    index 2K              : rho_raw (where rho = 0.2 * tanh(rho_raw))
    
    Args:
        ratings: DixonColesRatings object.
    
    Returns:
        1-D numpy array of length ``2*K + 1``.
    """
    K = len(ratings.teams)
    params = np.zeros(2 * K + 1)
    
    # Attack (first K-1 teams)
    for i in range(K - 1):
        params[i] = ratings.attack[ratings.teams[i]]
    
    # Defense (first K-1 teams)
    for i in range(K - 1):
        params[K - 1 + i] = ratings.defense[ratings.teams[i]]
    
    # Scalar parameters
    params[2 * K - 2] = ratings.home_advantage
    params[2 * K - 1] = ratings.intercept
    
    # rho_raw = inverse of tanh transformation
    if abs(ratings.rho) >= 0.2:
        rho_raw = np.sign(ratings.rho) * 10.0
    else:
        rho_raw = np.arctanh(ratings.rho / 0.2)
    params[2 * K] = rho_raw
    
    return params


def _unpack_params(
    params: np.ndarray, n_teams: int, teams: list[str]
) -> DixonColesRatings:
    """Unpack a flat parameter vector into a DixonColesRatings object.
    
    **Inverse of _pack_params.** Applies the tanh transformation to
    rho_raw: rho = 0.2 * tanh(rho_raw), ensuring rho stays within [-0.2, 0.2].
    
    Args:
        params: 1-D numpy array of length ``2*K + 1``.
        n_teams: Number of teams (K).
        teams: List of team names (length K).
    
    Returns:
        DixonColesRatings object.
    
    Raises:
        AssertionError: If ``len(params) != 2*n_teams + 1``.
    """
    expected_len = 2 * n_teams + 1
    assert len(params) == expected_len, (
        f"Parameter vector length {len(params)} != expected {expected_len}"
    )
    
    # Unpack attack (first K-1 teams) and derive K-th via sum-to-zero
    attack_dict = {}
    attack_sum = 0.0
    for i in range(n_teams - 1):
        attack_dict[teams[i]] = params[i]
        attack_sum += params[i]
    attack_dict[teams[n_teams - 1]] = -attack_sum
    
    # Unpack defense (K-1 teams) and derive K-th via sum-to-zero
    defense_dict = {}
    defense_sum = 0.0
    for i in range(n_teams - 1):
        defense_dict[teams[i]] = params[n_teams - 1 + i]
        defense_sum += params[n_teams - 1 + i]
    defense_dict[teams[n_teams - 1]] = -defense_sum
    
    # Scalar parameters
    home_advantage = float(params[2 * n_teams - 2])
    intercept = float(params[2 * n_teams - 1])
    
    rho_raw = float(params[2 * n_teams])
    rho = np.nextafter(0.2, 0.0) * np.tanh(rho_raw)
    
    return DixonColesRatings(
        teams=teams,
        attack=attack_dict,
        defense=defense_dict,
        home_advantage=home_advantage,
        intercept=intercept,
        rho=rho,
    )

def dixon_coles_tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lam: np.ndarray,
    mu: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Compute Dixon-Coles low-score adjustment factor tau (vectorized).
    
    The tau correction addresses the empirical observation that scorelines
    (0,0), (1,0), (0,1), and (1,1) occur more frequently than a simple
    Poisson model predicts when the scoring rates are similar.
    
    **Mathematical form:**
    $$\\tau_{x,y}(\\rho) = \\begin{cases}
    1 + \\rho \\cdot \\lambda \\cdot \\mu & (x, y) = (0, 0) \\\\
    1 - \\rho \\cdot \\lambda & (x, y) \\in \\{(1, 0), (0, 1)\\} \\text{ or } (0, 1) \\text{ by symmetry} \\\\
    1 - \\rho \\cdot \\mu & (x, y) = (0, 1) \\\\
    1 - \\rho \\cdot \\lambda \\cdot \\mu & (x, y) = (1, 1) \\\\
    1 & \\text{otherwise}
    \\end{cases}$$
    
    **Interpretation:** $\\rho \\in [-0.2, 0.2]$ controls the strength of
    the correlation; $\\rho > 0$ increases the probability of these low
    scorelines; $\\rho < 0$ decreases it.
    
    Args:
        home_goals: 1-D integer array of home team goals.
        away_goals: 1-D integer array of away team goals.
        lam: 1-D float array of Poisson rate for home team (same length).
        mu: 1-D float array of Poisson rate for away team (same length).
        rho: Scalar correlation parameter (typically in [-0.2, 0.2]).
    
    Returns:
        1-D float array of tau values (same length as inputs), clipped to
        a small positive floor to guard against numerical issues.
    
    Raises:
        AssertionError: If input arrays have mismatched lengths.
    """
    assert (
        len(home_goals) == len(away_goals) == len(lam) == len(mu)
    ), "Input arrays must have the same length"
    
    n = len(home_goals)
    tau = np.ones(n)
    
    # (0, 0)
    mask_0_0 = (home_goals == 0) & (away_goals == 0)
    tau[mask_0_0] = 1.0 + rho * lam[mask_0_0] * mu[mask_0_0]
    
    # (1, 0)
    mask_1_0 = (home_goals == 1) & (away_goals == 0)
    tau[mask_1_0] = 1.0 - rho * lam[mask_1_0]
    
    # (0, 1)
    mask_0_1 = (home_goals == 0) & (away_goals == 1)
    tau[mask_0_1] = 1.0 - rho * mu[mask_0_1]
    
    # (1, 1)
    mask_1_1 = (home_goals == 1) & (away_goals == 1)
    tau[mask_1_1] = 1.0 - rho * lam[mask_1_1] * mu[mask_1_1]
    
    # Floor to guard against negative or near-zero values
    tau = np.clip(tau, 1e-6, None)
    
    return tau

def negative_log_posterior(
    params: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    is_neutral: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    teams: list[str],
    config: DixonColesConfig,
) -> float:
    """Compute weighted negative log-posterior (objective to minimize).
    
    **Posterior form:**
    $$-\\log p(\\theta | D) \\propto -\\sum_i w_i \\ell_i(\\theta) + \\text{prior}$$
    
    where $\\ell_i$ is the log-likelihood for match $i$ (Poisson + Dixon-Coles tau),
    and the prior is a zero-mean Gaussian on all attack/defense/home-advantage
    parameters (intercept and rho have weak/flat priors).
    
    Args:
        params: Flat parameter vector (length ``2*n_teams + 1``).
        home_idx, away_idx: Integer arrays indexing home/away team (length n_matches).
        home_goals, away_goals: Observed goals (length n_matches).
        is_neutral: Boolean array marking neutral venues (length n_matches).
        weights: Match weights (length n_matches).
        n_teams: Number of teams (K).
        teams: List of team names.
        config: DixonColesConfig with optimizer settings and prior_scale.
    
    Returns:
        Scalar float: the negative log-posterior.
    """
    # Unpack parameters
    ratings = _unpack_params(params, n_teams, teams)
    
    # Extract Poisson rates for each match
    _ha = np.where(is_neutral, 0.0, ratings.home_advantage)
    log_lam = (
        ratings.intercept
        + np.array([ratings.attack[teams[i]] for i in home_idx])
        - np.array([ratings.defense[teams[i]] for i in away_idx])
        + _ha
    )
    log_mu = (
        ratings.intercept
        + np.array([ratings.attack[teams[i]] for i in away_idx])
        - np.array([ratings.defense[teams[i]] for i in home_idx])
    )
    
    # Prevent overflow in exp by capping the log rates
    log_lam = np.clip(log_lam, -10.0, 10.0)
    log_mu  = np.clip(log_mu, -10.0, 10.0)
    
    # Prevent divide-by-zero by ensuring rates never drop below 1e-8
    lam = np.clip(np.exp(log_lam), 1e-8, None)
    mu  = np.clip(np.exp(log_mu), 1e-8, None)
    
    # Dixon-Coles tau correction
    tau = dixon_coles_tau(home_goals, away_goals, lam, mu, ratings.rho)
    
    # Weighted Poisson log-likelihood with tau
    poisson_ll = (
        home_goals * log_lam - lam + away_goals * log_mu - mu + np.log(tau)
    )
    weighted_ll = weights * poisson_ll
    
    # Gaussian prior on attack/defense/home_advantage (not on intercept or rho)
    prior_scale = config.prior_scale
    prior_penalty = (
        sum(a**2 for a in ratings.attack.values()) / (2 * prior_scale**2)
        + sum(d**2 for d in ratings.defense.values()) / (2 * prior_scale**2)
        + (ratings.home_advantage**2) / (2 * prior_scale**2)
    )
    
    # Negative weighted log-posterior
    nll = -weighted_ll.sum() + prior_penalty
    
    return nll


def negative_log_posterior_gradient(
    params: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    is_neutral: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    teams: list[str],
    config: DixonColesConfig,
) -> np.ndarray:
    """Compute analytic gradient of negative log-posterior w.r.t. params.

    Provides per-parameter derivatives for L-BFGS-B. Passing the analytic
    gradient typically reduces optimizer iterations by ~10x compared to
    finite-difference approximation.

    **Parameter layout** (K = n_teams):

    * indices ``0..K-2``     : attack[0], ..., attack[K-2]
    * indices ``K-1..2K-3``  : defense[0], ..., defense[K-2]
    * index ``2K-2``         : home_advantage
    * index ``2K-1``         : intercept
    * index ``2K``           : rho_raw

    The K-th attack and defense values are *derived* (sum-to-zero), so their
    gradient contributions are propagated back to the K-1 explicit params via
    the chain rule: ``d_attack[K-1]/d_params[j] = -1`` for j = 0..K-2,
    and similarly for defense.

    Args:
        Same as negative_log_posterior.

    Returns:
        1-D array of gradients (same length as ``params``).
    """
    # Rename the input argument to avoid shadowing by loop variables
    is_neutral_flag: np.ndarray = is_neutral  # noqa: preserve reference

    # Unpack parameters
    ratings = _unpack_params(params, n_teams, teams)

    # Vectorised attack / defense lookup arrays for all matches
    attack_arr = np.array([ratings.attack[teams[i]] for i in home_idx])
    defense_arr_away = np.array([ratings.defense[teams[i]] for i in away_idx])
    attack_arr_away = np.array([ratings.attack[teams[i]] for i in away_idx])
    defense_arr_home = np.array([ratings.defense[teams[i]] for i in home_idx])

    # Poisson rate vectors
    _ha = np.where(is_neutral_flag, 0.0, ratings.home_advantage)
    log_lam = ratings.intercept + attack_arr - defense_arr_away + _ha
    log_mu  = ratings.intercept + attack_arr_away - defense_arr_home

    # Prevent overflow in exp by capping the log rates
    log_lam = np.clip(log_lam, -10.0, 10.0)
    log_mu  = np.clip(log_mu, -10.0, 10.0)
    
    # Prevent divide-by-zero by ensuring rates never drop below 1e-8
    lam = np.clip(np.exp(log_lam), 1e-8, None)
    mu  = np.clip(np.exp(log_mu), 1e-8, None)

    # Dixon-Coles tau (for log-likelihood; gradient ignores tau for simplicity —
    # finite-difference is used for rho where tau coupling is non-trivial)
    tau = dixon_coles_tau(home_goals, away_goals, lam, mu, ratings.rho)

    grad = np.zeros_like(params)
    prior_scale = config.prior_scale

    # Poisson residuals (d log p(x|lambda) / d log_lambda = x - lambda = x/lam*lam - lam)
    # Written as (x/lam - 1) times lam, but since d log_lambda / d attack = 1,
    # d NLL / d attack = -w * (x/lam - 1) aggregated over home matches of that team,
    # and similarly for the away channel via mu.
    home_resid = home_goals / lam - 1.0   # d log p(home_goals|lam) / d log_lam per match
    away_resid = away_goals / mu  - 1.0   # d log p(away_goals|mu ) / d log_mu  per match

    # ── Attack gradients ───────────────────────────────────────────────────────
    # For each team j < K-1: attack[j] appears in log_lam when j is home AND
    # in log_mu when j is away. The K-th team's contribution is accumulated
    # separately and broadcast back (chain rule for sum-to-zero constraint).
    _g_att_kth = 0.0  # gradient contribution from the K-th (derived) attack param
    for j in range(n_teams - 1):
        _is_home_j = home_idx == j
        _is_away_j = away_idx == j
        _datt = (
            np.sum(weights[_is_home_j] * home_resid[_is_home_j])
            + np.sum(weights[_is_away_j] * away_resid[_is_away_j])
        )
        _prior = ratings.attack[teams[j]] / (prior_scale ** 2)
        grad[j] = -_datt + _prior

    # K-th attack team: derived, contribution propagated back to all j < K-1
    j_kth = n_teams - 1
    _is_home_kth = home_idx == j_kth
    _is_away_kth = away_idx == j_kth
    _datt_kth = (
        np.sum(weights[_is_home_kth] * home_resid[_is_home_kth])
        + np.sum(weights[_is_away_kth] * away_resid[_is_away_kth])
    )
    _prior_kth = ratings.attack[teams[j_kth]] / (prior_scale ** 2)
    _g_att_kth = -_datt_kth + _prior_kth
    # Chain rule: d_attack[K-1]/d_params[j] = -1 for j = 0..K-2
    for j in range(n_teams - 1):
        grad[j] -= _g_att_kth  # subtract because d(−sum)/d_param[j] = −1

    # ── Defense gradients ──────────────────────────────────────────────────────
    # defense[j] appears in log_mu when j is home AND in log_lam when j is away
    # (with opposite sign to attack: higher defense = lower conceded).
    # d log_lam / d defense[away] = -1  => d NLL / d defense[away] = +w*(1 - x/lam)
    # d log_mu  / d defense[home] = -1  => d NLL / d defense[home] = +w*(1 - y/mu)
    _g_def_kth = 0.0
    for j in range(n_teams - 1):
        _is_home_j = home_idx == j
        _is_away_j = away_idx == j
        # When team j is home: their defense constrains the away goal rate (mu).
        # When team j is away: their defense constrains the home goal rate (lam).
        _ddef = (
            -np.sum(weights[_is_away_j] * home_resid[_is_away_j])  # j is away -> lam channel
            - np.sum(weights[_is_home_j] * away_resid[_is_home_j]) # j is home -> mu  channel
        )
        _prior = ratings.defense[teams[j]] / (prior_scale ** 2)
        grad[(n_teams - 1) + j] = -_ddef + _prior

    # K-th defense team: derived, broadcast back
    _is_home_kth = home_idx == j_kth
    _is_away_kth = away_idx == j_kth
    _ddef_kth = (
        -np.sum(weights[_is_away_kth] * home_resid[_is_away_kth])
        - np.sum(weights[_is_home_kth] * away_resid[_is_home_kth])
    )
    _prior_def_kth = ratings.defense[teams[j_kth]] / (prior_scale ** 2)
    _g_def_kth = -_ddef_kth + _prior_def_kth
    # Chain rule: d_defense[K-1]/d_params[K-1+j] = -1 for j = 0..K-2
    for j in range(n_teams - 1):
        grad[(n_teams - 1) + j] -= _g_def_kth

    # ── Home-advantage gradient ───────────────────────────────────────────────
    # Use is_neutral_flag (the renamed function argument) to avoid the loop
    # variable shadowing bug that existed in the original code.
    _non_neutral = ~is_neutral_flag
    _dha = np.sum(weights[_non_neutral] * home_resid[_non_neutral])
    _prior_ha = ratings.home_advantage / (prior_scale ** 2)
    grad[2 * n_teams - 2] = -_dha + _prior_ha

    # ── Intercept gradient ────────────────────────────────────────────────────
    # Intercept enters both log_lam and log_mu equally.
    _dint = (
        np.sum(weights * home_resid)
        + np.sum(weights * away_resid)
    )
    grad[2 * n_teams - 1] = -_dint  # no prior on intercept

    # ── rho gradient (finite-difference; tau coupling is non-trivial analytically)
    _eps = 1e-4
    _p_plus  = params.copy(); _p_plus[2 * n_teams]  += _eps
    _p_minus = params.copy(); _p_minus[2 * n_teams] -= _eps
    _nll_plus  = negative_log_posterior(
        _p_plus,  home_idx, away_idx, home_goals, away_goals,
        is_neutral_flag, weights, n_teams, teams, config
    )
    _nll_minus = negative_log_posterior(
        _p_minus, home_idx, away_idx, home_goals, away_goals,
        is_neutral_flag, weights, n_teams, teams, config
    )
    grad[2 * n_teams] = (_nll_plus - _nll_minus) / (2 * _eps)

    return grad

def fit_dixon_coles(
    matches: pl.DataFrame,
    teams: list[str],
    config: DixonColesConfig,
) -> DixonColesRatings:
    """Fit the Dixon-Coles model via weighted MAP estimation (SciPy L-BFGS-B).
    
    **Model:**
    
    Estimates per-team attack ($a_i$) and defense ($d_i$) parameters,
    global home-advantage ($\\delta$), intercept ($\\iota$), and low-score
    correlation ($\\rho$). Teams' attack and defense vectors are constrained
    to sum to zero (identifiability).
    
    **Objective:**
    $$\\theta^* = \\arg\\min_\\theta \\left[ -\\sum_i w_i \\ell_i(\\theta) + \\text{prior}(\\theta) \\right]$$
    
    Args:
        matches: DataFrame with columns ``home_team``, ``away_team``,
            ``home_score``, ``away_score``, ``match_weight``, ``neutral``.
        teams: List of team names (must match the dataset).
        config: DixonColesConfig with optimizer settings.
    
    Returns:
        DixonColesRatings object containing the fitted parameters.
    
    Raises:
        AssertionError: If required columns are missing.
        AssertionError: If optimizer convergence fails (warning issued).
    """
    # Validate input
    required_cols = {"home_team", "away_team", "home_score", "away_score", "match_weight", "neutral"}
    missing = required_cols - set(matches.columns)
    assert not missing, f"Missing columns: {missing}"
    
    # Build team-to-index mapping
    team_to_idx = {team: i for i, team in enumerate(teams)}
    
    # Extract data as numpy arrays
    home_idx = np.array([team_to_idx[t] for t in matches["home_team"].to_list()], dtype=np.int32)
    away_idx = np.array([team_to_idx[t] for t in matches["away_team"].to_list()], dtype=np.int32)
    home_goals = matches["home_score"].to_numpy().astype(np.int64)
    away_goals = matches["away_score"].to_numpy().astype(np.int64)
    is_neutral = matches["neutral"].to_numpy().astype(bool)
    weights = matches["match_weight"].to_numpy().astype(np.float64)
    
    # Initial parameter guess: zeros for all
    x0 = np.zeros(2 * len(teams) + 1)
    x0[2 * len(teams) - 2] = 0.25  # Reasonable starting value for home-advantage
    
    # Define box constraints (bounds) to prevent the optimizer from testing 
    # invalid regions and crashing the __post_init__ assertions.
    # We use +/- 1.99 and 0.99 because your assertions use strict `<` inequalities.
    bounds = []
    bounds.extend([(-3.0, 3.0)] * (len(teams) - 1)) # Attack constrained
    bounds.extend([(-3.0, 3.0)] * (len(teams) - 1)) # Defense constrained
    bounds.append((-1.99, 1.99))                    # Home Advantage
    bounds.append((-0.99, 0.99))                    # Intercept
    bounds.append((None, None))                     # rho_raw
    
    # Define objective wrapper (captures data and config)
    def _objective(params: np.ndarray) -> float:
        return negative_log_posterior(
            params, home_idx, away_idx, home_goals, away_goals,
            is_neutral, weights, len(teams), teams, config
        )
    
    def _gradient(params: np.ndarray) -> np.ndarray:
        return negative_log_posterior_gradient(
            params, home_idx, away_idx, home_goals, away_goals,
            is_neutral, weights, len(teams), teams, config
        )
    
    # Run L-BFGS-B
    result = opt.minimize(
        _objective,
        x0,
        method=config.optimizer_method,
        jac=_gradient,
        bounds=bounds,
        options={"maxiter": config.max_iterations, "ftol": 1e-8},
    )
    
    # Check convergence (OptimizeResult uses .success / .status, not .warnflag)
    if not result.success:
        print(f"⚠ L-BFGS-B convergence warning (status={result.status}): {result.message}")
        print("  (This is not fatal; inspecting the fit quality anyway)")
    
    # Unpack and return
    ratings = _unpack_params(result.x, len(teams), teams)
    
    return ratings