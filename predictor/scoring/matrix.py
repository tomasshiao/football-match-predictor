import arviz as az
import numpy as np
from scipy import stats

from predictor.models.dixon_coles import dixon_coles_tau
from predictor.models.bayesian import BayesianPosterior

def poisson_score_matrix(
    lam: float,
    mu: float,
    max_goals: int,
    rho: float = 0.0,
) -> np.ndarray:
    """Build a single (max_goals+1, max_goals+1) scoreline probability matrix.

    Computes P(home=i, away=j) for all (i,j) in [0..max_goals]².  When
    ``rho != 0`` the Dixon-Coles low-score correction (``tau``) is applied
    to the four cells (0,0), (0,1), (1,0), (1,1) to capture the empirical
    over-frequency of low-scoring draws and near-draws.

    The matrix is renormalized after the tau adjustment so that it sums to
    exactly 1.0; without renormalization the probability mass truncated at
    ``max_goals`` and the tau perturbation can shift the total slightly.

    Args:
        lam:       Expected home goals (Poisson rate λ).
        mu:        Expected away goals (Poisson rate μ).
        max_goals: Maximum goals per team represented in the matrix.
                   Rows and columns run from 0 to ``max_goals`` inclusive,
                   giving a (max_goals+1, max_goals+1) output.
        rho:       Dixon-Coles low-score correlation parameter.  Pass
                   ``DC_RATINGS_BACKTEST.rho`` for DC matrices; pass
                   ``0.0`` for Bayesian and XGBoost matrices (those models
                   do not estimate rho).

    Returns:
        2-D ``np.ndarray`` of shape ``(max_goals+1, max_goals+1)``,
        dtype float64, summing to 1.0.
    """
    k = max_goals + 1
    goals = np.arange(k)

    # Independent Poisson PMFs for each team
    home_pmf = stats.poisson.pmf(goals, lam)   # shape (k,)
    away_pmf = stats.poisson.pmf(goals, mu)    # shape (k,)

    # Outer product gives the independent-Poisson joint probability
    matrix = np.outer(home_pmf, away_pmf)      # shape (k, k)

    # Dixon-Coles tau correction for the four low-score cells
    if rho != 0.0:
        tau_00 = dixon_coles_tau(
            np.array([0]), np.array([0]),
            np.array([lam]), np.array([mu]), rho
        )[0]
        tau_01 = dixon_coles_tau(
            np.array([0]), np.array([1]),
            np.array([lam]), np.array([mu]), rho
        )[0]
        tau_10 = dixon_coles_tau(
            np.array([1]), np.array([0]),
            np.array([lam]), np.array([mu]), rho
        )[0]
        tau_11 = dixon_coles_tau(
            np.array([1]), np.array([1]),
            np.array([lam]), np.array([mu]), rho
        )[0]
        matrix[0, 0] *= tau_00
        matrix[0, 1] *= tau_01
        matrix[1, 0] *= tau_10
        matrix[1, 1] *= tau_11

    # Renormalize: mass outside [0..max_goals]² and tau perturbation may
    # push the total slightly away from 1.0.
    total = matrix.sum()
    if total > 0.0:
        matrix /= total

    return matrix


def batch_poisson_score_matrices(
    lam: np.ndarray,
    mu: np.ndarray,
    max_goals: int,
    rho: float = 0.0,
) -> np.ndarray:
    """Build a stack of scoreline matrices for many fixtures at once.

    Vectorized wrapper around ``poisson_score_matrix`` for the backtest
    scoring stage (Section 11) where matrices are needed for every test row.
    The tau correction is applied per-fixture using the same ``rho`` for all
    fixtures; this is correct because the DC model fits a single global rho.

    Args:
        lam:       1-D array of home goal rates, shape (n,).
        mu:        1-D array of away goal rates, shape (n,).
        max_goals: Maximum goals per team (see ``poisson_score_matrix``).
        rho:       Low-score correlation; pass ``DC_RATINGS_BACKTEST.rho``
                   for DC matrices, ``0.0`` for Bayesian/XGBoost.

    Returns:
        3-D ``np.ndarray`` of shape ``(n, max_goals+1, max_goals+1)``,
        where ``output[i]`` is the scoreline matrix for fixture ``i``.
    """
    n = len(lam)
    k = max_goals + 1
    out = np.empty((n, k, k), dtype=np.float64)
    for i in range(n):
        out[i] = poisson_score_matrix(lam[i], mu[i], max_goals, rho)
    return out

# ── Bayesian test matrices ─────────────────────────────────────────────────
# Use the posterior-mean attack/defense parameters from BAYES_POSTERIOR_BACKTEST
# to compute expected goal rates.  This is an accepted tractability approximation
# for the backtest (full posterior-predictive averaging is used only in Section 15
# for the headline fixture).  No rho correction — the Bayesian model does not
# estimate rho.

def bayesian_rate_means_batch(
    posterior: BayesianPosterior,
    home_teams: list[str],
    away_teams: list[str],
    is_neutral: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute posterior-mean Poisson rates for a batch of fixtures.

    Applies the same log-rate formula as the PyMC model (see Section 8),
    but substitutes posterior means for the sampled parameters:

        log λ_i = ι_mean + a_mean[home_i] − d_mean[away_i] + δ_mean·(1 − neutral_i)
        log μ_i = ι_mean + a_mean[away_i] − d_mean[home_i]

    This plug-in approximation ignores posterior uncertainty; it is
    appropriate for backtest scoring where we want a single summary
    prediction per test fixture rather than a full predictive distribution.

    Args:
        posterior:   ``BayesianPosterior`` from ``extract_posterior_means``;
                     must have been fit on the backtest training window.
        home_teams:  List of home team names aligned with the test rows.
        away_teams:  List of away team names aligned with the test rows.
        is_neutral:  Boolean array, length n_test.  ``True`` zeroes the
                     home-advantage term for neutral-venue fixtures.

    Returns:
        Two 1-D ``np.ndarray`` arrays ``(lambda_bayes, mu_bayes)`` of
        shape ``(n_test,)``, clipped to ``[1e-8, ∞)``.
    """
    ha  = np.where(is_neutral, 0.0, posterior.home_advantage_mean)
    iota = posterior.intercept_mean

    log_lam = np.array([
        iota + posterior.attack_mean[h] - posterior.defense_mean[a] + ha[i]
        for i, (h, a) in enumerate(zip(home_teams, away_teams))
    ])
    log_mu = np.array([
        iota + posterior.attack_mean[a] - posterior.defense_mean[h]
        for h, a in zip(home_teams, away_teams)
    ])

    return np.clip(np.exp(log_lam), 1e-8, None), np.clip(np.exp(log_mu), 1e-8, None)

def bayesian_rate_samples(
    idata:       az.InferenceData,
    team_index:  dict[str, int],
    home_team:   str,
    away_team:   str,
    is_neutral:  bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw full posterior samples of (lambda, mu) for a single fixture.

    Extracts the NUTS posterior samples of attack, defense, home_advantage,
    and intercept from an InferenceData object, then computes (lambda, mu)
    for the requested fixture for every sample, returning two 1-D arrays
    of length (n_chains * n_draws).

    This is used only for the PRODUCTION prediction (not the backtest),
    where we want genuine posterior-predictive uncertainty rather than the
    plug-in approximation used in Section 11.4.

    Args:
        idata:       az.InferenceData from fit_bayesian_model (BAYES_IDATA_PROD).
        team_index:  Mapping from team name to integer index in CORE_TEAMS.
        home_team:   Dataset name of the home team.
        away_team:   Dataset name of the away team.
        is_neutral:  True zeros out the home-advantage term.

    Returns:
        Two 1-D np.ndarray arrays (lam_samples, mu_samples) of shape
        (n_samples,), each clipped to [1e-8, inf).
    """
    # Extract flat arrays of posterior samples for each global/per-team parameter.
    # az.extract collapses the (chain, draw) dimensions into a single 'sample' dim.
    post = idata.posterior

    # attack[team_index] and defense[team_index] are per-team; extract the
    # specific team's samples across all chains and draws.
    _h_idx = team_index[home_team]
    _a_idx = team_index[away_team]

    # Shape: (n_chains * n_draws,) after stacking
    attack_h  = post["attack"].values[:, :, _h_idx].ravel()   # home attack samples
    attack_a  = post["attack"].values[:, :, _a_idx].ravel()   # away attack samples
    defense_h = post["defense"].values[:, :, _h_idx].ravel()  # home defense samples
    defense_a = post["defense"].values[:, :, _a_idx].ravel()  # away defense samples
    ha_samps  = post["home_advantage"].values.ravel()          # home advantage samples
    iota_samps = post["intercept"].values.ravel()              # intercept samples

    # Zero home advantage for neutral venues
    ha_term = 0.0 if is_neutral else ha_samps

    # Log-rate formula (mirrors build_bayesian_model)
    log_lam = iota_samps + ha_term + attack_h - defense_a
    log_mu  = iota_samps           + attack_a - defense_h

    lam_samples = np.clip(np.exp(log_lam), 1e-8, None)
    mu_samples  = np.clip(np.exp(log_mu),  1e-8, None)

    return lam_samples, mu_samples


def posterior_predictive_score_matrix(
    lam_samples: np.ndarray,
    mu_samples:  np.ndarray,
    max_goals:   int,
) -> np.ndarray:
    """Average independent-Poisson scoreline matrices over posterior draws.

    For each posterior draw i, computes the outer product of the Poisson PMF
    at (lam_samples[i], mu_samples[i]), yielding a (K+1, K+1) probability
    matrix.  The final result is the empirical average over all draws, which
    approximates the posterior-predictive distribution P(score | data).

    This is more expensive than the plug-in approximation (which uses only
    posterior-mean parameters) but gives properly calibrated uncertainty:
    teams with wide posteriors produce a more diffuse predictive matrix.

    Vectorized implementation: builds all (n_samples, K+1) PMF stacks in one
    NumPy operation via broadcasting, then contracts via einsum to avoid the
    Python-level loop over draws.

    Args:
        lam_samples: 1-D array of posterior home goal rate samples, shape (S,).
        mu_samples:  1-D array of posterior away goal rate samples, shape (S,).
        max_goals:   Maximum goals per team (matrix has shape (max_goals+1, max_goals+1)).

    Returns:
        2-D np.ndarray of shape (max_goals+1, max_goals+1), dtype float64,
        summing to 1.0 (up to floating-point precision).
    """
    k      = max_goals + 1
    goals  = np.arange(k, dtype=np.float64)  # shape (k,)
    n_samp = len(lam_samples)

    # Build PMF matrices for all samples simultaneously.
    # home_pmfs shape: (S, k) — each row is Poisson(goals | lam_samples[i])
    # away_pmfs shape: (S, k) — each row is Poisson(goals | mu_samples[i])
    home_pmfs = stats.poisson.pmf(
        goals[np.newaxis, :],          # broadcast over goals axis
        lam_samples[:, np.newaxis]     # broadcast over sample axis
    )  # shape (S, k)
    away_pmfs = stats.poisson.pmf(
        goals[np.newaxis, :],
        mu_samples[:, np.newaxis]
    )  # shape (S, k)

    # Outer product per sample via einsum: (S, k, k) = (S, k) x (S, k)
    # 'si,sj->sij' means: for each sample s, outer product over goal axes i, j
    all_matrices = np.einsum("si,sj->sij", home_pmfs, away_pmfs)  # shape (S, k, k)

    # Average over samples (axis 0)
    avg_matrix = all_matrices.mean(axis=0)  # shape (k, k)

    # Renormalize to exactly 1.0 (rounding and PMF truncation may shift total)
    total = avg_matrix.sum()
    if total > 0.0:
        avg_matrix /= total

    return avg_matrix

def bayesian_rate_samples(
    idata:       az.InferenceData,
    team_index:  dict[str, int],
    home_team:   str,
    away_team:   str,
    is_neutral:  bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw full posterior samples of (lambda, mu) for a single fixture.

    Extracts the NUTS posterior samples of attack, defense, home_advantage,
    and intercept from an InferenceData object, then computes (lambda, mu)
    for the requested fixture for every sample, returning two 1-D arrays
    of length (n_chains * n_draws).

    This is used only for the PRODUCTION prediction (not the backtest),
    where we want genuine posterior-predictive uncertainty rather than the
    plug-in approximation used in Section 11.4.

    Args:
        idata:       az.InferenceData from fit_bayesian_model (BAYES_IDATA_PROD).
        team_index:  Mapping from team name to integer index in CORE_TEAMS.
        home_team:   Dataset name of the home team.
        away_team:   Dataset name of the away team.
        is_neutral:  True zeros out the home-advantage term.

    Returns:
        Two 1-D np.ndarray arrays (lam_samples, mu_samples) of shape
        (n_samples,), each clipped to [1e-8, inf).
    """
    # Extract flat arrays of posterior samples for each global/per-team parameter.
    # az.extract collapses the (chain, draw) dimensions into a single 'sample' dim.
    post = idata.posterior

    # attack[team_index] and defense[team_index] are per-team; extract the
    # specific team's samples across all chains and draws.
    _h_idx = team_index[home_team]
    _a_idx = team_index[away_team]

    # Shape: (n_chains * n_draws,) after stacking
    attack_h  = post["attack"].values[:, :, _h_idx].ravel()   # home attack samples
    attack_a  = post["attack"].values[:, :, _a_idx].ravel()   # away attack samples
    defense_h = post["defense"].values[:, :, _h_idx].ravel()  # home defense samples
    defense_a = post["defense"].values[:, :, _a_idx].ravel()  # away defense samples
    ha_samps  = post["home_advantage"].values.ravel()          # home advantage samples
    iota_samps = post["intercept"].values.ravel()              # intercept samples

    # Zero home advantage for neutral venues
    ha_term = 0.0 if is_neutral else ha_samps

    # Log-rate formula (mirrors build_bayesian_model)
    log_lam = iota_samps + ha_term + attack_h - defense_a
    log_mu  = iota_samps           + attack_a - defense_h

    lam_samples = np.clip(np.exp(log_lam), 1e-8, None)
    mu_samples  = np.clip(np.exp(log_mu),  1e-8, None)

    return lam_samples, mu_samples


def posterior_predictive_score_matrix(
    lam_samples: np.ndarray,
    mu_samples:  np.ndarray,
    max_goals:   int,
) -> np.ndarray:
    """Average independent-Poisson scoreline matrices over posterior draws.

    For each posterior draw i, computes the outer product of the Poisson PMF
    at (lam_samples[i], mu_samples[i]), yielding a (K+1, K+1) probability
    matrix.  The final result is the empirical average over all draws, which
    approximates the posterior-predictive distribution P(score | data).

    This is more expensive than the plug-in approximation (which uses only
    posterior-mean parameters) but gives properly calibrated uncertainty:
    teams with wide posteriors produce a more diffuse predictive matrix.

    Vectorized implementation: builds all (n_samples, K+1) PMF stacks in one
    NumPy operation via broadcasting, then contracts via einsum to avoid the
    Python-level loop over draws.

    Args:
        lam_samples: 1-D array of posterior home goal rate samples, shape (S,).
        mu_samples:  1-D array of posterior away goal rate samples, shape (S,).
        max_goals:   Maximum goals per team (matrix has shape (max_goals+1, max_goals+1)).

    Returns:
        2-D np.ndarray of shape (max_goals+1, max_goals+1), dtype float64,
        summing to 1.0 (up to floating-point precision).
    """
    k      = max_goals + 1
    goals  = np.arange(k, dtype=np.float64)  # shape (k,)
    n_samp = len(lam_samples)

    # Build PMF matrices for all samples simultaneously.
    # home_pmfs shape: (S, k) — each row is Poisson(goals | lam_samples[i])
    # away_pmfs shape: (S, k) — each row is Poisson(goals | mu_samples[i])
    home_pmfs = stats.poisson.pmf(
        goals[np.newaxis, :],          # broadcast over goals axis
        lam_samples[:, np.newaxis]     # broadcast over sample axis
    )  # shape (S, k)
    away_pmfs = stats.poisson.pmf(
        goals[np.newaxis, :],
        mu_samples[:, np.newaxis]
    )  # shape (S, k)

    # Outer product per sample via einsum: (S, k, k) = (S, k) x (S, k)
    # 'si,sj->sij' means: for each sample s, outer product over goal axes i, j
    all_matrices = np.einsum("si,sj->sij", home_pmfs, away_pmfs)  # shape (S, k, k)

    # Average over samples (axis 0)
    avg_matrix = all_matrices.mean(axis=0)  # shape (k, k)

    # Renormalize to exactly 1.0 (rounding and PMF truncation may shift total)
    total = avg_matrix.sum()
    if total > 0.0:
        avg_matrix /= total

    return avg_matrix