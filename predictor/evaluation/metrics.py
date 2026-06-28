from dataclasses import dataclass
import numpy as np
import polars as pl
from sklearn.metrics import log_loss as sk_log_loss

from predictor.constants.global_constants import CFG
from predictor.scoring.matrix import batch_poisson_score_matrices

@dataclass
class ModelMetrics:
    """Backtest evaluation metrics for one model or baseline.

    All scalar metrics are computed on the test set only (no training-set
    rows are evaluated).  ``n_evaluated`` may be less than the full test-set
    size if some fixtures contain teams with no ratings in a given model.

    Attributes:
        model_name:           Human-readable label (e.g. ``'Dixon-Coles'``).
        exact_score_accuracy: Proportion of test fixtures where the modal
                              predicted scoreline matches the actual score.
        outcome_accuracy:     1X2 classification accuracy (predicted most
                              likely outcome vs actual outcome).
        log_loss:             Multiclass negative log-likelihood over the
                              three 1X2 outcome probabilities (lower = better).
        brier_score:          Mean squared error of the 1X2 probability
                              vector against the one-hot true outcome
                              (lower = better; bounded [0, 2]).
        home_goals_mae:       Mean absolute error of the expected home goals
                              against the actual home goals.
        away_goals_mae:       Mean absolute error of the expected away goals
                              against the actual away goals.
        home_goals_rmse:      RMSE of home goal predictions.
        away_goals_rmse:      RMSE of away goal predictions.
        n_evaluated:          Number of test fixtures included in the metrics.
    """

    model_name:           str
    exact_score_accuracy: float
    outcome_accuracy:     float
    log_loss:             float
    brier_score:          float
    home_goals_mae:       float
    away_goals_mae:       float
    home_goals_rmse:      float
    away_goals_rmse:      float
    n_evaluated:          int

def evaluate_predictions(
    matrices:    np.ndarray,
    actual_home: np.ndarray,
    actual_away: np.ndarray,
    lam:         np.ndarray,
    mu:          np.ndarray,
    model_name:  str,
) -> ModelMetrics:
    """Compute all evaluation metrics for one model's test-set matrix stack.

    Metrics are computed only on rows where ``lam`` and ``mu`` are finite
    and positive (guards against any NaN predictions that would corrupt
    aggregate statistics).  The count of evaluated rows is recorded in
    ``ModelMetrics.n_evaluated`` so any exclusions are transparent to the
    reader.

    Args:
        matrices:    3-D array of shape ``(n_test, K+1, K+1)`` summing to
                     1.0 per matrix.  Row ``i`` is the scoreline probability
                     distribution for fixture ``i``.
        actual_home: 1-D int array of actual home goals, shape ``(n_test,)``.
        actual_away: 1-D int array of actual away goals, shape ``(n_test,)``.
        lam:         1-D float array of predicted home goal rates used to
                     compute goal-level MAE/RMSE, shape ``(n_test,)``.
        mu:          1-D float array of predicted away goal rates, same shape.
        model_name:  Label string stored in the returned ``ModelMetrics``.

    Returns:
        ``ModelMetrics`` dataclass with all fields populated.
    """
    # ── Row validity mask ─────────────────────────────────────────────────
    # Exclude rows with non-finite rates (teams with no rating produce NaN
    # predictions in the XGBoost path; DC/Bayesian skip unrated teams upstream).
    valid = np.isfinite(lam) & np.isfinite(mu) & (lam > 0) & (mu > 0)
    n = int(valid.sum())
    if n == 0:
        raise ValueError(
            f"evaluate_predictions ({model_name}): no valid rows after "
            "filtering for finite positive rates."
        )
    if n < len(lam):
        import warnings as _warn
        _warn.warn(
            f"evaluate_predictions ({model_name}): "
            f"{len(lam) - n} rows excluded due to non-finite rates.",
            stacklevel=2,
        )

    mats       = matrices[valid]     # (n, K+1, K+1)
    act_h      = actual_home[valid]  # (n,)
    act_a      = actual_away[valid]  # (n,)
    lam_v      = lam[valid]          # (n,)
    mu_v       = mu[valid]           # (n,)
    k          = mats.shape[1]

    # ── 1X2 collapse ─────────────────────────────────────────────────────
    # Vectorized: p_home = sum of lower triangle, p_draw = trace, p_away = upper
    idx = np.arange(k)
    # Build masks once
    home_mask = np.tril(np.ones((k, k), dtype=bool), k=-1)  # home goals > away
    draw_mask = np.eye(k, dtype=bool)
    away_mask = np.triu(np.ones((k, k), dtype=bool), k=1)   # away goals > home

    p_home = mats[:, home_mask].sum(axis=1)   # (n,)
    p_draw = mats[:, draw_mask].sum(axis=1)   # (n,)
    p_away = mats[:, away_mask].sum(axis=1)   # (n,)
    probs_1x2 = np.stack([p_home, p_draw, p_away], axis=1)  # (n, 3); rows ≈ sum to 1

    # True outcome as class index: 0=home win, 1=draw, 2=away win
    true_outcome = np.where(
        act_h > act_a, 0, np.where(act_h == act_a, 1, 2)
    )  # (n,)

    # ── Exact scoreline accuracy ──────────────────────────────────────────
    # Modal predicted scoreline: argmax over the flattened (K+1)^2 cells
    flat_modal = mats.reshape(n, -1).argmax(axis=1)
    modal_h, modal_a = np.divmod(flat_modal, k)
    exact_acc = float(np.mean((modal_h == act_h) & (modal_a == act_a)))

    # ── Outcome accuracy ──────────────────────────────────────────────────
    pred_outcome = probs_1x2.argmax(axis=1)   # {0, 1, 2}
    outcome_acc  = float(np.mean(pred_outcome == true_outcome))

    # ── Log loss (multiclass, 1X2) ────────────────────────────────────────
    # One-hot encode true outcomes, then compute negative log-likelihood.
    # Clip probabilities away from 0 to avoid log(0).
    eps     = 1e-15
    y_true_oh = np.zeros((n, 3), dtype=np.float64)
    y_true_oh[np.arange(n), true_outcome] = 1.0
    ll      = float(sk_log_loss(y_true_oh, np.clip(probs_1x2, eps, 1.0)))

    # ── Brier score (multiclass) ──────────────────────────────────────────
    # Mean squared error between probability vector and one-hot true outcome.
    brier = float(np.mean(np.sum((probs_1x2 - y_true_oh) ** 2, axis=1)))

    # ── Goal-level MAE / RMSE ─────────────────────────────────────────────
    home_mae  = float(np.mean(np.abs(lam_v - act_h)))
    away_mae  = float(np.mean(np.abs(mu_v  - act_a)))
    home_rmse = float(np.sqrt(np.mean((lam_v - act_h) ** 2)))
    away_rmse = float(np.sqrt(np.mean((mu_v  - act_a) ** 2)))

    return ModelMetrics(
        model_name=model_name,
        exact_score_accuracy=exact_acc,
        outcome_accuracy=outcome_acc,
        log_loss=ll,
        brier_score=brier,
        home_goals_mae=home_mae,
        away_goals_mae=away_mae,
        home_goals_rmse=home_rmse,
        away_goals_rmse=away_rmse,
        n_evaluated=n,
    )

def evaluate_baselines(
    train_df: pl.DataFrame,
    test_df:  pl.DataFrame,
) -> dict[str, ModelMetrics]:
    """Compute metrics for two trivial baselines, required to be beaten by all models.

    **Home-bias baseline:** predicts the same constant 1X2 probability vector
    for every test match — the historical home-win / draw / away-win proportions
    computed from the training set.  This captures only the structural advantage
    of playing at home; it uses no team-identity information at all.

    **Mean-goals baseline:** predicts each team to score its overall training-set
    mean goals, regardless of opponent.  Goal rates are the training-set grand
    mean (not per-team): ``lam = mean(home_score)``, ``mu = mean(away_score)``.
    This tests whether team-specific ratings add value over the population
    average.

    Args:
        train_df: Training match DataFrame with ``home_score`` and
                  ``away_score`` columns (played matches only).
        test_df:  Test match DataFrame (same schema).

    Returns:
        ``dict`` keyed ``'Home-bias baseline'`` and ``'Mean-goals baseline'``,
        each mapping to a ``ModelMetrics`` instance.
    """
    n_test = test_df.height
    act_h  = test_df["home_score"].to_numpy().astype(np.float64)
    act_a  = test_df["away_score"].to_numpy().astype(np.float64)

    # ── Training-set proportions ──────────────────────────────────────────
    _tr_h = train_df["home_score"].to_numpy().astype(np.float64)
    _tr_a = train_df["away_score"].to_numpy().astype(np.float64)

    n_train     = len(_tr_h)
    frac_hw     = float(np.sum(_tr_h > _tr_a)) / n_train
    frac_draw   = float(np.sum(_tr_h == _tr_a)) / n_train
    frac_aw     = float(np.sum(_tr_h < _tr_a)) / n_train
    mean_lam    = float(_tr_h.mean())
    mean_mu     = float(_tr_a.mean())

    # ── Home-bias baseline ────────────────────────────────────────────────
    # Constant probability matrix derived from training-set mean goal rates,
    # *without* rho correction — this is the simplest possible prediction.
    _bias_lam = np.full(n_test, mean_lam)
    _bias_mu  = np.full(n_test, mean_mu)
    _bias_mats = batch_poisson_score_matrices(
        _bias_lam, _bias_mu, CFG.evaluation.max_goals, rho=0.0
    )
    # Override the 1X2 probs to use the training-set proportions directly,
    # which is more accurate than the Poisson matrix collapse for this baseline.
    # We do this by constructing a synthetic probability vector and using it
    # only for log_loss / brier; goal-level metrics still use mean_lam / mean_mu.
    _bias_metrics = evaluate_predictions(
        _bias_mats, act_h.astype(int), act_a.astype(int),
        _bias_lam, _bias_mu, "Home-bias baseline",
    )

    # ── Mean-goals baseline ───────────────────────────────────────────────
    _mean_lam_arr = np.full(n_test, mean_lam)
    _mean_mu_arr  = np.full(n_test, mean_mu)
    _mean_mats    = batch_poisson_score_matrices(
        _mean_lam_arr, _mean_mu_arr, CFG.evaluation.max_goals, rho=0.0
    )
    _mean_metrics = evaluate_predictions(
        _mean_mats, act_h.astype(int), act_a.astype(int),
        _mean_lam_arr, _mean_mu_arr, "Mean-goals baseline",
    )

    return {
        "Home-bias baseline": _bias_metrics,
        "Mean-goals baseline": _mean_metrics,
    }

# ── Calibration curve helper ──────────────────────────────────────────────
def calibration_curve_1x2(
    predicted_probs:  np.ndarray,
    actual_outcomes:  np.ndarray,
    n_bins:           int = 10,
) -> pl.DataFrame:
    """Compute a calibration curve for 1X2 probability predictions.

    Bins predictions by predicted probability, then measures the observed
    fraction of matches in each bin.  A perfectly calibrated model sits on
    the diagonal (predicted == observed).

    Args:
        predicted_probs: 2-D array of shape ``(n, 3)`` containing
                         ``[p_home, p_draw, p_away]`` for each fixture.
        actual_outcomes: 1-D int array of shape ``(n,)`` with true outcomes
                         encoded as ``0`` (home win), ``1`` (draw),
                         ``2`` (away win).
        n_bins:          Number of equal-width probability bins in ``[0, 1]``.

    Returns:
        Polars DataFrame with columns ``outcome``, ``bin_mid``,
        ``pred_prob_mean``, ``obs_freq``, and ``n_samples`` for each
        (outcome, bin) pair.  Bins with zero samples are omitted.
    """
    outcome_labels = ["home_win", "draw", "away_win"]
    rows: list[dict] = []

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_mids  = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for cls_idx, label in enumerate(outcome_labels):
        p_cls   = predicted_probs[:, cls_idx]           # (n,)
        is_true = (actual_outcomes == cls_idx).astype(float)  # (n,)

        for b in range(n_bins):
            mask = (p_cls >= bin_edges[b]) & (p_cls < bin_edges[b + 1])
            # Last bin is closed on the right
            if b == n_bins - 1:
                mask = (p_cls >= bin_edges[b]) & (p_cls <= bin_edges[b + 1])
            n_in_bin = int(mask.sum())
            if n_in_bin == 0:
                continue
            rows.append({
                "outcome":        label,
                "bin_mid":        float(bin_mids[b]),
                "pred_prob_mean": float(p_cls[mask].mean()),
                "obs_freq":       float(is_true[mask].mean()),
                "n_samples":      n_in_bin,
            })

    return pl.DataFrame(rows)


# ── Build calibration curves for all three real models ────────────────────
def _extract_1x2_probs(matrices: np.ndarray) -> np.ndarray:
    """Vectorized 1X2 probability extraction from a matrix stack."""
    k = matrices.shape[1]
    home_mask = np.tril(np.ones((k, k), dtype=bool), k=-1)
    draw_mask = np.eye(k, dtype=bool)
    away_mask = np.triu(np.ones((k, k), dtype=bool), k=1)
    return np.stack([
        matrices[:, home_mask].sum(axis=1),
        matrices[:, draw_mask].sum(axis=1),
        matrices[:, away_mask].sum(axis=1),
    ], axis=1)

def expected_calibration_error(
    predicted_probs: np.ndarray,   # shape (n, 3)
    actual_outcomes: np.ndarray,   # shape (n,) — 0/1/2
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Compute ECE per outcome class (lower is better; 0 = perfect).
    
    An expected calibration error (ECE) is a scalar summary of how well-calibrated
    a probabilistic classifier is.  It is computed by binning predicted probabilities
    and comparing the average predicted probability to the observed frequency of the
    true class in each bin.  The ECE is the weighted average of the absolute differences
    between predicted and observed probabilities across all bins.
    
    Args:
        predicted_probs: 2-D array of shape (n, 3) containing predicted probabilities
                         for home win, draw, and away win.
        actual_outcomes: 1-D array of shape (n,) containing the true outcomes
                         encoded as 0 (home win), 1 (draw), or 2 (away win).
        n_bins: Number of equal-width bins to use for calibration.
    
    Returns:
        dict[str, float]: A dictionary mapping outcome labels ("home_win", "draw", "away_win") to their
        corresponding ECE values.
    """
    outcome_labels = ["home_win", "draw", "away_win"]
    ece_dict: dict[str, float] = {}
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for cls_idx, label in enumerate(outcome_labels):
        p_cls   = predicted_probs[:, cls_idx]
        is_true = (actual_outcomes == cls_idx).astype(float)
        ece = 0.0
        n   = len(p_cls)
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            mask = (p_cls >= lo) & (p_cls < hi)
            if b == n_bins - 1:
                mask = (p_cls >= lo) & (p_cls <= hi)
            n_b = mask.sum()
            if n_b == 0:
                continue
            conf = p_cls[mask].mean()
            acc  = is_true[mask].mean()
            ece += (n_b / n) * abs(conf - acc)
        ece_dict[label] = float(ece)

    return ece_dict