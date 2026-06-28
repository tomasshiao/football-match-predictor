import numpy as np

from predictor.evaluation.metrics import ModelMetrics

def compute_ensemble_weights(
    backtest_metrics: dict[str, ModelMetrics],
    temperature:      float,
    real_model_names: list[str],
) -> dict[str, float]:
    """Derive ensemble combination weights from backtest log-loss scores.

    Uses a softmax over negative log-loss to convert each model's test-set
    performance into a weight:

        weight_m = exp(−loss_m / T) / Σ_m' exp(−loss_m' / T)

    where T = ``temperature``.  Lower T concentrates weight on the
    best-performing model; higher T blends all three more uniformly.

    Baseline entries in ``backtest_metrics`` are excluded from the
    weighting: they serve as a quality floor, not as ensemble members.

    Args:
        backtest_metrics: Full metrics dict keyed by model name, as
                          produced by the Section 12 evaluation cells.
                          May include baseline entries.
        temperature:      Softmax temperature (``CFG.ensemble.temperature``).
                          Must be strictly positive.
        real_model_names: Ordered list of model names to include in the
                          ensemble (excludes baselines).  Must exactly match
                          the keys used in ``FIXTURE_MATRICES`` (Section 15).

    Returns:
        ``dict[str, float]`` mapping each real model name to its weight.
        Values are non-negative and sum to 1.0 to within floating-point
        precision.

    Raises:
        ValueError: If ``temperature <= 0`` or any model name in
                    ``real_model_names`` is absent from ``backtest_metrics``.
    """
    if temperature <= 0.0:
        raise ValueError(
            f"temperature must be strictly positive, got {temperature}."
        )
    _missing = [n for n in real_model_names if n not in backtest_metrics]
    if _missing:
        raise ValueError(
            f"compute_ensemble_weights: model names not found in "
            f"backtest_metrics: {_missing}"
        )

    losses = np.array(
        [backtest_metrics[n].log_loss for n in real_model_names],
        dtype=np.float64,
    )

    # Numerically stable softmax via subtraction of the minimum loss.
    # Subtracting the minimum (best model) before exp prevents overflow when
    # temperatures are very small; it does not change the ratio of weights.
    shifted = -losses / temperature
    shifted -= shifted.max()
    exp_w   = np.exp(shifted)
    w       = exp_w / exp_w.sum()

    return {name: float(wi) for name, wi in zip(real_model_names, w)}

def _weighted_ensemble_matrices(
    matrices_dc:    np.ndarray,
    matrices_bayes: np.ndarray,
    matrices_xgb:   np.ndarray,
    weights:        dict[str, float],
) -> np.ndarray:
    """Combine three test-set matrix stacks into a single ensemble stack.

    Applies the per-model weights to produce the weight-averaged scoreline
    matrix for every test fixture.  The result is renormalized row-wise to
    guard against any accumulated floating-point drift across the three
    addends.

    Args:
        matrices_dc:    DC matrix stack, shape ``(n, K+1, K+1)``.
        matrices_bayes: Bayesian matrix stack, same shape.
        matrices_xgb:   XGBoost matrix stack, same shape.
        weights:        Dict mapping ``'Dixon-Coles'``, ``'Bayesian'``,
                        ``'XGBoost'`` to their respective scalar weights.

    Returns:
        Ensemble matrix stack, shape ``(n, K+1, K+1)``, each matrix
        summing to 1.0.
    """
    combined = (
        weights["Dixon-Coles"] * matrices_dc
        + weights["Bayesian"]   * matrices_bayes
        + weights["XGBoost"]    * matrices_xgb
    )
    # Renormalize each matrix to exactly 1.0
    row_sums = combined.sum(axis=(1, 2), keepdims=True)
    combined /= row_sums
    return combined

def combine_score_matrices(
    matrices: dict[str, np.ndarray],
    weights:  dict[str, float],
) -> np.ndarray:
    """Combine per-model scoreline matrices into a single ensemble matrix.

    Computes the weighted sum of matrices using the backtest-derived weights
    from ``compute_ensemble_weights``, then renormalises the result to
    exactly 1.0 to guard against floating-point drift in the summation.

    Args:
        matrices: Dict mapping model name to its scoreline probability
                  matrix (shape (K+1, K+1), summing to 1.0).
        weights:  Dict mapping model name to its ensemble weight (non-negative,
                  summing to 1.0).  Typically ``ENSEMBLE_WEIGHTS``.

    Returns:
        Combined (K+1, K+1) matrix summing to 1.0.

    Raises:
        KeyError: If any key in ``weights`` is absent from ``matrices``.
        ValueError: If matrices have inconsistent shapes.
    """
    _missing_keys = [k for k in weights if k not in matrices]
    if _missing_keys:
        raise KeyError(
            f"combine_score_matrices: weight keys not found in matrices: "
            f"{_missing_keys}"
        )

    _shapes = {k: v.shape for k, v in matrices.items()}
    if len(set(v for v in _shapes.values())) > 1:
        raise ValueError(
            f"Inconsistent matrix shapes in combine_score_matrices: {_shapes}"
        )

    # Weighted sum — start from zeros to avoid accumulating dtype surprises
    _shape = next(iter(matrices.values())).shape
    combined = np.zeros(_shape, dtype=np.float64)
    for _name, _w in weights.items():
        combined += _w * matrices[_name]

    # Renormalise to exactly 1.0
    _total = combined.sum()
    if _total > 0.0:
        combined /= _total

    return combined


def prediction_entropy(matrix: np.ndarray) -> float:
    """Compute the Shannon entropy of a scoreline probability matrix.

    Entropy H = -Σ p * log(p) across all cells, interpreted as bits if
    log base 2 is used (here we use nats for consistency with log_loss
    elsewhere in the pipeline).  Higher entropy means the prediction is
    more spread across possible scorelines (less certain).

    Args:
        matrix: 2-D np.ndarray summing to 1.0.

    Returns:
        Entropy in nats (float ≥ 0).
    """
    eps = 1e-15  # avoids log(0); consistent with sk_log_loss clip in Section 12
    flat = matrix.ravel()
    return float(-np.sum(flat * np.log(flat + eps)))