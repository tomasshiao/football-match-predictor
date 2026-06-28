import numpy as np
import optuna
import polars as pl
from sklearn.metrics import mean_poisson_deviance
from typing import Iterator
import xgboost as xgb

from predictor.config.pipeline_config import OptunaConfig, XGBoostConfig

def _validate_xgboost_inputs(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_cols: list[str],
) -> None:
    """Verify chronology, target presence, and feature availability.

    Args:
        train_df:     Training feature DataFrame (``TRAIN_FEATURES``).
        test_df:      Test feature DataFrame (``TEST_FEATURES``).
        feature_cols: Feature column names required by the XGBoost models.

    Raises:
        AssertionError: If the maximum training date is not strictly before
            the minimum test date, if target score columns are missing, or
            if any required feature column is absent from ``train_df``.
    """
    print("── Pre-implementation Validation Phase ──────────────────────────────")

    # 1. Chronological ordering: no test dates may precede training dates
    train_max = train_df["date"].max()
    test_min  = test_df["date"].min()
    assert train_max < test_min, (
        f"Chronology broken: Train max ({train_max}) >= Test min ({test_min})"
    )
    print(f"  ✓ Chronology: Train max ({train_max}) < Test min ({test_min})")

    # 2. Target columns must be present (both home and away goals)
    assert "home_score" in train_df.columns and "away_score" in train_df.columns, (
        "Target columns 'home_score' and 'away_score' missing from train_df."
    )
    print("  ✓ Target definitions: 'home_score' and 'away_score' present.")

    # 3. All required feature columns must exist in the training frame
    missing = set(feature_cols) - set(train_df.columns)
    assert not missing, f"Missing feature columns in TRAIN_FEATURES: {missing}"
    print("  ✓ Feature availability: all XGBoost feature columns present.")

    print("───────────────────────────────────────────────────────────────────\n")

def expanding_window_splits(
    df: pl.DataFrame,
    n_splits: int,
    min_train_fraction: float,
) -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]:
    """Generate chronological expanding-window train / validation folds.

    Splits are date-based (not row-index based) to avoid breaking temporal
    ordering when multiple matches share the same date.  The ``n_splits``
    unique date boundaries are evenly distributed across the sorted date range
    of ``df``; each fold expands the training window to that boundary and
    uses the next block as validation.

    Args:
        df:                 Training feature DataFrame, already filtered to
                            dates before the outer test cutoff.
        n_splits:           Number of folds to generate.
        min_train_fraction: Minimum fraction of ``df`` rows that a fold's
                            training window must cover.  Folds whose training
                            set would be smaller than this fraction are
                            skipped to prevent models trained on very little
                            data from distorting the Optuna objective.

    Yields:
        ``(train_fold, val_fold)`` Polars DataFrame pairs.  Both have at
        least one row; empty folds are never yielded.

    Raises:
        AssertionError: If no folds were generated (all fell below
            ``min_train_fraction`` or ``n_splits`` is too large for the
            date range).
    """
    df_sorted   = df.sort("date")
    n_rows      = df_sorted.height
    all_dates   = df_sorted["date"].unique().sort().to_list()
    n_dates     = len(all_dates)

    # Distribute n_splits cutpoints evenly across the date range.
    # Fold i uses dates[cutpoint_i] as the exclusive upper boundary for train.
    step        = max(1, n_dates // (n_splits + 1))
    cutpoints   = [all_dates[min(i * step, n_dates - 1)] for i in range(1, n_splits + 1)]

    n_yielded   = 0
    for cutpoint in cutpoints:
        train_fold = df_sorted.filter(pl.col("date") <  cutpoint)
        val_fold   = df_sorted.filter(pl.col("date") >= cutpoint)

        # Skip folds whose training set is too small
        if train_fold.height / n_rows < min_train_fraction:
            continue

        # Skip degenerate folds where the validation set is empty
        if val_fold.height == 0:
            print(
                f"Skipping fold with cutpoint {cutpoint} because validation set is empty.")
            continue

        n_yielded += 1
        yield train_fold, val_fold

    assert n_yielded > 0, (
        f"expanding_window_splits produced 0 folds "
        f"(n_splits={n_splits}, min_train_fraction={min_train_fraction}, "
        f"n_dates={n_dates}).  "
        "Reduce min_train_fraction or increase the date range of the training set."
    )

def optuna_objective(
    trial: optuna.Trial,
    train_features: pl.DataFrame,
    feature_cols: list[str],
    weight_col: str,
    config: OptunaConfig,
    xgb_config: XGBoostConfig,
) -> float:
    """Optuna trial objective: mean weighted Poisson deviance over CV folds.

    For each trial, samples hyperparameters from ``xgb_config.search_space``,
    trains separate home and away XGBoost models on each expanding-window CV
    fold, predicts on the validation fold, and returns the mean weighted
    Poisson deviance averaged across folds.  The Poisson deviance is
    consistent with the ``count:poisson`` training objective used by XGBoost.

    Args:
        trial:          Optuna trial object used to suggest hyperparameter
                        values.
        train_features: Training feature DataFrame (``TRAIN_FEATURES``).
                        The outer test set must never be passed here.
        feature_cols:   Ordered list of feature column names.
        weight_col:     Name of the sample-weight column.
        config:         ``OptunaConfig`` controlling CV fold count and
                        minimum training fraction.
        xgb_config:     ``XGBoostConfig`` providing the search space and
                        fixed hyperparameters.

    Returns:
        Mean Poisson deviance across all CV folds (lower is better).

    Raises:
        AssertionError: If ``expanding_window_splits`` yields zero folds.
    """
    params = {
        "n_estimators":    trial.suggest_int(
            "n_estimators", *xgb_config.search_space["n_estimators"]
        ),
        "max_depth":       trial.suggest_int(
            "max_depth", *xgb_config.search_space["max_depth"]
        ),
        "learning_rate":   trial.suggest_float(
            "learning_rate", *xgb_config.search_space["learning_rate"]
        ),
        "subsample":       trial.suggest_float(
            "subsample", *xgb_config.search_space["subsample"]
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", *xgb_config.search_space["colsample_bytree"]
        ),
        "reg_alpha":       trial.suggest_float(
            "reg_alpha", *xgb_config.search_space["reg_alpha"]
        ),
        "reg_lambda":      trial.suggest_float(
            "reg_lambda", *xgb_config.search_space["reg_lambda"]
        ),
        **xgb_config.fixed_params,
    }

    deviance_scores: list[float] = []

    for train_fold, val_fold in expanding_window_splits(
        train_features, config.cv_folds, config.min_train_fraction
    ):
        x_train      = train_fold.select(feature_cols).to_numpy()
        w_train      = train_fold[weight_col].to_numpy()
        y_train_home = train_fold["home_score"].to_numpy()
        y_train_away = train_fold["away_score"].to_numpy()

        x_val      = val_fold.select(feature_cols).to_numpy()
        w_val      = val_fold[weight_col].to_numpy()
        y_val_home = val_fold["home_score"].to_numpy()
        y_val_away = val_fold["away_score"].to_numpy()

        # Fit home model and predict
        model_home = xgb.XGBRegressor(**params)
        model_home.fit(x_train, y_train_home, sample_weight=w_train)
        pred_home = np.clip(model_home.predict(x_val), 1e-6, None)

        # Fit away model and predict
        model_away = xgb.XGBRegressor(**params)
        model_away.fit(x_train, y_train_away, sample_weight=w_train)
        pred_away = np.clip(model_away.predict(x_val), 1e-6, None)

        # Weighted Poisson deviance (consistent with count:poisson objective)
        dev_home = mean_poisson_deviance(y_val_home, pred_home, sample_weight=w_val)
        dev_away = mean_poisson_deviance(y_val_away, pred_away, sample_weight=w_val)
        deviance_scores.append((dev_home + dev_away) / 2.0)

    # Guard against empty fold list (expanding_window_splits already asserts,
    # but defensive check here keeps the float return type contract valid).
    assert len(deviance_scores) > 0, (
        "No CV folds were generated; cannot compute objective."
    )

    return float(np.mean(deviance_scores))

def run_optuna_search(
    train_features: pl.DataFrame,
    feature_cols: list[str],
    weight_col: str,
    config: "OptunaConfig",
    xgb_config: "XGBoostConfig",
) -> dict[str, float | int | str]:
    """Run Optuna hyperparameter optimisation and return the best parameters.

    Creates a TPE study with a reproducible seed and minimises the Optuna
    objective (mean weighted Poisson deviance) for ``config.n_trials`` trials
    or until ``config.timeout_seconds`` wall-clock seconds have elapsed.

    Args:
        train_features: Training feature DataFrame (``TRAIN_FEATURES``).
                        The outer test set must never be passed here.
        feature_cols:   Ordered list of feature column names.
        weight_col:     Name of the sample-weight column.
        config:         ``OptunaConfig`` controlling the search.
        xgb_config:     ``XGBoostConfig`` providing the search space and
                        fixed hyperparameters merged into the result.

    Returns:
        Dictionary of best tuned hyperparameters merged with the fixed
        parameters from ``xgb_config.fixed_params``.
    """
    sampler = optuna.samplers.TPESampler(seed=config.sampler_seed)
    study   = optuna.create_study(direction="minimize", sampler=sampler)

    study.optimize(
        lambda trial: optuna_objective(
            trial, train_features, feature_cols, weight_col, config, xgb_config
        ),
        n_trials=config.n_trials,
        timeout=config.timeout_seconds,
        show_progress_bar=False,   # verbosity is controlled globally via
                                   # optuna.logging.set_verbosity(WARNING)
    )

    return {**study.best_params, **xgb_config.fixed_params}

def train_xgboost_goal_models(
    train_df: pl.DataFrame,
    feature_cols: list[str],
    weight_col: str,
    hyperparameters: dict[str, float | int | str],
) -> tuple[xgb.XGBRegressor, xgb.XGBRegressor]:
    """Train separate XGBoost Poisson regressors for home and away goals.

    Fits two independent ``XGBRegressor`` instances — one per target — so
    that each model can specialise its tree structure to the home or away
    goal distribution respectively.  Both models share the same
    hyperparameters (tuned by Optuna in Section 9) and the same sample
    weights so that recent, high-importance matches exert proportionally
    more influence on the learned parameters.

    The ``count:poisson`` objective (enforced via ``XGBoostConfig``) uses a
    log link, which constrains predictions to be strictly positive and is
    consistent with the Poisson generative assumption shared by the
    Dixon-Coles and Bayesian models.

    Args:
        train_df:        Training DataFrame containing feature columns, the
                         ``home_score`` and ``away_score`` target columns, and
                         the sample-weight column.  Must *not* include any
                         test-set rows (chronological leakage guard).
        feature_cols:    Ordered list of feature column names; must exactly
                         match ``CFG.features.feature_columns``.
        weight_col:      Name of the pre-computed sample-weight column
                         (``"match_weight"`` in the standard pipeline).
        hyperparameters: Merged dict of tuned + fixed XGBoost hyperparameters
                         as returned by ``run_optuna_search``.  Must include
                         ``objective='count:poisson'`` and ``random_state``.

    Returns:
        A two-tuple ``(home_model, away_model)`` of fitted
        ``xgb.XGBRegressor`` instances.  The home model predicts
        ``home_score``; the away model predicts ``away_score``.
    """
    x      = train_df.select(feature_cols).to_numpy()
    w      = train_df[weight_col].to_numpy()
    y_home = train_df["home_score"].to_numpy()
    y_away = train_df["away_score"].to_numpy()

    home_model = xgb.XGBRegressor(**hyperparameters)
    home_model.fit(x, y_home, sample_weight=w)

    away_model = xgb.XGBRegressor(**hyperparameters)
    away_model.fit(x, y_away, sample_weight=w)

    return home_model, away_model

def predict_goal_rates(
    home_model: xgb.XGBRegressor,
    away_model: xgb.XGBRegressor,
    features: pl.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Predict expected goal rates (λ, μ) for a batch of fixtures.

    Applies the fitted home and away XGBoost models to the feature matrix
    and clips the output to a safe positive lower bound.  The clip is a
    defensive measure: ``count:poisson`` uses a log link so raw predictions
    are theoretically non-negative, but floating-point edge cases and
    ``np.clip`` guard against a zero or negative rate being passed downstream
    to ``np.log`` inside the Poisson PMF.

    This function is called in two contexts:
    - **Backtest** (Section 11.5): ``features = TEST_FEATURES``
    - **Production prediction** (Section 15): ``features`` is the single-row
      fixture frame built by ``build_fixture_feature_row``.

    Args:
        home_model:   Trained ``XGBRegressor`` for home goal prediction,
                      as returned by ``train_xgboost_goal_models``.
        away_model:   Trained ``XGBRegressor`` for away goal prediction,
                      as returned by ``train_xgboost_goal_models``.
        features:     Polars DataFrame of feature rows.  Column order must
                      match the order used during training.
        feature_cols: Ordered list of feature column names used to select
                      and order the feature matrix from ``features``.

    Returns:
        A two-tuple ``(lambda_xgb, mu_xgb)`` of 1-D ``np.ndarray`` arrays,
        each of length ``len(features)``, clipped to ``[1e-6, ∞)``.
        ``lambda_xgb[i]`` is the predicted home goal rate for fixture ``i``;
        ``mu_xgb[i]`` is the predicted away goal rate.
    """
    x = features.select(feature_cols).to_numpy()
    lam = np.clip(home_model.predict(x), 1e-6, None)
    mu  = np.clip(away_model.predict(x), 1e-6, None)
    return lam, mu