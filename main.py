"""FastAPI application exposing /predict and /backtest endpoints.

The ``predictor`` package is imported as a library; no pipeline logic lives
here.  A single ``PipelineConfig`` singleton is built at startup from the
request body and threaded through every function call.

Endpoints
---------
POST /predict
    Full two-pass pipeline (backtest → production).  Returns a
    ``FixturePrediction``-shaped JSON payload.

POST /backtest
    Backtest pass only (no production refit or fixture prediction).
    Returns per-model and ensemble ``ModelMetrics`` in JSON.
"""

from __future__ import annotations

import datetime
import io
import logging
import warnings
from typing import Any

import joblib
import numpy as np
import arviz as az
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from predictor import (
    # Config
    PipelineConfig,
    build_default_pipeline_config,
    # Data
    fetch_results_csv,
    fetch_shootout_csv,
    validate_schema,
    # Preparation
    load_fifa_code_mapping,
    resolve_team_name,
    select_core_teams,
    filter_to_core_teams,
    split_played_and_upcoming,
    filter_from_date,
    train_test_split_by_date,
    compute_match_weights,
    # Models
    fit_dixon_coles,
    build_bayesian_model,
    fit_bayesian_model,
    check_convergence,
    extract_posterior_means,
    bayesian_rate_samples,
    run_optuna_search,
    train_xgboost_goal_models,
    predict_goal_rates,
    run_feature_engineering,
    build_fixture_feature_row,
    compute_current_form,
    # Scoring
    batch_poisson_score_matrices,
    posterior_predictive_score_matrix,
    poisson_score_matrix,
    score_matrix_to_outcome_probs,
    score_matrix_to_playoff_outcome_probs,
    top_n_scorelines,
    combine_score_matrices,
    compute_ensemble_weights,
    prediction_entropy,
    # Evaluation
    evaluate_predictions,
    evaluate_baselines,
    # Outcomes
    FixturePrediction,
    OutcomeProbabilities,
    PlayoffOutcomeProbabilities,
    ModelMetrics,
)
from predictor.common.fifa_country_codes import FIFA_TO_DATASET_TEAM

logging.basicConfig(level=logging.INFO)
LOGGER: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_REAL_MODEL_NAMES: list[str] = ["Dixon-Coles", "Bayesian", "XGBoost"]

app = FastAPI(
    title="Football Score Predictor",
    description="Ensemble football match score prediction API.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for POST /predict.

    Args:
        home_team_fifa_code: Upper-case three-letter FIFA code of the home team.
        away_team_fifa_code: Upper-case three-letter FIFA code of the away team.
        match_date: ISO-format date string (YYYY-MM-DD) of the fixture.
        is_neutral_venue: Whether the match is played at a neutral venue.
        is_playoff: Whether the fixture is a knockout/playoff match.
    """

    home_team_fifa_code: str = Field(..., min_length=3, max_length=3)
    away_team_fifa_code: str = Field(..., min_length=3, max_length=3)
    match_date: datetime.date
    is_neutral_venue: bool = False
    is_playoff: bool = False


class BacktestRequest(BaseModel):
    """Request body for POST /backtest.

    Args:
        home_team_fifa_code: FIFA code used only for the leakage spot-check.
        away_team_fifa_code: FIFA code used only for the leakage spot-check.
        match_date: Fixture date (sets ``prod_ref_date`` in the config).
        is_neutral_venue: Neutral venue flag forwarded to PipelineConfig.
        is_playoff: Playoff flag forwarded to PipelineConfig.
    """

    home_team_fifa_code: str = Field(..., min_length=3, max_length=3)
    away_team_fifa_code: str = Field(..., min_length=3, max_length=3)
    match_date: datetime.date
    is_neutral_venue: bool = False
    is_playoff: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _metrics_to_dict(m: ModelMetrics) -> dict[str, Any]:
    """Serialise a ModelMetrics dataclass to a plain dict.

    Args:
        m: Populated ``ModelMetrics`` instance.

    Returns:
        JSON-serialisable dictionary of all metric fields.
    """
    return {
        "model_name":           m.model_name,
        "exact_score_accuracy": m.exact_score_accuracy,
        "outcome_accuracy":     m.outcome_accuracy,
        "log_loss":             m.log_loss,
        "brier_score":          m.brier_score,
        "home_goals_mae":       m.home_goals_mae,
        "away_goals_mae":       m.away_goals_mae,
        "home_goals_rmse":      m.home_goals_rmse,
        "away_goals_rmse":      m.away_goals_rmse,
        "n_evaluated":          m.n_evaluated,
    }


def _outcome_to_dict(o: OutcomeProbabilities) -> dict[str, float]:
    """Serialise OutcomeProbabilities to a plain dict.

    Args:
        o: ``OutcomeProbabilities`` instance.

    Returns:
        JSON-serialisable dictionary.
    """
    return {
        "p_home_win": o.p_home_win,
        "p_draw":     o.p_draw,
        "p_away_win": o.p_away_win,
    }


def _playoff_outcome_to_dict(p: PlayoffOutcomeProbabilities) -> dict[str, float]:
    """Serialise PlayoffOutcomeProbabilities to a plain dict.

    Args:
        p: ``PlayoffOutcomeProbabilities`` instance.

    Returns:
        JSON-serialisable dictionary.
    """
    return {
        "p_home_win":             p.p_home_win,
        "p_home_win_penalties":   p.p_home_win_penalties,
        "p_away_win_penalties":   p.p_away_win_penalties,
        "p_away_win":             p.p_away_win,
    }


def _run_backtest_pass(
    cfg: PipelineConfig,
    fixture_home_team: str,
    fixture_away_team: str,
) -> tuple[
    dict[str, ModelMetrics],
    dict[str, float],
    "np.ndarray",
    "np.ndarray",
    "np.ndarray",
]:
    """Execute the full backtest pass and return metrics plus production inputs.

    Runs data ingestion, preparation, all three model fits on the training
    window, backtest scoring, and ensemble weight derivation.  Also returns
    the full played match DataFrame and core-team list needed by the
    production pass so that data is not re-fetched.

    Args:
        cfg: Fully built ``PipelineConfig`` singleton.
        fixture_home_team: Dataset team name of the home team.
        fixture_away_team: Dataset team name of the away team.

    Returns:
        Five-tuple of:
        - ``backtest_metrics``: dict[model_name → ModelMetrics] (models + baselines)
        - ``ensemble_weights``: dict[model_name → float]
        - ``core_match_df``: Polars DataFrame of all played core-team matches
        - ``core_teams``: sorted list of core team names
        - ``shootout_df``: Polars DataFrame of historical shootouts

    Raises:
        HTTPException: On data fetch failure or convergence assertion.
    """
    import polars as pl

    # ── §2  Data ingestion ─────────────────────────────────────────────────
    LOGGER.info("Fetching results CSV …")
    raw_df = fetch_results_csv(cfg.data.source_url, cfg.data.cache_path)
    validate_schema(raw_df, start_date=cfg.data.start_date, pipeline_config=cfg)

    LOGGER.info("Fetching shootout CSV …")
    shootout_df = fetch_shootout_csv(cfg.data.shootout_url, cfg.data.cache_path)

    # ── §3  Team registry & core-team selection ────────────────────────────
    played_df, _ = split_played_and_upcoming(raw_df)
    core_df      = filter_from_date(played_df, cfg.data.start_date)

    train_df, test_df = train_test_split_by_date(core_df, cfg.split.cutoff_date)

    all_teams       = select_core_teams(train_df, cfg.team_filter.min_matches)
    core_match_df   = filter_to_core_teams(core_df, all_teams)
    train_core_df   = filter_to_core_teams(train_df, all_teams)
    test_core_df    = filter_to_core_teams(test_df, all_teams)

    LOGGER.info(
        "Core teams: %d  |  train rows: %d  |  test rows: %d",
        len(all_teams), train_core_df.height, test_core_df.height,
    )

    # ── §4–5  Weighting ────────────────────────────────────────────────────
    train_ref_date     = train_core_df["date"].max()
    train_weighted_df  = compute_match_weights(
        train_core_df, train_ref_date, cfg.weighting
    )

    # ── §6  Dixon-Coles (backtest) ─────────────────────────────────────────
    LOGGER.info("Fitting Dixon-Coles (backtest) …")
    dc_ratings_bt = fit_dixon_coles(train_weighted_df, all_teams, cfg.dixon_coles)

    # ── §7  Feature engineering ────────────────────────────────────────────
    LOGGER.info("Engineering features …")
    train_features, test_features, _ = run_feature_engineering(
        core_match_df     = core_match_df,
        train_weighted_df = train_weighted_df,
        dc_ratings_backtest = dc_ratings_bt,
        cfg               = cfg,
        fixture_home_team = fixture_home_team,
        fixture_away_team = fixture_away_team,
    )

    # ── §8–9  Bayesian (backtest) ──────────────────────────────────────────
    LOGGER.info("Building & sampling Bayesian model (backtest) …")
    _team_idx_bt   = {t: i for i, t in enumerate(all_teams)}
    _home_idx_bt   = np.array(
        [_team_idx_bt[t] for t in train_core_df["home_team"].to_list()], dtype=np.int32
    )
    _away_idx_bt   = np.array(
        [_team_idx_bt[t] for t in train_core_df["away_team"].to_list()], dtype=np.int32
    )
    _home_goals_bt = train_core_df["home_score"].to_numpy().astype(np.int64)
    _away_goals_bt = train_core_df["away_score"].to_numpy().astype(np.int64)
    _neutral_bt    = train_core_df["neutral"].to_numpy().astype(bool)
    _weights_bt    = train_weighted_df["match_weight"].to_numpy().astype(np.float64)

    bayes_model_bt = build_bayesian_model(
        home_idx   = _home_idx_bt,
        away_idx   = _away_idx_bt,
        home_goals = _home_goals_bt,
        away_goals = _away_goals_bt,
        is_neutral = _neutral_bt,
        weights    = _weights_bt,
        n_teams    = len(all_teams),
        config     = cfg.bayesian,
    )
    idata_bt       = fit_bayesian_model(bayes_model_bt, cfg.bayesian)
    conv_report_bt = check_convergence(idata_bt, cfg.bayesian)
    if not conv_report_bt.passed:
        LOGGER.warning("Bayesian backtest convergence check FAILED: %s", conv_report_bt)
    posterior_bt   = extract_posterior_means(idata_bt, all_teams, conv_report_bt)

    # ── §10  XGBoost (backtest) ────────────────────────────────────────────
    LOGGER.info("Running Optuna search …")
    best_params = run_optuna_search(
        train_features = train_features,
        feature_cols   = cfg.features.feature_columns,
        weight_col     = "match_weight",
        config         = cfg.optuna,
        xgb_config     = cfg.xgboost,
    )
    LOGGER.info("Training XGBoost models (backtest) …")
    xgb_home_bt, xgb_away_bt = train_xgboost_goal_models(
        train_df         = train_features,
        feature_cols     = cfg.features.feature_columns,
        weight_col       = "match_weight",
        hyperparameters  = best_params,
    )

    # ── §11  Backtest scoring ──────────────────────────────────────────────
    LOGGER.info("Scoring backtest …")
    _test_home   = test_core_df["home_team"].to_list()
    _test_away   = test_core_df["away_team"].to_list()
    _test_neutral = test_core_df["neutral"].to_numpy().astype(bool)
    _act_home    = test_core_df["home_score"].to_numpy().astype(int)
    _act_away    = test_core_df["away_score"].to_numpy().astype(int)
    _n_test      = test_core_df.height
    _max_g       = cfg.evaluation.max_goals

    # Dixon-Coles
    _dc_lam = np.array([
        dc_ratings_bt.predict_rate(h, a, n)[0]
        for h, a, n in zip(_test_home, _test_away, _test_neutral.tolist())
    ])
    _dc_mu = np.array([
        dc_ratings_bt.predict_rate(h, a, n)[1]
        for h, a, n in zip(_test_home, _test_away, _test_neutral.tolist())
    ])
    dc_mats = batch_poisson_score_matrices(_dc_lam, _dc_mu, _max_g, rho=dc_ratings_bt.rho)

    # Bayesian (plug-in posterior means)
    from predictor.scoring.matrix import bayesian_rate_means_batch
    _bayes_lam, _bayes_mu = bayesian_rate_means_batch(
        posterior  = posterior_bt,
        home_teams = _test_home,
        away_teams = _test_away,
        is_neutral = _test_neutral,
    )
    bayes_mats = batch_poisson_score_matrices(_bayes_lam, _bayes_mu, _max_g, rho=0.0)

    # XGBoost
    _xgb_lam, _xgb_mu = predict_goal_rates(
        xgb_home_bt, xgb_away_bt, test_features, cfg.features.feature_columns
    )
    xgb_mats = batch_poisson_score_matrices(_xgb_lam, _xgb_mu, _max_g, rho=0.0)

    # ── §12  Metrics ───────────────────────────────────────────────────────
    LOGGER.info("Computing backtest metrics …")
    backtest_metrics: dict[str, ModelMetrics] = {
        "Dixon-Coles": evaluate_predictions(
            dc_mats, _act_home, _act_away, _dc_lam, _dc_mu, "Dixon-Coles"
        ),
        "Bayesian": evaluate_predictions(
            bayes_mats, _act_home, _act_away, _bayes_lam, _bayes_mu, "Bayesian"
        ),
        "XGBoost": evaluate_predictions(
            xgb_mats, _act_home, _act_away, _xgb_lam, _xgb_mu, "XGBoost"
        ),
    }
    baselines = evaluate_baselines(train_core_df, test_core_df, cfg)
    backtest_metrics.update(baselines)

    # ── §13  Ensemble weights ──────────────────────────────────────────────
    ensemble_weights = compute_ensemble_weights(
        backtest_metrics = backtest_metrics,
        temperature      = cfg.ensemble.temperature,
        real_model_names = _REAL_MODEL_NAMES,
    )
    LOGGER.info("Ensemble weights: %s", ensemble_weights)

    return (
        backtest_metrics,
        ensemble_weights,
        core_match_df,
        all_teams,
        shootout_df,
    )


def _run_production_pass(
    cfg:              PipelineConfig,
    fixture_home_team: str,
    fixture_away_team: str,
    ensemble_weights: dict[str, float],
    core_match_df:    "pl.DataFrame",
    all_teams:        list[str],
    shootout_df:      "pl.DataFrame",
) -> FixturePrediction:
    """Refit all models on the full history and predict the headline fixture.

    Args:
        cfg: PipelineConfig singleton.
        fixture_home_team: Dataset name of the home team.
        fixture_away_team: Dataset name of the away team.
        ensemble_weights: Backtest-derived model combination weights.
        core_match_df: Full played-match history (train + test, core teams).
        all_teams: Sorted list of core team names.
        shootout_df: Historical shootout DataFrame.

    Returns:
        Populated ``FixturePrediction`` dataclass.
    """
    import polars as pl

    _max_g = cfg.evaluation.max_goals

    # ── Production weighting (reference date = today) ──────────────────────
    prod_weighted_df = compute_match_weights(
        core_match_df, cfg.prod_ref_date, cfg.weighting
    )

    # ── §6 prod  Dixon-Coles (full history) ───────────────────────────────
    LOGGER.info("Fitting Dixon-Coles (production) …")
    dc_ratings_prod = fit_dixon_coles(prod_weighted_df, all_teams, cfg.dixon_coles)

    # ── §7 prod  Feature engineering ──────────────────────────────────────
    # We need CURRENT_FORM_DF for the fixture row; run_feature_engineering
    # returns form_df as its third element.
    LOGGER.info("Engineering features (production) …")
    _, _, form_df = run_feature_engineering(
        core_match_df       = core_match_df,
        train_weighted_df   = prod_weighted_df,
        dc_ratings_backtest = dc_ratings_prod,
        cfg                 = cfg,
        fixture_home_team   = fixture_home_team,
        fixture_away_team   = fixture_away_team,
    )
    current_form_df = compute_current_form(core_match_df, cfg.features.rolling_window)

    fixture_features = build_fixture_feature_row(
        home_team    = fixture_home_team,
        away_team    = fixture_away_team,
        is_neutral   = cfg.fixture.is_neutral_venue,
        ratings      = dc_ratings_prod,
        current_form = current_form_df,
        feature_cols = cfg.features.feature_columns,
    )

    # ── §8–9 prod  Bayesian (full history) ────────────────────────────────
    LOGGER.info("Building & sampling Bayesian model (production) …")
    _team_idx_prod  = {t: i for i, t in enumerate(all_teams)}
    _home_idx_prod  = np.array(
        [_team_idx_prod[t] for t in core_match_df["home_team"].to_list()], dtype=np.int32
    )
    _away_idx_prod  = np.array(
        [_team_idx_prod[t] for t in core_match_df["away_team"].to_list()], dtype=np.int32
    )
    _home_goals_prod = core_match_df["home_score"].to_numpy().astype(np.int64)
    _away_goals_prod = core_match_df["away_score"].to_numpy().astype(np.int64)
    _neutral_prod    = core_match_df["neutral"].to_numpy().astype(bool)
    _weights_prod    = prod_weighted_df["match_weight"].to_numpy().astype(np.float64)

    bayes_model_prod = build_bayesian_model(
        home_idx   = _home_idx_prod,
        away_idx   = _away_idx_prod,
        home_goals = _home_goals_prod,
        away_goals = _away_goals_prod,
        is_neutral = _neutral_prod,
        weights    = _weights_prod,
        n_teams    = len(all_teams),
        config     = cfg.bayesian,
    )
    idata_prod       = fit_bayesian_model(bayes_model_prod, cfg.bayesian)
    conv_report_prod = check_convergence(idata_prod, cfg.bayesian)
    if not conv_report_prod.passed:
        LOGGER.warning("Bayesian production convergence check FAILED: %s", conv_report_prod)

    # ── §10 prod  XGBoost (full history) ──────────────────────────────────
    LOGGER.info("Training XGBoost models (production) …")
    xgb_home_prod, xgb_away_prod = train_xgboost_goal_models(
        train_df        = form_df.join(
            prod_weighted_df.select(["match_id", "match_weight"]),
            on="match_id", how="left",
        ).drop_nulls(subset=cfg.features.feature_columns),
        feature_cols    = cfg.features.feature_columns,
        weight_col      = "match_weight",
        hyperparameters = {
            **cfg.xgboost.fixed_params,
            **{k: int(round((v[0] + v[1]) / 2)) if isinstance(v[0], int) else (v[0] + v[1]) / 2
               for k, v in cfg.xgboost.search_space.items()},
        },
    )

    # ── §15  Production scoring ────────────────────────────────────────────
    LOGGER.info("Computing production scoreline matrices …")

    # Dixon-Coles
    _dc_lam_p, _dc_mu_p = dc_ratings_prod.predict_rate(
        fixture_home_team, fixture_away_team, cfg.fixture.is_neutral_venue
    )
    dc_matrix_prod = poisson_score_matrix(_dc_lam_p, _dc_mu_p, _max_g, rho=dc_ratings_prod.rho)

    # Bayesian (full posterior predictive)
    lam_samples, mu_samples = bayesian_rate_samples(
        idata      = idata_prod,
        team_index = _team_idx_prod,
        home_team  = fixture_home_team,
        away_team  = fixture_away_team,
        is_neutral = cfg.fixture.is_neutral_venue,
    )
    bayes_matrix_prod = posterior_predictive_score_matrix(lam_samples, mu_samples, _max_g)

    # XGBoost
    _xgb_lam_p, _xgb_mu_p = predict_goal_rates(
        xgb_home_prod, xgb_away_prod, fixture_features, cfg.features.feature_columns
    )
    xgb_matrix_prod = poisson_score_matrix(
        float(_xgb_lam_p[0]), float(_xgb_mu_p[0]), _max_g, rho=0.0
    )

    per_model_matrices: dict[str, np.ndarray] = {
        "Dixon-Coles": dc_matrix_prod,
        "Bayesian":    bayes_matrix_prod,
        "XGBoost":     xgb_matrix_prod,
    }

    # ── §15  Ensemble combination ──────────────────────────────────────────
    ensemble_matrix = combine_score_matrices(per_model_matrices, ensemble_weights)

    # ── §15  Outcome probabilities ─────────────────────────────────────────
    outcome = score_matrix_to_outcome_probs(ensemble_matrix)
    playoff_outcome: PlayoffOutcomeProbabilities | None = None
    if cfg.fixture.is_playoff:
        playoff_outcome = score_matrix_to_playoff_outcome_probs(
            matrix     = ensemble_matrix,
            home_team  = fixture_home_team,
            away_team  = fixture_away_team,
            shootout_df = shootout_df,
        )

    # ── Persist production artefacts ───────────────────────────────────────
    joblib.dump(xgb_home_prod, cfg.paths.xgb_home_model_prod)
    joblib.dump(xgb_away_prod, cfg.paths.xgb_away_model_prod)
    idata_prod.to_netcdf(str(cfg.paths.bayes_idata_prod))
    LOGGER.info("Production artefacts saved to %s", cfg.paths.root_dir)

    return FixturePrediction(
        home_team           = fixture_home_team,
        away_team           = fixture_away_team,
        match_date          = cfg.fixture.match_date,
        per_model_matrices  = per_model_matrices,
        ensemble_matrix     = ensemble_matrix,
        ensemble_weights    = ensemble_weights,
        outcome             = outcome,
        playoff_outcome     = playoff_outcome,
        top_scorelines      = top_n_scorelines(ensemble_matrix, n=10),
        entropy             = prediction_entropy(ensemble_matrix),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict")
def predict(request: PredictRequest) -> JSONResponse:
    """Run the full two-pass pipeline and return the fixture prediction.

    Executes the backtest pass to derive ensemble weights, then the production
    pass to refit all models on the full history and predict the requested
    fixture.

    Args:
        request: ``PredictRequest`` body.

    Returns:
        JSON with keys: ``home_team``, ``away_team``, ``match_date``,
        ``outcome``, ``playoff_outcome`` (if applicable), ``top_scorelines``,
        ``ensemble_weights``, ``entropy``, ``backtest_metrics``.

    Raises:
        HTTPException 400: If either FIFA code is unknown.
        HTTPException 500: On any unhandled pipeline error.
    """
    # Resolve FIFA codes to dataset team names
    fifa_map = load_fifa_code_mapping(FIFA_TO_DATASET_TEAM)
    try:
        home_team = resolve_team_name(request.home_team_fifa_code.upper(), fifa_map)
        away_team = resolve_team_name(request.away_team_fifa_code.upper(), fifa_map)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Build the singleton config for this request
    try:
        cfg = build_default_pipeline_config(
            home_team_fifa_code = request.home_team_fifa_code.upper(),
            away_team_fifa_code = request.away_team_fifa_code.upper(),
            match_date          = request.match_date,
            is_neutral_venue    = request.is_neutral_venue,
            is_playoff          = request.is_playoff,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Config build failed: {exc}") from exc

    try:
        backtest_metrics, ensemble_weights, core_match_df, all_teams, shootout_df = (
            _run_backtest_pass(cfg, home_team, away_team)
        )
        prediction = _run_production_pass(
            cfg               = cfg,
            fixture_home_team = home_team,
            fixture_away_team = away_team,
            ensemble_weights  = ensemble_weights,
            core_match_df     = core_match_df,
            all_teams         = all_teams,
            shootout_df       = shootout_df,
        )
    except Exception as exc:
        LOGGER.exception("Pipeline error in /predict")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload: dict[str, Any] = {
        "home_team":        prediction.home_team,
        "away_team":        prediction.away_team,
        "match_date":       prediction.match_date.isoformat(),
        "outcome":          _outcome_to_dict(prediction.outcome),
        "playoff_outcome":  (
            _playoff_outcome_to_dict(prediction.playoff_outcome)
            if prediction.playoff_outcome is not None else None
        ),
        "top_scorelines": [
            {"home_goals": h, "away_goals": a, "probability": p}
            for (h, a), p in prediction.top_scorelines
        ],
        "ensemble_weights": prediction.ensemble_weights,
        "entropy":          prediction.entropy,
        "backtest_metrics": {
            name: _metrics_to_dict(m) for name, m in backtest_metrics.items()
        },
    }
    return JSONResponse(content=payload)


@app.post("/backtest")
def backtest(request: BacktestRequest) -> JSONResponse:
    """Run the backtest pass only and return per-model metrics.

    No production refit or fixture prediction is performed.  Useful for
    evaluating model quality independently of any specific match.

    Args:
        request: ``BacktestRequest`` body.

    Returns:
        JSON with keys: ``backtest_metrics``, ``ensemble_weights``.

    Raises:
        HTTPException 400: If either FIFA code is unknown.
        HTTPException 500: On any unhandled pipeline error.
    """
    fifa_map = load_fifa_code_mapping(FIFA_TO_DATASET_TEAM)
    try:
        home_team = resolve_team_name(request.home_team_fifa_code.upper(), fifa_map)
        away_team = resolve_team_name(request.away_team_fifa_code.upper(), fifa_map)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        cfg = build_default_pipeline_config(
            home_team_fifa_code = request.home_team_fifa_code.upper(),
            away_team_fifa_code = request.away_team_fifa_code.upper(),
            match_date          = request.match_date,
            is_neutral_venue    = request.is_neutral_venue,
            is_playoff          = request.is_playoff,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Config build failed: {exc}") from exc

    try:
        backtest_metrics, ensemble_weights, _, _, _ = (
            _run_backtest_pass(cfg, home_team, away_team)
        )
    except Exception as exc:
        LOGGER.exception("Pipeline error in /backtest")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload: dict[str, Any] = {
        "backtest_metrics": {
            name: _metrics_to_dict(m) for name, m in backtest_metrics.items()
        },
        "ensemble_weights": ensemble_weights,
    }
    return JSONResponse(content=payload)


# ---------------------------------------------------------------------------
# Console-script entry point
# ---------------------------------------------------------------------------

def serve() -> None:
    """Launch the API with uvicorn.

    This is what ``[project.scripts]`` in ``pyproject.toml`` points the
    ``football-predictor`` command at (``main:serve``). It mirrors the
    Dockerfile's ``CMD`` invocation so the console script and the
    containerized entrypoint behave identically. Import is local to keep
    ``uvicorn`` off the import path for anything that only needs ``app``
    (e.g. ``uvicorn main:app`` itself, or tests that import this module).
    """
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        timeout_keep_alive=75,
        log_level="info",
    )


if __name__ == "__main__":
    serve()