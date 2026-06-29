"""Football score prediction package.

This package exposes the full public API of the prediction pipeline via a
single import surface.  Every symbol re-exported here corresponds to a
documented class or function defined in one of the sub-packages; nothing
is constructed or executed at import time beyond the re-exports themselves.

Typical usage::

    from predictor import PipelineConfig, build_default_pipeline_config
    from predictor import FixturePrediction, OutcomeProbabilities

Modules are grouped below in pipeline-execution order so that this file
also serves as a reading guide to the package structure.
"""

# ---------------------------------------------------------------------------
# §1  Configuration — dataclasses and builder
# ---------------------------------------------------------------------------
from predictor.config.data_config import (
    DataConfig,
    TeamFilterConfig,
    WeightingConfig,
    SplitConfig,
    get_tournament_weight_table,
)
from predictor.config.model_configs import (
    DixonColesConfig,
    BayesianConfig,
    FeatureConfig,
    OptunaConfig,
    XGBoostConfig,
    EnsembleConfig,
)
from predictor.config.pipeline_config import (
    FixtureConfig,
    EvaluationConfig,
    PathConfig,
    PipelineConfig,
    build_default_pipeline_config,
)

# ---------------------------------------------------------------------------
# §2  Data ingestion
# ---------------------------------------------------------------------------
from predictor.data.fetcher import (
    fetch_results_csv,
    fetch_shootout_csv,
)
from predictor.data.schema import (
    EXPECTED_COLUMNS,
    validate_schema,
)

# ---------------------------------------------------------------------------
# §3–5  Data preparation
# ---------------------------------------------------------------------------
from predictor.preparation.team_registry import (
    load_fifa_code_mapping,
    resolve_team_name,
    select_core_teams,
    filter_to_core_teams,
)
from predictor.preparation.splitter import (
    split_played_and_upcoming,
    filter_from_date,
    train_test_split_by_date,
)
from predictor.preparation.weighting import (
    compute_time_decay_weight,
    compute_tournament_weight,
    compute_match_weights,
)

# ---------------------------------------------------------------------------
# §6–10  Models
# ---------------------------------------------------------------------------
from predictor.models.dixon_coles import (
    DixonColesRatings,
    fit_dixon_coles,
)
from predictor.models.bayesian import (
    ConvergenceReport,
    BayesianPosterior,
    build_bayesian_model,
    fit_bayesian_model,
    check_convergence,
    posterior_predictive_check,
    extract_posterior_means,
)
from predictor.models.xgboost import (
    expanding_window_splits,
    run_optuna_search,
    train_xgboost_goal_models,
    predict_goal_rates,
)
from predictor.models.features import (
    add_rolling_form_features,
    run_rolling_form_leakage_spot_check,
    add_dixon_coles_features,
    assemble_feature_df,
    compute_current_form,
    run_feature_engineering,
    build_fixture_feature_row,
)

# ---------------------------------------------------------------------------
# §11–13, 15  Scoring
# ---------------------------------------------------------------------------
from predictor.scoring.matrix import (
    poisson_score_matrix,
    batch_poisson_score_matrices,
    posterior_predictive_score_matrix,
)
from predictor.scoring.outcomes import (
    OutcomeProbabilities,
    PlayoffOutcomeProbabilities,
    FixturePrediction,
    score_matrix_to_outcome_probs,
    score_matrix_to_playoff_outcome_probs,
    top_n_scorelines,
    estimate_shootout_win_prob,
)
from predictor.scoring.ensemble import (
    compute_ensemble_weights,
    combine_score_matrices,
    prediction_entropy,
)

# ---------------------------------------------------------------------------
# §12, 16–19  Evaluation and visualisation
# ---------------------------------------------------------------------------
from predictor.evaluation.metrics import (
    ModelMetrics,
    evaluate_predictions,
    evaluate_baselines,
    calibration_curve_1x2,
    expected_calibration_error,
)
from predictor.evaluation.visualisation import (
    plot_score_heatmap,
    plot_outcome_probabilities,
    plot_top_n_scorelines,
    plot_calibration_curve,
    plot_backtest_metrics_table,
)

# ---------------------------------------------------------------------------
# Public API declaration
# ---------------------------------------------------------------------------
__all__: list[str] = [
    # config.data_config
    "DataConfig",
    "TeamFilterConfig",
    "WeightingConfig",
    "SplitConfig",
    "get_tournament_weight_table",
    # config.model_configs
    "DixonColesConfig",
    "BayesianConfig",
    "FeatureConfig",
    "OptunaConfig",
    "XGBoostConfig",
    "EnsembleConfig",
    # config.pipeline_config
    "FixtureConfig",
    "EvaluationConfig",
    "PathConfig",
    "PipelineConfig",
    "build_default_pipeline_config",
    # data.fetcher
    "fetch_results_csv",
    "fetch_shootout_csv",
    # data.schema
    "EXPECTED_COLUMNS",
    "validate_schema",
    # preparation.team_registry
    "load_fifa_code_mapping",
    "resolve_team_name",
    "select_core_teams",
    "filter_to_core_teams",
    # preparation.splitter
    "split_played_and_upcoming",
    "filter_from_date",
    "train_test_split_by_date",
    # preparation.weighting
    "compute_time_decay_weight",
    "compute_tournament_weight",
    "compute_match_weights",
    # models.dixon_coles
    "DixonColesRatings",
    "fit_dixon_coles",
    # models.bayesian
    "ConvergenceReport",
    "BayesianPosterior",
    "build_bayesian_model",
    "fit_bayesian_model",
    "check_convergence",
    "posterior_predictive_check",
    "extract_posterior_means",
    # models.xgboost
    "expanding_window_splits",
    "run_optuna_search",
    "train_xgboost_goal_models",
    "predict_goal_rates",
    # models.features
    "add_rolling_form_features",
    "run_rolling_form_leakage_spot_check",
    "add_dixon_coles_features",
    "assemble_feature_df",
    "compute_current_form",
    "run_feature_engineering",
    "build_fixture_feature_row",
    # scoring.matrix
    "poisson_score_matrix",
    "batch_poisson_score_matrices",
    "posterior_predictive_score_matrix",
    # scoring.outcomes
    "OutcomeProbabilities",
    "PlayoffOutcomeProbabilities",
    "FixturePrediction",
    "score_matrix_to_outcome_probs",
    "score_matrix_to_playoff_outcome_probs",
    "top_n_scorelines",
    "estimate_shootout_win_prob",
    # scoring.ensemble
    "compute_ensemble_weights",
    "combine_score_matrices",
    "prediction_entropy",
    # evaluation.metrics
    "ModelMetrics",
    "evaluate_predictions",
    "evaluate_baselines",
    "calibration_curve_1x2",
    "expected_calibration_error",
    # evaluation.visualisation
    "plot_score_heatmap",
    "plot_outcome_probabilities",
    "plot_top_n_scorelines",
    "plot_calibration_curve",
    "plot_backtest_metrics_table",
]