import numpy as np
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import polars as pl

from predictor.config.pipeline_config import PipelineConfig
from predictor.constants.constants import COLOURS
from predictor.evaluation.metrics import ModelMetrics
from predictor.scoring.outcomes import OutcomeProbabilities, PlayoffOutcomeProbabilities, top_n_scorelines

def plot_score_heatmap(
    matrix: np.ndarray,
    home_team: str,
    away_team: str,
    max_goals_display: int = 6,
    title_suffix: str = "",
    model_weights: dict[str, float] | None = None,
) -> matplotlib.figure.Figure:
    """Render a scoreline probability matrix as a publication-quality heatmap.

    Each cell (i, j) shows P(home = i goals, away = j goals) as a percentage.
    Diagonal cells (draws) are framed with a green border. Cells below 0.5 %
    are left unannotated to reduce clutter.

    Args:
        matrix:            (K+1, K+1) scoreline probability matrix summing to 1.
        home_team:         Dataset name of the home team (y-axis label).
        away_team:         Dataset name of the away team (x-axis label).
        max_goals_display: Maximum goals per team shown on each axis.
        title_suffix:      Optional parenthetical appended to the figure title
                           e.g. "(Ensemble)" or "(Dixon-Coles)".
        model_weights:     If provided, renders a small weight legend in the
                           figure subtitle (use for the ensemble panel).

    Returns:
        matplotlib.figure.Figure
    """
    k = min(max_goals_display + 1, matrix.shape[0])
    sub = matrix[:k, :k].copy()
    # Renormalise the displayed sub-matrix so percentages are interpretable
    # (probability mass beyond max_goals_display is not shown).
    sub_pct = sub * 100.0

    # ── Colour map: dark background → electric blue (home-team colour) ────
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "pitch_heat",
        [COLOURS["bg"], "#0d3a6e", COLOURS["home"]],
    )

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    fig.patch.set_facecolor(COLOURS["bg"])
    ax.set_facecolor(COLOURS["bg"])

    im = ax.imshow(
        sub_pct,
        cmap=cmap,
        aspect="auto",
        vmin=0,
        vmax=max(sub_pct.max(), 1.0),
        interpolation="nearest",
        origin="upper",
    )

    # ── Grid lines between cells ─────────────────────────────────────────
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.grid(which="minor", color=COLOURS["border"], linewidth=0.8, alpha=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    # ── Diagonal draw borders ────────────────────────────────────────────
    for d in range(k):
        ax.add_patch(mpatches.FancyBboxPatch(
            (d - 0.48, d - 0.48), 0.96, 0.96,
            boxstyle="square,pad=0",
            linewidth=1.6,
            edgecolor=COLOURS["accent"],
            facecolor="none",
            zorder=3,
        ))

    # ── Cell annotations ─────────────────────────────────────────────────
    threshold = 0.5   # suppress cells below 0.5 %
    for i in range(k):
        for j in range(k):
            val = sub_pct[i, j]
            if val < threshold:
                continue
            # Choose annotation text colour based on cell luminance
            cell_norm = sub[i, j] / max(sub.max(), 1e-9)
            text_col  = COLOURS["text"] if cell_norm < 0.55 else COLOURS["bg"]
            weight    = "bold" if val == sub_pct.max() else "normal"
            ax.text(
                j, i, f"{val:.1f}%",
                ha="center", va="center",
                fontsize=8, color=text_col, fontweight=weight,
            )

    # ── Axes ─────────────────────────────────────────────────────────────
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(range(k), fontsize=9, color=COLOURS["text"])
    ax.set_yticklabels(range(k), fontsize=9, color=COLOURS["text"])
    ax.set_xlabel(f"{away_team}  (away goals)", fontsize=10,
                  color=COLOURS["text"], labelpad=8)
    ax.set_ylabel(f"{home_team}  (home goals)", fontsize=10,
                  color=COLOURS["text"], labelpad=8)

    # ── Colourbar ────────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cbar.set_label("Probability (%)", fontsize=9, color=COLOURS["text"], labelpad=8)
    cbar.ax.yaxis.set_tick_params(color=COLOURS["text_muted"], labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COLOURS["text_muted"])
    cbar.outline.set_edgecolor(COLOURS["border"])

    # ── Title ────────────────────────────────────────────────────────────
    title_parts = [f"{home_team}  vs  {away_team}"]
    if title_suffix:
        title_parts.append(title_suffix)
    ax.set_title(
        "  ·  ".join(title_parts),
        fontsize=12, fontweight="bold", color=COLOURS["text"], pad=12,
    )

    # ── Outcome probability strips on the margins ─────────────────────────
    # Compute 1X2 from the displayed sub-matrix
    p_home = float(np.tril(sub, k=-1).sum())
    p_draw = float(np.trace(sub))
    p_away = float(np.triu(sub, k=1).sum())
    outcome_str = (
        f"  Home win {p_home:.1%}   ·   Draw {p_draw:.1%}"
        f"   ·   Away win {p_away:.1%}  "
    )
    ax.text(
        0.5, -0.09, outcome_str,
        ha="center", va="top", fontsize=8.5,
        color=COLOURS["text_muted"], transform=ax.transAxes,
    )

    # ── Draw border legend ────────────────────────────────────────────────
    _draw_patch = mpatches.Patch(
        facecolor="none", edgecolor=COLOURS["accent"], linewidth=1.5,
        label="Draw (diagonal)",
    )
    ax.legend(
        handles=[_draw_patch],
        loc="upper right", fontsize=8,
        framealpha=0.7,
    )

    # ── Weight legend (ensemble panel only) ──────────────────────────────
    if model_weights:
        weight_lines = [
            f"  {name}: {w:.0%}"
            for name, w in model_weights.items()
        ]
        fig.text(
            0.01, 0.02, "Ensemble weights\n" + "\n".join(weight_lines),
            fontsize=7.5, color=COLOURS["text_muted"], va="bottom",
        )

    return fig

def plot_outcome_probabilities(
    outcome: OutcomeProbabilities,
    home_team: str,
    away_team: str,
    per_model_outcomes: dict[str, OutcomeProbabilities] | None = None,
    ensemble_weights: dict[str, float] | None = None,
    playoff_outcome: PlayoffOutcomeProbabilities | None = None,
) -> matplotlib.figure.Figure:
    """Render 1X2 outcome probabilities as a publication-quality bar chart.

    The ensemble probabilities are shown as solid bars (home blue, draw grey,
    away coral).  If per_model_outcomes is provided, each model's values are
    overlaid as narrow semi-transparent bars to expose model disagreement.

    Args:
        outcome:             Ensemble OutcomeProbabilities.
        home_team:           Dataset name of the home team.
        away_team:           Dataset name of the away team.
        per_model_outcomes:  Dict mapping model name → OutcomeProbabilities.
        ensemble_weights:    Dict mapping model name → weight (for legend).

    Returns:
        matplotlib.figure.Figure
    """
    if playoff_outcome is not None:
        # ── Playoff: four-bar layout ───────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        fig.patch.set_facecolor(COLOURS["bg"])
        ax.set_facecolor(COLOURS["surface"])

        outcomes_4 = [
            (f"{home_team} win (90 min)",   playoff_outcome.p_home_win,           COLOURS["home"]),
            (f"{home_team} win (penalties)", playoff_outcome.p_home_win_penalties, COLOURS["home_pens"]),  # lighter blue
            (f"{away_team} win (penalties)", playoff_outcome.p_away_win_penalties, COLOURS["away_pens"]),  # lighter coral
            (f"{away_team} win (90 min)",   playoff_outcome.p_away_win,           COLOURS["away"]),
        ]

        bar_height  = 0.52
        y_positions = np.arange(len(outcomes_4))

        for y, (label, prob, colour) in zip(y_positions, outcomes_4):
            ax.barh(y, prob, height=bar_height, color=colour, alpha=0.92,
                    linewidth=0, zorder=3)
            x_ann = prob - 0.015 if prob > 0.12 else prob + 0.008
            ha_ann = "right" if prob > 0.12 else "left"
            col_ann = COLOURS["bg"] if prob > 0.12 else COLOURS["text"]
            ax.text(x_ann, y, f"{prob:.1%}", ha=ha_ann, va="center",
                    fontsize=11, fontweight="bold", color=col_ann, zorder=4)

        # Bracket annotation showing total probability of penalties
        _pen_total = playoff_outcome.p_home_win_penalties + playoff_outcome.p_away_win_penalties
        ax.annotate(
            f"P(penalties) = {_pen_total:.1%}",
            xy=(max(playoff_outcome.p_home_win_penalties,
                    playoff_outcome.p_away_win_penalties) + 0.01, 1.5),
            fontsize=8, color=COLOURS["text_muted"],
        )

        ax.set_yticks(y_positions)
        ax.set_yticklabels([lbl for lbl, _, _ in outcomes_4],
                           fontsize=10, color=COLOURS["text"])
        ax.set_xlim(0, 1.0)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_xlabel("Probability", fontsize=9, color=COLOURS["text_muted"], labelpad=6)
        ax.invert_yaxis()
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
        ax.grid(axis="x", color=COLOURS["border"], linewidth=0.5, alpha=0.5)
        ax.grid(axis="y", visible=False)
        ax.set_title(
            f"{home_team}  vs  {away_team}  —  Playoff Outcome Probabilities",
            fontsize=11.5, fontweight="bold", color=COLOURS["text"], pad=10,
        )
        return fig

    else:
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        fig.patch.set_facecolor(COLOURS["bg"])
        ax.set_facecolor(COLOURS["surface"])

        outcomes = [
            (f"{home_team} win", outcome.p_home_win, COLOURS["home"]),
            ("Draw",             outcome.p_draw,     COLOURS["draw"]),
            (f"{away_team} win", outcome.p_away_win, COLOURS["away"]),
        ]

        bar_height  = 0.52
        y_positions = np.arange(len(outcomes))

        # ── Per-model background bars (rendered first, behind ensemble) ───────
        if per_model_outcomes:
            model_names  = list(per_model_outcomes.keys())
            n_models     = len(model_names)
            sub_h        = bar_height / (n_models + 1)
            model_colours = [COLOURS["home"], COLOURS["away"], COLOURS["accent"]]

            for m_idx, (m_name, m_oc) in enumerate(per_model_outcomes.items()):
                m_vals = [m_oc.p_home_win, m_oc.p_draw, m_oc.p_away_win]
                y_off  = (m_idx - n_models / 2.0) * sub_h * 0.9
                m_col  = model_colours[m_idx % len(model_colours)]
                _w     = ensemble_weights.get(m_name, None) if ensemble_weights else None
                _label = f"{m_name}" + (f"  {_w:.0%}" if _w is not None else "")

                bars_m = ax.barh(
                    y_positions + y_off, m_vals,
                    height=sub_h * 0.75,
                    color=m_col, alpha=0.35, linewidth=0,
                    label=_label,
                    zorder=2,
                )

        # ── Ensemble bars ─────────────────────────────────────────────────────
        for y, (label, prob, colour) in zip(y_positions, outcomes):
            ax.barh(
                y, prob, height=bar_height,
                color=colour, alpha=0.92,
                linewidth=0, zorder=3,
            )
            # Probability annotation inside or outside bar
            x_ann = prob - 0.015 if prob > 0.12 else prob + 0.008
            ha_ann = "right"      if prob > 0.12 else "left"
            col_ann = COLOURS["bg"]    if prob > 0.12 else COLOURS["text"]
            ax.text(
                x_ann, y, f"{prob:.1%}",
                ha=ha_ann, va="center",
                fontsize=11, fontweight="bold", color=col_ann,
                zorder=4,
            )

        # ── Reference line at 33.3 % (uniform baseline) ──────────────────────
        ax.axvline(1 / 3, color=COLOURS["border"], linewidth=1.0,
                linestyle="--", alpha=0.7, zorder=1)
        ax.text(
            1 / 3 + 0.005, -0.55,
            "Equal chance\n(33.3%)",
            fontsize=7.5, color=COLOURS["text_muted"], va="bottom",
        )

        # ── Axes ─────────────────────────────────────────────────────────────
        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            [label for label, _, _ in outcomes],
            fontsize=10.5, color=COLOURS["text"],
        )
        ax.set_xlim(0, 1.0)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_xlabel("Probability", fontsize=9, color=COLOURS["text_muted"], labelpad=6)
        ax.invert_yaxis()
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
        ax.grid(axis="x", color=COLOURS["border"], linewidth=0.5, alpha=0.5)
        ax.grid(axis="y", visible=False)

        ax.set_title(
            f"{home_team}  vs  {away_team}  —  1X2 Outcome Probabilities",
            fontsize=11.5, fontweight="bold", color=COLOURS["text"], pad=10,
        )

        if per_model_outcomes:
            ax.legend(
                title="Individual models",
                loc="lower right", fontsize=8,
                framealpha=0.8, ncol=1,
            )

        return fig

def plot_top_n_scorelines(
    matrix: np.ndarray,
    home_team: str,
    away_team: str,
    n: int = 10,
    pipeline_config: PipelineConfig | None = None,
) -> matplotlib.figure.Figure:
    """Render the top-N most probable exact scorelines as a bar chart.

    Bars are coloured by outcome:
      • Home win  → electric blue
      • Draw      → neutral grey
      • Away win  → coral red

    Each bar is annotated with its probability as a percentage.
    The cumulative probability covered by the displayed scorelines is shown
    as a subtitle annotation.

    Args:
        matrix:    (K+1, K+1) scoreline probability matrix.
        home_team: Dataset name of the home team (x-axis legend).
        away_team: Dataset name of the away team.
        n:         Number of scorelines to display.

    Returns:
        matplotlib.figure.Figure
    """
    assert pipeline_config is not None, "pipeline_config must be provided for evaluation"
    
    scorelines = top_n_scorelines(matrix, n)

    labels = []
    probs  = []
    colors = []
    for (h, a), p in scorelines:
        labels.append(f"{pipeline_config.fixture.home_team_fifa_code} {h}-{a} {pipeline_config.fixture.away_team_fifa_code}")
        probs.append(p)
        if h > a:
            colors.append(COLOURS["home"])
        elif h == a:
            colors.append(COLOURS["draw"])
        else:
            colors.append(COLOURS["away"])

    cumulative = sum(probs)

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    fig.patch.set_facecolor(COLOURS["bg"])
    ax.set_facecolor(COLOURS["surface"])

    x = np.arange(len(labels))
    bars = ax.bar(
        x, probs,
        color=colors, alpha=0.88,
        width=0.62, linewidth=0,
        zorder=3,
    )

    # ── Value annotations ─────────────────────────────────────────────────
    max_p = max(probs)
    for bar, p in zip(bars, probs):
        y_ann = bar.get_height() + max_p * 0.018
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_ann,
            f"{p:.1%}",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="semibold",
            color=COLOURS["text"],
        )

    # ── Axes ─────────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, color=COLOURS["text"])
    ax.set_xlim(-0.55, len(labels) - 0.45)
    ax.set_ylim(0, max_p * 1.22)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_ylabel("Probability", fontsize=9, color=COLOURS["text_muted"], labelpad=6)
    ax.grid(axis="y", color=COLOURS["border"], linewidth=0.5, alpha=0.5)
    ax.grid(axis="x", visible=False)
    ax.spines["bottom"].set_color(COLOURS["border"])

    # ── Rank labels ───────────────────────────────────────────────────────
    for i, bar in enumerate(bars, 1):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            -max_p * 0.07,
            f"#{i}",
            ha="center", va="top",
            fontsize=7, color=COLOURS["text_muted"],
            transform=ax.transData,
        )

    # ── Legend ───────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=COLOURS["home"], label=f"{home_team} win"),
        mpatches.Patch(color=COLOURS["draw"], label="Draw"),
        mpatches.Patch(color=COLOURS["away"], label=f"{away_team} win"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=8.5, framealpha=0.8)

    # ── Title and subtitle ────────────────────────────────────────────────
    ax.set_title(
        f"{home_team}  vs  {away_team}  —  Top {n} Most Probable Scorelines",
        fontsize=11.5, fontweight="bold", color=COLOURS["text"], pad=10,
    )
    ax.text(
        0.5, 1.01,
        f"Combined probability of displayed scorelines: {cumulative:.1%}",
        ha="center", va="bottom", fontsize=8, color=COLOURS["text_muted"],
        transform=ax.transAxes,
    )

    return fig

def plot_calibration_curve(
    curves: dict[str, pl.DataFrame],
) -> matplotlib.figure.Figure:
    """Plot 1X2 calibration curves for multiple models on shared axes.

    For each model and each outcome class ('home_win', 'draw', 'away_win'),
    the mean predicted probability in each bin is plotted against the
    observed frequency.  A perfectly calibrated model sits on the diagonal.

    Args:
        curves:         Dict mapping model name → calibration DataFrame
                        (as returned by calibration_curve_1x2).
        outcome_filter: Which outcome class to highlight in the primary panel.
                        One of 'home_win', 'draw', 'away_win'.

    Returns:
        matplotlib.figure.Figure
    """
    outcome_labels = ["home_win", "draw", "away_win"]
    outcome_display = {
        "home_win": "Home Win",
        "draw":     "Draw",
        "away_win": "Away Win",
    }
    outcome_colors  = {
        "home_win": COLOURS["home"],
        "draw":     COLOURS["draw"],
        "away_win": COLOURS["away"],
    }

    model_names   = list(curves.keys())
    model_markers = ["o", "s", "^"]
    model_ls      = ["-", "--", ":"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.patch.set_facecolor(COLOURS["bg"])
    fig.suptitle(
        "1X2 Calibration Curves — Predicted vs Observed Probability",
        fontsize=12, fontweight="bold", color=COLOURS["text"],
    )

    for ax, outcome in zip(axes, outcome_labels):
        ax.set_facecolor(COLOURS["surface"])

        # Perfect calibration diagonal
        ax.plot(
            [0, 1], [0, 1],
            color=COLOURS["border"], linewidth=1.2,
            linestyle="--", label="Perfect calibration",
        )
        ax.fill_between(
            [0, 1], [0, 1], color=COLOURS["border"], alpha=0.06,
        )

        for (model_name, marker, ls) in zip(model_names, model_markers, model_ls):
            df_model = curves[model_name].filter(pl.col("outcome") == outcome)
            if df_model.is_empty():
                continue

            pred_means = df_model["pred_prob_mean"].to_numpy()
            obs_freqs  = df_model["obs_freq"].to_numpy()
            n_samples  = df_model["n_samples"].to_numpy()

            ax.plot(
                pred_means, obs_freqs,
                color=outcome_colors[outcome],
                linewidth=1.4, linestyle=ls,
                marker=marker, markersize=5,
                alpha=0.85, label=model_name,
            )
            # Bubble size proportional to sample count
            ax.scatter(
                pred_means, obs_freqs,
                s=np.sqrt(n_samples) * 3.5,
                color=outcome_colors[outcome],
                alpha=0.25, linewidths=0, zorder=2,
            )

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_title(
            outcome_display[outcome],
            fontsize=10, fontweight="semibold", color=COLOURS["text"], pad=7,
        )
        ax.set_xlabel("Predicted Probability", fontsize=8.5,
                      color=COLOURS["text_muted"])
        ax.set_ylabel("Observed Frequency", fontsize=8.5,
                      color=COLOURS["text_muted"])
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(colors=COLOURS["text_muted"])
        for sp in ax.spines.values():
            sp.set_edgecolor(COLOURS["border"])
        ax.legend(fontsize=7.5, framealpha=0.7, loc="upper left")
        ax.grid(color=COLOURS["border"], linewidth=0.4, alpha=0.5)

    return fig

# --- Helper function for colour interpolation between two RGB colours ----------
def _blend(c1: tuple[float, float, float], c2: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    """
    Linear interpolation between two RGB colors.
    
    Args:
        c1: First color as an RGB tuple (r, g, b), each in [0, 1].
        c2: Second color as an RGB tuple (r, g, b), each in [0, 1].
        t: Interpolation factor in [0, 1], where 0 returns c1 and 1 returns c2.
    
    Returns:
        Interpolated color as an RGB tuple (r, g, b).
    """
    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    return (1 - t) * c1 + t * c2

def plot_backtest_metrics_table(
    metrics: dict[str, ModelMetrics],
    real_model_names: list[str],
) -> matplotlib.figure.Figure:
    """Render a colour-coded backtest metrics table as a Matplotlib figure.

    Metric columns are normalised so the best-performing cell (green) and
    worst-performing cell (red) are highlighted.  Baseline rows use a
    lighter background to distinguish them from model rows.

    Args:
        metrics:          Dict mapping model name → ModelMetrics (all entries,
                          including baselines).
        real_model_names: Ordered list of real model names (not baselines).

    Returns:
        matplotlib.figure.Figure
    """
    # Column definitions: (display_name, attribute, lower_is_better)
    _cols = [
        ("Log-loss",         "log_loss",             True),
        ("Brier Score",      "brier_score",           True),
        ("Outcome Acc",      "outcome_accuracy",     False),
        ("Exact Score Acc",  "exact_score_accuracy", False),
        ("Home Goals MAE",   "home_goals_mae",        True),
        ("Away Goals MAE",   "away_goals_mae",        True),
        ("N Evaluated",      "n_evaluated",          False),
    ]

    row_names = list(metrics.keys())
    col_names = [c[0] for c in _cols]

    # Build data array
    data = np.array([
        [getattr(metrics[r], attr) for _, attr, _ in _cols]
        for r in row_names
    ], dtype=float)

    fig, ax = plt.subplots(
        figsize=(13, max(3.5, len(row_names) * 0.72 + 1.8)),
        constrained_layout=True,
    )
    fig.patch.set_facecolor(COLOURS["bg"])
    ax.set_facecolor(COLOURS["bg"])
    ax.axis("off")

    ax.set_title(
        "Backtest Evaluation Metrics",
        fontsize=12, fontweight="bold", color=COLOURS["text"],
        pad=14, loc="left",
    )

    # ── Colour-code cells per column ────────────────────────────────────
    cell_colours = []
    for r_idx, r_name in enumerate(row_names):
        row_colours = []
        is_baseline = r_name not in real_model_names
        for c_idx, (_, _, lower_is_better) in enumerate(_cols):
            col_vals = data[:, c_idx]
            v        = data[r_idx, c_idx]
            mn, mx   = col_vals.min(), col_vals.max()
            rng = mx - mn

            if rng < 1e-9 or col_names[c_idx] == "N Evaluated":
                cell_bg = COLOURS["surface"]
            else:
                # Normalise to [0, 1]; 0 = best, 1 = worst
                norm = (v - mn) / rng
                if not lower_is_better:
                    norm = 1.0 - norm
                # Interpolate: green (best) → surface → red (worst)
                if norm < 0.5:
                    t = norm * 2
                    cell_bg = mcolors.to_hex(
                        _blend(
                            mcolors.to_rgb("#1a4d2e"),   # deep green
                            mcolors.to_rgb(COLOURS["surface"]),
                            t,
                        )
                    )
                else:
                    t = (norm - 0.5) * 2
                    cell_bg = mcolors.to_hex(
                        _blend(
                            mcolors.to_rgb(COLOURS["surface"]),
                            mcolors.to_rgb("#5a1a1a"),   # deep red
                            t,
                        )
                    )

            if is_baseline:
                # Lighten baselines slightly for visual separation
                r, g, b = mcolors.to_rgb(cell_bg)
                cell_bg = mcolors.to_hex(
                    (min(r + 0.04, 1), min(g + 0.04, 1), min(b + 0.04, 1))
                )

            row_colours.append(cell_bg)
        cell_colours.append(row_colours)

    # ── Format cell text ─────────────────────────────────────────────────
    cell_text = []
    for r_idx, r_name in enumerate(row_names):
        row_text = []
        for c_idx, (_, attr, _) in enumerate(_cols):
            v = data[r_idx, c_idx]
            if attr == "n_evaluated":
                row_text.append(f"{int(v):,}")
            elif "accuracy" in attr:
                row_text.append(f"{v:.1%}")
            else:
                row_text.append(f"{v:.4f}")
        cell_text.append(row_text)

    # ── Row labels (italicise baselines) ──────────────────────────────────
    row_labels_display = []
    for r_name in row_names:
        if r_name not in real_model_names:
            row_labels_display.append(f"  {r_name}  ")   # italic-like indent
        else:
            row_labels_display.append(f"  {r_name}  ")

    # ── Render table ─────────────────────────────────────────────────────
    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels_display,
        colLabels=col_names,
        cellColours=cell_colours,
        rowColours=[
            COLOURS["border"] if r not in real_model_names else COLOURS["surface"]
            for r in row_names
        ],
        colColours=[COLOURS["home"]] * len(col_names),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.55)

    # Style header and row label cells
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(COLOURS["border"])
        cell.set_linewidth(0.5)
        if row == 0:                         # column headers
            cell.set_text_props(color=COLOURS["bg"], fontweight="bold", fontsize=8)
        elif col == -1:                       # row labels
            cell.set_text_props(
                color=COLOURS["text"], fontsize=8.5,
                style="italic" if row_names[row - 1] not in real_model_names
                                else "normal",
            )
        else:
            cell.set_text_props(color=COLOURS["text"], fontsize=8.5)

    # ── Legend annotations ────────────────────────────────────────────────
    _green_patch = mpatches.Patch(color="#1a4d2e", label="Best in column")
    _red_patch   = mpatches.Patch(color="#5a1a1a", label="Worst in column")
    _base_patch  = mpatches.Patch(color=COLOURS["border"], label="Baseline row")
    ax.legend(
        handles=[_green_patch, _red_patch, _base_patch],
        loc="upper right", fontsize=7.5, framealpha=0.8,
        bbox_to_anchor=(1.0, 1.0),
    )

    return fig