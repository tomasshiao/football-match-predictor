(() => {
  "use strict";

  const form          = document.getElementById("predict-form");
  const submitBtn     = document.getElementById("submit-btn");
  const homeSelect    = document.getElementById("home-team");
  const awaySelect    = document.getElementById("away-team");
  const dateInput     = document.getElementById("match-date");
  const neutralCheck  = document.getElementById("neutral-venue");
  const playoffCheck  = document.getElementById("is-playoff");

  const results       = document.getElementById("results");
  const errorBanner   = document.getElementById("error-banner");
  const errorDetail   = document.getElementById("error-detail");
  const successPanel  = document.getElementById("success-panel");
  const statStrip     = document.getElementById("stat-strip");

  const CHART_IMG_IDS = {
    score_heatmap:           "chart-score-heatmap",
    outcome_probabilities:   "chart-outcome-probabilities",
    top_scorelines:          "chart-top-scorelines",
    backtest_metrics_table:  "chart-backtest-metrics-table",
  };

  let homeTomSelect = null;
  let awayTomSelect = null;

  /**
   * Fetch the full team list once and use it to populate both pickers.
   * Both selects share the same option list — nothing here excludes a
   * team from one side just because it's chosen on the other; that's
   * enforced separately at submit time so the two dropdowns don't need
   * to stay in sync with each other while the user is still choosing.
   */
  async function loadTeams() {
    const response = await fetch("/teams");
    if (!response.ok) {
      throw new Error(`Failed to load team list (HTTP ${response.status})`);
    }
    const teams = await response.json();

    for (const { code, name } of teams) {
      const optionHtml = `<option value="${code}">${name} (${code})</option>`;
      homeSelect.insertAdjacentHTML("beforeend", optionHtml);
      awaySelect.insertAdjacentHTML("beforeend", optionHtml);
    }

    const tomSelectOptions = {
      placeholder: "Type to search…",
      maxOptions: null,
      allowEmptyOption: false,
    };
    homeTomSelect = new TomSelect(homeSelect, tomSelectOptions);
    awayTomSelect = new TomSelect(awaySelect, tomSelectOptions);
  }

  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    submitBtn.classList.toggle("loading", isLoading);
  }

  function showError(message) {
    results.hidden = false;
    errorBanner.hidden = false;
    successPanel.hidden = true;
    errorDetail.textContent = message;
  }

  function pct(x) {
    return `${(x * 100).toFixed(1)}%`;
  }

  function renderStatStrip(payload) {
    const outcome = payload.outcome || {};
    statStrip.innerHTML = "";

    const entries = [
      ["Home win", outcome.home_win],
      ["Draw", outcome.draw],
      ["Away win", outcome.away_win],
    ];

    for (const [label, value] of entries) {
      if (value === undefined) continue;
      const el = document.createElement("div");
      el.className = "stat";
      el.innerHTML = `
        <span class="stat-value">${pct(value)}</span>
        <span class="stat-label">${label}</span>
      `;
      statStrip.appendChild(el);
    }
  }

  function renderCharts(charts) {
    for (const [key, imgId] of Object.entries(CHART_IMG_IDS)) {
      const img = document.getElementById(imgId);
      const dataUrl = charts ? charts[key] : null;
      const card = img.closest(".chart-card");
      if (dataUrl) {
        img.src = dataUrl;
        card.hidden = false;
      } else {
        // Chart generation is best-effort server-side; hide any card
        // whose image didn't come back rather than showing a broken image.
        card.hidden = true;
      }
    }
  }

  function showSuccess(payload) {
    results.hidden = false;
    errorBanner.hidden = true;
    successPanel.hidden = false;
    renderStatStrip(payload);
    renderCharts(payload.charts);
  }

  async function handleSubmit(event) {
    event.preventDefault();

    const homeCode = homeTomSelect ? homeTomSelect.getValue() : "";
    const awayCode = awayTomSelect ? awayTomSelect.getValue() : "";

    if (!homeCode || !awayCode) {
      showError("Choose a home team and an away team.");
      return;
    }
    if (homeCode === awayCode) {
      showError("Home and away teams must be different.");
      return;
    }
    if (!dateInput.value) {
      showError("Choose a match date.");
      return;
    }

    const body = {
      home_team_fifa_code: homeCode,
      away_team_fifa_code: awayCode,
      match_date: dateInput.value,
      is_neutral_venue: neutralCheck.checked,
      is_playoff: playoffCheck.checked,
    };

    setLoading(true);
    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const payload = await response.json().catch(() => null);

      if (!response.ok) {
        const detail = payload && payload.detail
          ? payload.detail
          : `Request failed (HTTP ${response.status}).`;
        showError(detail);
        return;
      }

      showSuccess(payload);
    } catch (err) {
      showError(`Could not reach the prediction service: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  form.addEventListener("submit", handleSubmit);

  loadTeams().catch((err) => {
    showError(`Could not load the team list: ${err.message}`);
  });
})();