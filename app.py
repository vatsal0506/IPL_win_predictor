"""IPL Win Probability Estimator - Streamlit dashboard.

Run locally:  streamlit run app.py
The app only performs inference with the saved model in ``models/``; it never
retrains. All cricket rules and model calls live in ``src/``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_logic import probability_label, situation_sentence, team_options, venue_options, what_if_next_over, \
    what_if_wickets
from src.predict import InvalidMatchStateError, MatchState, ModelNotAvailableError, WinProbabilityPredictor
from src.utils import FIGURES_DIR, MODEL_PATH, OVERS_PER_INNINGS, format_overs

BAT_COLOR = "#0B6E4F"   # batting side (pitch green)
BOWL_COLOR = "#C0392B"  # bowling side (cricket-ball red)

st.set_page_config(page_title="IPL Win Probability Estimator", page_icon="🏏", layout="wide")

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 1.6rem; max-width: 1200px; }}
      .hero {{ background: linear-gradient(120deg, #0B3D2E 0%, #0B6E4F 55%, #14915F 100%);
               color: #fff; padding: 1.3rem 1.6rem; border-radius: 14px; margin-bottom: 1.2rem; }}
      .hero h1 {{ color: #fff; font-size: 2.0rem; margin: 0 0 .25rem 0; padding: 0; }}
      .hero p {{ color: #E3F4EC; margin: 0; font-size: .98rem; }}
      .prob-card {{ border-radius: 14px; padding: 1.1rem 1.3rem; color: #fff; }}
      .prob-card .team {{ font-size: 1.0rem; opacity: .92; }}
      .prob-card .value {{ font-size: 2.8rem; font-weight: 700; line-height: 1.1; }}
      .prob-card .tag {{ font-size: .85rem; opacity: .9; }}
      .bat {{ background: {BAT_COLOR}; }}
      .bowl {{ background: {BOWL_COLOR}; }}
      .disclaimer {{ font-size: .82rem; color: #6b7280; border-top: 1px solid #e5e7eb;
                     padding-top: .8rem; margin-top: 1.5rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# Optional overrides (see .env.example). IPL_ALLOW_SYNTHETIC_MODEL exists only for automated UI tests.
MODEL_FILE = Path(os.environ.get("IPL_MODEL_PATH", MODEL_PATH))
ALLOW_SYNTHETIC = os.environ.get("IPL_ALLOW_SYNTHETIC_MODEL") == "1"


@st.cache_resource(show_spinner="Loading trained model ...")
def load_predictor(path: str, allow_synthetic: bool) -> WinProbabilityPredictor:
    return WinProbabilityPredictor.load(Path(path), allow_synthetic=allow_synthetic)


st.markdown(
    """<div class="hero"><h1>🏏 IPL Win Probability Estimator</h1>
    <p>Machine-learning estimate of the chasing team's chance of winning, from the current match situation.
    Enter the situation below - this is a scenario simulator, not a live-score feed.</p></div>""",
    unsafe_allow_html=True,
)

try:
    predictor = load_predictor(str(MODEL_FILE), ALLOW_SYNTHETIC)
except ModelNotAvailableError as exc:
    st.error("**The trained model is not available.**")
    st.code(str(exc), language="text")
    st.stop()

meta = predictor.metadata
if meta.get("data_kind") != "real":
    st.warning("TEST MODE: this model was trained on SYNTHETIC data. Its probabilities are meaningless.")

# --------------------------------------------------------------------------- #
# Sidebar: match setup
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Match setup")
    include_defunct = st.checkbox("Include former franchises", value=False,
                                  help="e.g. Deccan Chargers, Gujarat Lions - teams no longer in the IPL.")
    teams = team_options(meta, include_defunct)
    batting_team = st.selectbox("Batting team (chasing)", teams, index=teams.index("Mumbai Indians")
                                if "Mumbai Indians" in teams else 0, key="batting_team")
    bowling_choices = [t for t in teams if t != batting_team]
    bowling_team = st.selectbox("Bowling team (defending)", bowling_choices,
                                index=bowling_choices.index("Chennai Super Kings")
                                if "Chennai Super Kings" in bowling_choices else 0, key="bowling_team")
    venue = st.selectbox("Venue", venue_options(meta), key="venue",
                         help="Venues are ordered by number of training matches. Unlisted grounds use a generic "
                              "venue effect.")
    target = st.number_input("Target (runs to win)", min_value=1, max_value=400, value=181, step=1, key="target",
                             help="First-innings total + 1.")
    impact_era = st.toggle("Impact Player rule era (IPL 2023+)", value=True, key="impact_era",
                           help="On = current rules, which brought higher scores. Off = pre-2023 conditions.")
    st.caption("Model assumes a full 20-over chase (no rain reduction).")

# --------------------------------------------------------------------------- #
# Main: current situation
# --------------------------------------------------------------------------- #
st.subheader("Current situation in the chase")
c1, c2, c3, c4 = st.columns(4)
current_score = c1.number_input("Current score", min_value=0, max_value=420, value=96, step=1, key="score")
wickets_lost = c2.number_input("Wickets lost", min_value=0, max_value=10, value=3, step=1, key="wickets")
overs_completed = c3.number_input("Overs completed", min_value=0, max_value=OVERS_PER_INNINGS, value=11, step=1,
                                  key="overs")
balls_in_over = c4.number_input("Balls into current over", min_value=0, max_value=5, value=2, step=1, key="balls",
                                help="Legal balls only - wides and no-balls are re-bowled and do not count.")

state = MatchState(batting_team, bowling_team, venue, int(target), int(current_score), int(wickets_lost),
                   int(overs_completed), int(balls_in_over), impact_era)

try:
    result = predictor.predict(state)
except InvalidMatchStateError as exc:
    st.error(f"**Invalid match situation.** {exc}")
    st.stop()

summary = result.summary
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Runs needed", max(summary.runs_left, 0))
m2.metric("Balls left", summary.balls_left, help=f"{summary.balls_left / 6:.1f} overs remaining")
m3.metric("Wickets in hand", summary.wickets_left)
m4.metric("Current run rate", f"{summary.current_run_rate:.2f}")
m5.metric("Required run rate", f"{summary.required_run_rate:.2f}" if summary.required_run_rate is not None else "-")

st.markdown(situation_sentence(state, result))

if not result.is_live:
    # A finished chase has a known outcome - no probability is shown.
    st.info(f"**Match complete.** {result.message}")
    st.stop()

for warning in result.warnings:
    st.warning(warning)

p_bat = result.batting_win_probability
p_bowl = result.bowling_win_probability

st.subheader("Estimated win probability")
left, right = st.columns(2)
left.markdown(f"""<div class="prob-card bat"><div class="team">{batting_team} (batting)</div>
    <div class="value">{p_bat:.1%}</div><div class="tag">{probability_label(p_bat)}</div></div>""",
              unsafe_allow_html=True)
right.markdown(f"""<div class="prob-card bowl"><div class="team">{bowling_team} (bowling)</div>
    <div class="value">{p_bowl:.1%}</div><div class="tag">{probability_label(p_bowl)}</div></div>""",
               unsafe_allow_html=True)

bar = go.Figure()
bar.add_bar(y=[""], x=[p_bat], orientation="h", name=batting_team, marker_color=BAT_COLOR,
            text=[f"{batting_team} {p_bat:.0%}"], textposition="inside", insidetextanchor="start")
bar.add_bar(y=[""], x=[p_bowl], orientation="h", name=bowling_team, marker_color=BOWL_COLOR,
            text=[f"{p_bowl:.0%} {bowling_team}"], textposition="inside", insidetextanchor="end")
bar.update_layout(barmode="stack", height=95, margin=dict(l=0, r=0, t=8, b=0), showlegend=False,
                  xaxis=dict(range=[0, 1], tickformat=".0%", showgrid=False), yaxis=dict(showticklabels=False))
st.plotly_chart(bar, width="stretch", config={"displayModeBar": False})

# --------------------------------------------------------------------------- #
# Scenario exploration
# --------------------------------------------------------------------------- #
tab_wkts, tab_over, tab_model = st.tabs(["What if: wickets", "What if: next over", "Model information"])

with tab_wkts:
    df = what_if_wickets(predictor, state)
    fig = go.Figure(go.Bar(x=df["wickets_lost"], y=df["win_probability"],
                           marker_color=[BAT_COLOR if c else "#9BC9B5" for c in df["is_current"]],
                           text=[f"{p:.0%}" for p in df["win_probability"]], textposition="outside"))
    fig.update_layout(height=330, margin=dict(l=0, r=0, t=30, b=0), yaxis=dict(range=[0, 1.1], tickformat=".0%"),
                      xaxis_title="Wickets lost", yaxis_title=f"{batting_team} win probability",
                      title=f"Same score ({current_score}) after {format_overs(state.balls_bowled)} overs, "
                            "different wickets lost")
    st.plotly_chart(fig, width="stretch")
    st.caption("Dark bar = the current situation.")

with tab_over:
    if summary.balls_left > 6:
        df = what_if_next_over(predictor, state)
        fig = go.Figure(go.Scatter(x=df["runs_in_over"], y=df["win_probability"], mode="lines+markers",
                                   line=dict(color=BAT_COLOR)))
        fig.add_hline(y=p_bat, line_dash="dot", line_color="grey", annotation_text="now")
        fig.update_layout(height=330, margin=dict(l=0, r=0, t=30, b=0), yaxis=dict(range=[0, 1], tickformat=".0%"),
                          xaxis_title="Runs scored in the next over (no wicket)",
                          yaxis_title=f"{batting_team} win probability",
                          title="How the next six legal balls could swing the chase")
        st.plotly_chart(fig, width="stretch")
        st.caption("Points at 100% mean the target is reached in that over (a certain result, not a prediction).")
    else:
        st.info("Less than one over remains - next-over scenarios are not shown.")

with tab_model:
    tm = meta["test_metrics"]
    st.markdown(
        f"""
**Model:** {meta['model_name']}  ·  **Trained on:** IPL {meta['train_seasons'][0]}-{meta['train_seasons'][1]}
({meta['n_train_matches']:,} matches, {meta['n_train_states']:,} match states)  ·  **Tested on:** IPL
{', '.join(map(str, meta['test_seasons']))} ({meta['n_test_matches']:,} unseen matches)

**How it works.** Every ball of every completed 20-over IPL chase becomes a training example describing the
situation *at that moment* (teams, venue, target, score, wickets, legal balls, run rates), labelled with whether the
chasing side went on to win. Logistic Regression, Random Forest and Gradient Boosting models were tuned with
`GridSearchCV` using season-by-season, expanding-window cross-validation (whole matches never split across folds),
and the model with the best cross-validated **log loss** was selected. The most recent seasons were held out entirely
and used once for the final evaluation below.
        """
    )
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Test log loss", f"{tm['log_loss']:.3f}", help="Lower is better; measures probability quality.")
    k2.metric("Brier score", f"{tm['brier']:.3f}", help="Mean squared error of probabilities; lower is better.")
    k3.metric("Calibration error", f"{tm['ece']:.3f}", help="Expected calibration error; lower is better.")
    k4.metric("ROC-AUC", f"{tm['roc_auc']:.3f}")
    k5.metric("Accuracy @ 50%", f"{tm['accuracy']:.1%}")

    comp = pd.DataFrame(meta["comparison"])
    cols = {"model": "Model", "cv_log_loss": "CV log loss", "test_log_loss": "Test log loss", "test_brier": "Brier",
            "test_ece": "ECE", "test_roc_auc": "ROC-AUC", "test_accuracy": "Accuracy", "test_f1": "F1"}
    st.dataframe(comp[list(cols)].rename(columns=cols).round(4), hide_index=True, width="stretch")
    calib_png = FIGURES_DIR / "calibration_curves.png"
    if calib_png.exists():
        st.image(str(calib_png), caption="Calibration on held-out test seasons", width=520)
    st.caption(f"Data: {meta['data_source']}. Model trained {meta['trained_at']} with scikit-learn "
               f"{meta['sklearn_version']}.")

st.markdown(
    """<div class="disclaimer"><b>Disclaimer:</b> probabilities are statistical estimates learned from historical
    IPL matches. They describe how often similar situations were won in the past and are <b>not guarantees</b> of any
    match result. The app is not connected to live scores; it does not account for player form, injuries, pitch,
    weather, dew, impact-player substitutions or rain-reduced (DLS) matches. Not for betting.</div>""",
    unsafe_allow_html=True,
)
