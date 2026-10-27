"""UI-independent helpers for the Streamlit app (kept here so they can be unit-tested)."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from src.predict import MatchState, PredictionResult, WinProbabilityPredictor
from src.utils import ACTIVE_TEAMS, BALLS_PER_OVER, UNKNOWN_VENUE, format_overs


def team_options(metadata: dict, include_defunct: bool = False) -> list[str]:
    """Active franchises first; defunct ones (seen in training) optionally appended."""
    known = set(metadata.get("teams", {}))
    active = [t for t in ACTIVE_TEAMS if t in known] or sorted(known)
    if include_defunct:
        active += sorted(known - set(active))
    return active


def venue_options(metadata: dict) -> list[str]:
    """Venues ordered by how many training matches they hosted, plus a generic fallback."""
    venues = metadata.get("venues", {})
    return sorted(venues, key=lambda v: (-venues[v], v)) + [UNKNOWN_VENUE]


def probability_label(p: float) -> str:
    """Express a probability in words without implying certainty."""
    if p >= 0.85:
        return "Strong favourite"
    if p >= 0.65:
        return "Favourite"
    if p > 0.55:
        return "Slight edge"
    if p >= 0.45:
        return "Evenly poised"
    if p >= 0.35:
        return "Slight underdog"
    if p >= 0.15:
        return "Underdog"
    return "Long shot"


def situation_sentence(state: MatchState, result: PredictionResult) -> str:
    s = result.summary
    if not result.is_live:
        return result.message
    balls = "ball" if s.balls_left == 1 else "balls"
    runs = "run" if s.runs_left == 1 else "runs"
    return (f"{state.batting_team} need **{s.runs_left} {runs}** from **{s.balls_left} {balls}** "
            f"with **{s.wickets_left} wicket{'s' if s.wickets_left != 1 else ''}** in hand "
            f"({state.current_score}/{state.wickets_lost} after {format_overs(state.balls_bowled)} overs, "
            f"chasing {state.target}).")


def what_if_wickets(predictor: WinProbabilityPredictor, state: MatchState) -> pd.DataFrame:
    """Win probability at the current score/overs for every possible number of wickets lost."""
    candidates = [replace(state, wickets_lost=w) for w in range(0, 10)]
    probs = predictor.predict_many(candidates)
    return pd.DataFrame({"wickets_lost": range(0, 10), "win_probability": probs,
                         "is_current": [w == state.wickets_lost for w in range(0, 10)]})


def what_if_next_over(predictor: WinProbabilityPredictor, state: MatchState, max_runs: int = 24) -> pd.DataFrame:
    """Win probability after the next full over for different runs scored (no wicket falls).

    Only scenarios that leave the chase live are scored by the model; the
    target being reached is shown as 1.0 (a certain result, not a prediction).
    """
    balls_after = state.balls_bowled + BALLS_PER_OVER
    if balls_after >= 120:
        raise ValueError("Next-over scenarios need more than one over remaining.")
    rows = []
    for runs in range(0, max_runs + 1):
        score = state.current_score + runs
        if score >= state.target:
            rows.append({"runs_in_over": runs, "win_probability": 1.0, "decided": True})
            continue
        nxt = replace(state, current_score=score, overs_completed=balls_after // BALLS_PER_OVER,
                      balls_in_over=balls_after % BALLS_PER_OVER)
        rows.append({"runs_in_over": runs, "win_probability": float(predictor.predict_many([nxt])[0]),
                     "decided": False})
    return pd.DataFrame(rows)
