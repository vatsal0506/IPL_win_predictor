"""Turn cleaned ball-by-ball data into "match state" training rows.

Each row describes the chase *after* a given delivery, using only deliveries
bowled up to and including that one. The label is the final outcome of the
match (did the chasing side win?). No future information - the final score,
the winner, or later deliveries - is ever used as a feature.

``derive_state_features`` is the single source of truth for the numeric
features and is shared by training and inference so both always see exactly
the same transformations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import BALLS_PER_INNINGS, BALLS_PER_OVER, IMPACT_PLAYER_FIRST_SEASON, MAX_WICKETS

CATEGORICAL_FEATURES = ["batting_team", "bowling_team", "venue"]
# Raw inputs a user (or a historical state) supplies.
BASE_NUMERIC_INPUTS = ["target", "current_score", "wickets_lost", "balls_bowled", "impact_player_era"]
NUMERIC_FEATURES = [
    "target",
    "current_score",
    "runs_left",
    "wickets_lost",
    "wickets_left",
    "balls_bowled",
    "balls_left",
    "current_run_rate",
    "required_run_rate",
    "impact_player_era",  # 1 for IPL 2023+ (Impact Player rule), else 0
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "chase_won"

# A required rate above 36 runs/over (6 per legal ball) cannot be achieved
# without extras; clipping keeps the feature bounded near the end of a chase.
MAX_REQUIRED_RUN_RATE = 36.0


def derive_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived chase features from target, current score, wickets and legal balls.

    * runs_left        = target - current_score
    * wickets_left     = 10 - wickets_lost
    * balls_left       = 120 - legal balls bowled (20-over innings)
    * current_run_rate = runs per over so far = score * 6 / legal balls (0 before the first ball)
    * required_run_rate= runs per over needed = runs_left * 6 / balls_left
    """
    out = df.copy()
    out["runs_left"] = out["target"] - out["current_score"]
    out["wickets_left"] = MAX_WICKETS - out["wickets_lost"]
    out["balls_left"] = BALLS_PER_INNINGS - out["balls_bowled"]

    balls = out["balls_bowled"].to_numpy(dtype=float)
    balls_left = out["balls_left"].to_numpy(dtype=float)
    score = out["current_score"].to_numpy(dtype=float)
    runs_left = out["runs_left"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        crr = np.where(balls > 0, score * BALLS_PER_OVER / balls, 0.0)
        rrr = np.where(balls_left > 0, runs_left * BALLS_PER_OVER / balls_left, MAX_REQUIRED_RUN_RATE)
    out["current_run_rate"] = crr
    out["required_run_rate"] = np.clip(rrr, 0.0, MAX_REQUIRED_RUN_RATE)
    return out


def build_chase_states(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Build one row per second-innings delivery (plus the pre-chase state) for each match.

    Only *live* states are kept: runs still needed, wickets in hand and legal
    balls remaining. Finished states (target reached, all out, overs complete)
    have a known result and are handled by rules at inference time instead.
    """
    chase = deliveries[deliveries["innings"] == 2].sort_values(["match_id", "seq"]).copy()
    grouped = chase.groupby("match_id", sort=False)
    # Cumulative totals *after* each delivery - only past information.
    chase["current_score"] = grouped["total_runs"].cumsum()
    chase["wickets_lost"] = grouped["wickets"].cumsum()
    chase["balls_bowled"] = grouped["is_legal"].cumsum().astype(int)
    states = chase[["match_id", "seq", "current_score", "wickets_lost", "balls_bowled"]]

    # The state before the first ball of the chase (0/0 after 0 balls).
    start = pd.DataFrame(
        {"match_id": matches["match_id"], "seq": -1, "current_score": 0, "wickets_lost": 0, "balls_bowled": 0}
    )
    states = pd.concat([start, states], ignore_index=True)

    meta_cols = ["match_id", "date", "season_year", "batting_team", "bowling_team", "venue", "target", TARGET_COLUMN]
    states = states.merge(matches[meta_cols], on="match_id", how="inner")
    states["impact_player_era"] = (states["season_year"] >= IMPACT_PLAYER_FIRST_SEASON).astype(int)
    states = derive_state_features(states)

    live = (states["runs_left"] > 0) & (states["wickets_left"] > 0) & (states["balls_left"] > 0)
    states = states[live].sort_values(["date", "match_id", "seq"]).reset_index(drop=True)
    return states
