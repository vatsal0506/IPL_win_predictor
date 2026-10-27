"""Cleaning and filtering of the canonical IPL data.

The goal is a set of matches whose *chase outcome* is unambiguous, so every
second-innings state can be labelled "did the chasing team win?".

Matches are excluded (with the reason recorded) when:

* the result is a **no result** (abandoned / washed out);
* the match was **tied** - the regulation chase neither succeeded nor failed and
  the winner was decided by a super over, so the label is ambiguous;
* a **D/L (DLS) method** was applied, or the match / target was **reduced below
  20 overs** - the target or available balls changed mid-match, so states built
  on a 20-over, full-target assumption would be wrong;
* metadata is missing or inconsistent (no winner, winner not one of the two
  teams, missing second innings, same team batting in both innings, the
  recorded target disagreeing with the first-innings total + 1, or the recorded
  winner disagreeing with the ball-by-ball result).

Super-over innings (innings 3+) are always discarded.

Usage::

    python -m src.preprocessing
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_raw_data
from src.utils import (
    BALLS_PER_INNINGS,
    CLEAN_DELIVERIES_PATH,
    CLEAN_MATCHES_PATH,
    EXCLUSIONS_PATH,
    MAX_WICKETS,
    OVERS_PER_INNINGS,
    PROCESSED_DATA_DIR,
    get_logger,
    normalize_team,
    normalize_venue,
)

logger = get_logger(__name__)


def clean_deliveries(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Normalise names, remove duplicates / invalid rows and add derived per-ball columns."""
    df = deliveries.copy()
    df["match_id"] = df["match_id"].astype(str)
    df["batting_team"] = df["batting_team"].map(normalize_team)
    df["bowling_team"] = df["bowling_team"].map(normalize_team)

    n_before = len(df)
    df = df.drop_duplicates(subset=["match_id", "innings", "ball_label"], keep="first")
    if len(df) < n_before:
        logger.info("Dropped %d duplicate deliveries", n_before - len(df))

    # Only the two regulation innings; 3+ are super overs.
    df = df[df["innings"].isin([1, 2])]

    for col in ["runs_off_bat", "extras", "wickets"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["is_wide"] = df["is_wide"].fillna(False).astype(bool)
    df["is_noball"] = df["is_noball"].fillna(False).astype(bool)
    df = df.dropna(subset=["innings", "over", "batting_team", "bowling_team"])
    df["innings"] = df["innings"].astype(int)
    df["over"] = df["over"].astype(int)

    # Cricket rules: wides and no-balls are not legal deliveries and must be
    # re-bowled, so they do not count towards the 6 balls of an over. Byes and
    # leg-byes ARE legal deliveries. Every run (bat + all extras) counts.
    df["is_legal"] = ~(df["is_wide"] | df["is_noball"])
    df["total_runs"] = df["runs_off_bat"] + df["extras"]

    df = df.sort_values(["match_id", "innings", "seq"], kind="stable").reset_index(drop=True)
    df["seq"] = df.groupby(["match_id", "innings"]).cumcount()
    return df


def clean_matches(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy()
    df["match_id"] = df["match_id"].astype(str)
    df = df.drop_duplicates(subset="match_id", keep="first")
    for col in ["team1", "team2", "winner", "eliminator"]:
        df[col] = df[col].map(normalize_team)
    df["venue"] = df["venue"].map(normalize_venue)
    df["date"] = pd.to_datetime(df["date"].astype(str).str.replace("/", "-"), errors="coerce")
    # Season labels like "2007/08" or "2020/21" are ambiguous; the calendar year
    # of the match date is the IPL edition (2008, 2020, ...).
    df["season_year"] = df["date"].dt.year
    df["outcome"] = df["outcome"].fillna("result").astype(str).str.lower()
    for col in ["target_runs", "target_overs", "scheduled_overs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_match_table(
    deliveries: pd.DataFrame, matches: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (eligible matches with chase label, exclusions with reasons)."""
    innings_totals = (
        deliveries.groupby(["match_id", "innings"])
        .agg(
            runs=("total_runs", "sum"),
            wickets=("wickets", "sum"),
            legal_balls=("is_legal", "sum"),
            batting_team=("batting_team", "first"),
        )
        .reset_index()
    )
    first = innings_totals[innings_totals["innings"] == 1].set_index("match_id")
    second = innings_totals[innings_totals["innings"] == 2].set_index("match_id")

    m = matches.set_index("match_id")
    m["first_batting_team"] = first["batting_team"]
    m["first_innings_total"] = first["runs"]
    m["chasing_team"] = second["batting_team"]
    m["chase_runs"] = second["runs"]
    m["chase_wickets"] = second["wickets"]
    m["chase_legal_balls"] = second["legal_balls"]

    reasons = pd.Series("", index=m.index, dtype=object)

    def flag(mask: pd.Series, reason: str) -> None:
        mask = mask.fillna(False).astype(bool) & (reasons == "")
        reasons[mask] = reason

    flag(m["team1"].isna() | m["team2"].isna(), "missing teams")
    flag(m["outcome"].eq("no result"), "no result / abandoned")
    flag(m["outcome"].eq("tie") | m["eliminator"].notna(), "tie decided by super over")
    flag(m["method"].notna(), "D/L method applied")
    flag(
        m["scheduled_overs"].fillna(OVERS_PER_INNINGS).ne(OVERS_PER_INNINGS)
        | m["target_overs"].fillna(OVERS_PER_INNINGS).ne(OVERS_PER_INNINGS),
        "reduced-overs match",
    )
    flag(m["winner"].isna(), "no winner recorded")
    flag(~((m["winner"] == m["team1"]) | (m["winner"] == m["team2"])), "winner not one of the teams")
    flag(m["first_innings_total"].isna() | m["chase_runs"].isna(), "missing an innings")
    flag(m["first_batting_team"] == m["chasing_team"], "same team batted in both innings")
    flag(m["date"].isna() | m["venue"].isna(), "missing date or venue")

    # Target: the recorded target where available, otherwise first-innings total + 1.
    m["target"] = m["target_runs"].fillna(m["first_innings_total"] + 1)
    flag(m["target"] != m["first_innings_total"] + 1, "target inconsistent with first innings")

    m["chase_won"] = (m["winner"] == m["chasing_team"]).astype(int)
    reached = m["chase_runs"] >= m["target"]
    flag(reached != m["chase_won"].astype(bool), "recorded winner contradicts ball-by-ball data")
    flag(
        (m["chase_legal_balls"] > BALLS_PER_INNINGS) | (m["chase_wickets"] > MAX_WICKETS),
        "invalid second innings (too many balls or wickets)",
    )

    excluded = (
        m[reasons != ""]
        .assign(reason=reasons[reasons != ""])
        .reset_index()[["match_id", "season_year", "team1", "team2", "outcome", "method", "reason"]]
    )
    keep = m[reasons == ""].reset_index()
    keep = keep.rename(columns={"chasing_team": "batting_team", "first_batting_team": "bowling_team"})
    keep["target"] = keep["target"].astype(int)
    keep["first_innings_total"] = keep["first_innings_total"].astype(int)
    cols = [
        "match_id", "date", "season_year", "venue", "city", "batting_team", "bowling_team",
        "first_innings_total", "target", "chase_runs", "chase_wickets", "chase_legal_balls",
        "winner", "chase_won",
    ]
    keep = keep[cols].sort_values(["date", "match_id"]).reset_index(drop=True)
    return keep, excluded


def run_preprocessing(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_deliveries, raw_matches = load_raw_data()
    deliveries = clean_deliveries(raw_deliveries)
    matches = clean_matches(raw_matches)
    deliveries = deliveries[deliveries["match_id"].isin(matches["match_id"])]

    clean, excluded = build_match_table(deliveries, matches)
    deliveries = deliveries[deliveries["match_id"].isin(clean["match_id"])].reset_index(drop=True)

    logger.info(
        "Matches: %d raw -> %d eligible (%d excluded). Chasing-team win rate: %.3f",
        len(matches), len(clean), len(excluded), clean["chase_won"].mean() if len(clean) else np.nan,
    )
    if len(excluded):
        logger.info("Exclusion reasons:\n%s", excluded["reason"].value_counts().to_string())

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        clean.to_csv(CLEAN_MATCHES_PATH, index=False)
        deliveries.to_csv(CLEAN_DELIVERIES_PATH, index=False)
        excluded.to_csv(EXCLUSIONS_PATH, index=False)
        logger.info("Saved cleaned data to %s", PROCESSED_DATA_DIR)
    return clean, deliveries, excluded


if __name__ == "__main__":
    run_preprocessing()
