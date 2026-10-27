"""Tests for loading, cleaning, cricket rules and leakage-free state construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_deliveries
from src.data_loader import DELIVERY_COLUMNS, MATCH_COLUMNS, DataValidationError, detect_source, load_kaggle
from src.feature_engineering import FEATURE_COLUMNS, build_chase_states, derive_state_features
from src.preprocessing import build_match_table, clean_deliveries, clean_matches
from src.train import chronological_split, temporal_cv_splits
from src.utils import normalize_team, normalize_venue


# ---------------------------------------------------------------- names ---- #
@pytest.mark.parametrize("raw, expected", [
    ("Delhi Daredevils", "Delhi Capitals"),
    ("Kings XI Punjab", "Punjab Kings"),
    ("Royal Challengers Bangalore", "Royal Challengers Bengaluru"),
    ("Rising Pune Supergiants", "Rising Pune Supergiant"),
    ("Deccan Chargers", "Deccan Chargers"),  # defunct franchise is NOT merged into SRH
    ("  Mumbai   Indians ", "Mumbai Indians"),
])
def test_normalize_team(raw, expected):
    assert normalize_team(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Eden Gardens, Kolkata", "Eden Gardens"),
    ("MA Chidambaram Stadium, Chepauk, Chennai", "MA Chidambaram Stadium"),
    ("M.Chinnaswamy Stadium", "M Chinnaswamy Stadium"),
    ("Feroz Shah Kotla", "Arun Jaitley Stadium"),
    ("Punjab Cricket Association Stadium, Mohali", "Punjab Cricket Association IS Bindra Stadium"),
])
def test_normalize_venue(raw, expected):
    assert normalize_venue(raw) == expected


# ---------------------------------------------------------- cricket rules -- #
def test_wides_and_noballs_are_not_legal_but_byes_are():
    d = clean_deliveries(make_deliveries([
        {"runs": 1}, {"extras": 1, "wide": True}, {"runs": 4, "extras": 1, "noball": True},
        {"extras": 1},  # leg-bye: legal delivery
    ]))
    assert d["is_legal"].tolist() == [True, False, False, True]
    assert d["total_runs"].tolist() == [1, 1, 5, 1]


def test_duplicate_deliveries_and_super_overs_removed():
    d = make_deliveries([{"runs": 1, "ball_label": "0.1"}, {"runs": 1, "ball_label": "0.1"},
                         {"runs": 2, "ball_label": "0.2"}, {"innings": 3, "runs": 6, "ball_label": "0.1"}])
    out = clean_deliveries(d)
    assert len(out) == 2
    assert set(out["innings"]) == {2}


def test_derive_state_features_math():
    df = derive_state_features(pd.DataFrame({"target": [181, 150], "current_score": [96, 0],
                                             "wickets_lost": [3, 0], "balls_bowled": [66, 0]}))
    r = df.iloc[0]
    assert r["runs_left"] == 85 and r["wickets_left"] == 7 and r["balls_left"] == 54
    assert r["current_run_rate"] == pytest.approx(96 * 6 / 66)
    assert r["required_run_rate"] == pytest.approx(85 * 6 / 54)
    assert df.iloc[1]["current_run_rate"] == 0.0  # no division by zero before the first ball


def test_required_rate_is_clipped():
    df = derive_state_features(pd.DataFrame({"target": [200], "current_score": [100],
                                             "wickets_lost": [5], "balls_bowled": [119]}))
    assert df.iloc[0]["required_run_rate"] == 36.0


# ------------------------------------------------------ match filtering ---- #
def _match_row(mid, **kw):
    base = {"match_id": mid, "season": "2020", "date": "2020-04-01", "venue": "Wankhede Stadium, Mumbai",
            "city": "Mumbai", "team1": "Mumbai Indians", "team2": "Chennai Super Kings", "winner": "Chennai Super Kings",
            "outcome": "result", "method": None, "eliminator": None, "target_runs": None, "target_overs": 20,
            "scheduled_overs": 20}
    base.update(kw)
    return base


def _two_innings(mid, first_runs, second_runs):
    first = [{"innings": 1, "runs": first_runs, "batting_team": "Mumbai Indians",
              "bowling_team": "Chennai Super Kings", "ball_label": "1.1"}]
    second = [{"innings": 2, "runs": second_runs, "batting_team": "Chennai Super Kings",
               "bowling_team": "Mumbai Indians", "ball_label": "2.1"}]
    return make_deliveries(first + second, match_id=mid)


def test_match_filters_and_label():
    deliveries = clean_deliveries(pd.concat([
        _two_innings("won", 150, 151), _two_innings("lost", 150, 120), _two_innings("nr", 150, 10),
        _two_innings("tie", 150, 150), _two_innings("dl", 150, 140), _two_innings("short", 100, 101),
        _two_innings("bad", 150, 151),
    ]))
    matches = clean_matches(pd.DataFrame([
        _match_row("won"), _match_row("lost", winner="Mumbai Indians"),
        _match_row("nr", outcome="no result", winner=None),
        _match_row("tie", outcome="tie", winner=None, eliminator="Mumbai Indians"),
        _match_row("dl", method="D/L", target_runs=140, target_overs=17),
        _match_row("short", scheduled_overs=8, target_overs=8),
        _match_row("bad", winner="Mumbai Indians"),  # contradicts ball-by-ball data
    ]))
    clean, excluded = build_match_table(deliveries, matches)
    assert set(clean["match_id"]) == {"won", "lost"}
    assert clean.set_index("match_id")["chase_won"].to_dict() == {"won": 1, "lost": 0}
    assert clean.set_index("match_id").loc["won", "target"] == 151
    reasons = excluded.set_index("match_id")["reason"]
    assert reasons["nr"] == "no result / abandoned"
    assert reasons["tie"] == "tie decided by super over"
    assert reasons["dl"] == "D/L method applied"
    assert reasons["short"] == "reduced-overs match"
    assert reasons["bad"] == "recorded winner contradicts ball-by-ball data"


# --------------------------------------------------- state construction ---- #
def test_chase_states_use_only_past_deliveries():
    first = [{"innings": 1, "runs": 6, "batting_team": "Mumbai Indians", "bowling_team": "Chennai Super Kings",
              "ball_label": f"1.{i}"} for i in range(5)]  # 30 runs -> target 31
    second = [{"runs": 4}, {"extras": 1, "wide": True}, {"runs": 0, "wicket": 1}, {"runs": 6},
              {"runs": 6}, {"runs": 6}, {"runs": 6}, {"runs": 6}]  # reaches 35 >= 31
    second = [{**r, "innings": 2, "batting_team": "Chennai Super Kings", "bowling_team": "Mumbai Indians"}
              for r in second]
    deliveries = clean_deliveries(make_deliveries(first + second))
    matches = clean_matches(pd.DataFrame([_match_row("m1")]))
    clean, _ = build_match_table(deliveries, matches)
    states = build_chase_states(deliveries, clean)

    # Pre-chase state then one state per delivery while the chase is live (the winning ball is excluded).
    assert states["current_score"].tolist() == [0, 4, 5, 5, 11, 17, 23, 29]
    assert states["wickets_lost"].tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    assert states["balls_bowled"].tolist() == [0, 1, 1, 2, 3, 4, 5, 6]  # wide does not count
    assert (states["runs_left"] > 0).all()
    assert states["chase_won"].eq(1).all()
    # Features never contain the final result / final score.
    assert "chase_runs" not in FEATURE_COLUMNS and "winner" not in FEATURE_COLUMNS
    assert not {"chase_runs", "chase_wickets", "winner"} & set(states.columns)


# ------------------------------------------------------- data loaders ------ #
def test_cricsheet_loader_schema_and_exclusions(synthetic_pipeline):
    assert list(synthetic_pipeline["raw_deliveries"].columns) == DELIVERY_COLUMNS
    assert list(synthetic_pipeline["raw_matches"].columns) == MATCH_COLUMNS
    excl = synthetic_pipeline["excluded"]
    assert "no result / abandoned" in set(excl["reason"])
    states = synthetic_pipeline["states"]
    assert states["wickets_left"].between(1, 10).all()
    assert states["balls_left"].between(1, 120).all()
    assert states[FEATURE_COLUMNS].notna().all().all()


def test_kaggle_style_loader(tmp_path):
    pd.DataFrame([{"id": 1, "season": "2019", "date": "2019-04-01", "venue": "Eden Gardens", "city": "Kolkata",
                   "team1": "Kolkata Knight Riders", "team2": "Delhi Daredevils", "winner": "Delhi Daredevils",
                   "result": "wickets", "method": np.nan}]).to_csv(tmp_path / "matches.csv", index=False)
    rows = []
    for inn, bat, bowl in [(1, "Kolkata Knight Riders", "Delhi Daredevils"),
                           (2, "Delhi Daredevils", "Kolkata Knight Riders")]:
        for b in range(1, 4):
            rows.append({"match_id": 1, "inning": inn, "batting_team": bat, "bowling_team": bowl, "over": 0,
                         "ball": b, "batsman_runs": 4 if inn == 2 else 1, "extra_runs": 0, "total_runs": 0,
                         "extras_type": "wides" if b == 2 and inn == 1 else np.nan, "is_wicket": 0,
                         "dismissal_kind": np.nan})
    pd.DataFrame(rows).to_csv(tmp_path / "deliveries.csv", index=False)

    source, paths = detect_source(tmp_path)
    assert source == "kaggle"
    deliveries, matches = load_kaggle(paths["matches"], paths["deliveries"])
    assert list(deliveries.columns) == DELIVERY_COLUMNS
    assert deliveries["is_wide"].sum() == 1
    clean, _ = build_match_table(clean_deliveries(deliveries), clean_matches(matches))
    assert clean.iloc[0]["batting_team"] == "Delhi Capitals"  # normalised
    assert clean.iloc[0]["chase_won"] == 1


def test_missing_dataset_gives_helpful_error(tmp_path):
    with pytest.raises(DataValidationError, match="--download"):
        detect_source(tmp_path)


# ------------------------------------------------------------ splitting ---- #
def test_chronological_split_has_no_match_leakage(synthetic_pipeline):
    states = synthetic_pipeline["states"]
    train, test = chronological_split(states, test_seasons=1)
    assert not set(train["match_id"]) & set(test["match_id"])
    assert train["date"].max() < test["date"].min()

    for tr, va in temporal_cv_splits(train, n_splits=2):
        assert not set(train.iloc[tr]["match_id"]) & set(train.iloc[va]["match_id"])
        assert train.iloc[tr]["date"].max() < train.iloc[va]["date"].min()  # never train on the future
