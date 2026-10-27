"""Tests for inference: probability outputs, edge cases, invalid inputs and model loading."""

from __future__ import annotations

import joblib
import numpy as np
import pytest

from src.feature_engineering import FEATURE_COLUMNS
from src.predict import InvalidMatchStateError, MatchState, ModelNotAvailableError, WinProbabilityPredictor

TEAM_A, TEAM_B = "Synthetic Lions", "Synthetic Tigers"
VENUE = "Test Ground A"


def state(**kw) -> MatchState:
    base = dict(batting_team=TEAM_A, bowling_team=TEAM_B, venue=VENUE, target=170, current_score=80,
                wickets_lost=3, overs_completed=10, balls_in_over=0)
    base.update(kw)
    return MatchState(**base)


# ------------------------------------------------------------ probabilities #
def test_probability_in_range_and_complementary(synthetic_predictor):
    result = synthetic_predictor.predict(state())
    assert result.status == "in_progress"
    assert 0.0 <= result.batting_win_probability <= 1.0
    assert result.batting_win_probability + result.bowling_win_probability == pytest.approx(1.0)


def test_pre_chase_state_is_live(synthetic_predictor):
    result = synthetic_predictor.predict(state(current_score=0, wickets_lost=0, overs_completed=0))
    assert result.is_live and 0 < result.batting_win_probability < 1


def test_predict_many_matches_single_predictions(synthetic_predictor):
    states = [state(wickets_lost=w) for w in range(0, 10)]
    batch = synthetic_predictor.predict_many(states)
    single = [synthetic_predictor.predict(s).batting_win_probability for s in states]
    assert np.allclose(batch, single)


def test_inference_features_match_training_schema(synthetic_predictor, synthetic_pipeline):
    """A historical state passed through the inference path must produce identical features."""
    row = synthetic_pipeline["states"].iloc[57]
    s = MatchState(row["batting_team"], row["bowling_team"], row["venue"], int(row["target"]),
                   int(row["current_score"]), int(row["wickets_lost"]), int(row["balls_bowled"]) // 6,
                   int(row["balls_bowled"]) % 6, bool(row["impact_player_era"]))
    frame = synthetic_predictor.feature_frame([s])
    assert list(frame.columns) == FEATURE_COLUMNS
    for col in FEATURE_COLUMNS:
        assert frame.iloc[0][col] == pytest.approx(row[col]) if col not in ("batting_team", "bowling_team", "venue") \
            else frame.iloc[0][col] == row[col]


# ------------------------------------------------------- completed states -- #
@pytest.mark.parametrize("kw, status", [
    (dict(current_score=170), "batting_won"),               # target reached exactly
    (dict(current_score=174), "batting_won"),               # finished with a boundary
    (dict(wickets_lost=10), "bowling_won"),                 # all out
    (dict(overs_completed=20, current_score=150), "bowling_won"),  # overs complete
    (dict(overs_completed=20, current_score=169), "tie"),   # scores level
    (dict(wickets_lost=10, current_score=169), "tie"),
])
def test_completed_matches_have_no_probability(synthetic_predictor, kw, status):
    result = synthetic_predictor.predict(state(**kw))
    assert result.status == status
    assert result.batting_win_probability is None and result.bowling_win_probability is None
    assert result.message


# --------------------------------------------------------- invalid inputs -- #
@pytest.mark.parametrize("kw, msg", [
    (dict(bowling_team=TEAM_A), "must be different"),
    (dict(batting_team=""), "must be selected"),
    (dict(wickets_lost=11), "Wickets lost"),
    (dict(wickets_lost=-1), "Wickets lost"),
    (dict(balls_in_over=6), "Balls in the current over"),
    (dict(overs_completed=21), "Overs completed"),
    (dict(overs_completed=20, balls_in_over=2), "only 20 overs"),
    (dict(current_score=-5), "negative"),
    (dict(target=0), "Target"),
    (dict(target=1000), "Target"),
    (dict(current_score=300), "beyond the target"),
    (dict(current_score=12.5), "whole number"),
    (dict(target="abc"), "whole number"),
    (dict(wickets_lost=None), "whole number"),
])
def test_invalid_states_raise(synthetic_predictor, kw, msg):
    with pytest.raises(InvalidMatchStateError, match=msg):
        synthetic_predictor.predict(state(**kw))


def test_unknown_team_and_venue_handled_with_warning(synthetic_predictor):
    result = synthetic_predictor.predict(state(batting_team="Brand New Franchise", venue="Moon Stadium"))
    assert result.is_live and 0.0 <= result.batting_win_probability <= 1.0
    assert any("Brand New Franchise" in w for w in result.warnings)
    assert any("Venue" in w for w in result.warnings)


# ----------------------------------------------------------- model loading - #
def test_missing_model_gives_helpful_error(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="python -m src.train"):
        WinProbabilityPredictor.load(tmp_path / "nope.joblib")


def test_corrupted_model_gives_helpful_error(tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a model")
    with pytest.raises(ModelNotAvailableError, match="Could not load"):
        WinProbabilityPredictor.load(bad)


def test_synthetic_model_refused_by_default(synthetic_model_path):
    with pytest.raises(ModelNotAvailableError, match="SYNTHETIC"):
        WinProbabilityPredictor.load(synthetic_model_path)


def test_incompatible_feature_schema_rejected(tmp_path, synthetic_model_path):
    bundle = joblib.load(synthetic_model_path)
    bundle["metadata"]["feature_columns"] = ["batting_team"]
    path = tmp_path / "old_schema.joblib"
    joblib.dump(bundle, path)
    with pytest.raises(ModelNotAvailableError, match="incompatible"):
        WinProbabilityPredictor.load(path, allow_synthetic=True)


def test_model_artifact_roundtrip(synthetic_model_path):
    bundle = joblib.load(synthetic_model_path)
    assert {"model", "metadata"} <= set(bundle)
    meta = bundle["metadata"]
    assert meta["feature_columns"] == FEATURE_COLUMNS
    assert meta["data_kind"] == "synthetic"
    assert set(meta["comparison"][0]) >= {"model", "cv_log_loss", "test_log_loss", "test_brier"}


# ------------------------------------------------ real model sanity checks - #
REAL_A, REAL_B, REAL_VENUE = "Mumbai Indians", "Chennai Super Kings", "Wankhede Stadium"


def test_real_model_is_real_and_sane(real_predictor):
    assert real_predictor.metadata["data_kind"] == "real"
    easy = real_predictor.predict(MatchState(REAL_A, REAL_B, REAL_VENUE, 160, 150, 2, 17, 0))
    hard = real_predictor.predict(MatchState(REAL_A, REAL_B, REAL_VENUE, 220, 120, 7, 17, 0))
    assert easy.batting_win_probability > 0.85
    assert hard.batting_win_probability < 0.10


def test_real_model_era_flag_is_used(real_predictor):
    modern = MatchState(REAL_A, REAL_B, REAL_VENUE, 200, 110, 3, 12, 0, impact_player_era=True)
    classic = MatchState(REAL_A, REAL_B, REAL_VENUE, 200, 110, 3, 12, 0, impact_player_era=False)
    p_modern, p_classic = real_predictor.predict_many([modern, classic])
    assert 0 < p_modern < 1 and 0 < p_classic < 1


def test_real_model_monotonic_in_wickets_and_runs(real_predictor):
    """Losing wickets should not help; scoring more runs should not hurt (allowing tiny noise)."""
    wk = real_predictor.predict_many([MatchState(REAL_A, REAL_B, REAL_VENUE, 180, 90, w, 11, 0) for w in range(10)])
    assert wk[0] > wk[-1]
    assert np.all(np.diff(wk) <= 0.03)
    rn = real_predictor.predict_many([MatchState(REAL_A, REAL_B, REAL_VENUE, 180, r, 4, 12, 0)
                                      for r in range(60, 171, 10)])
    assert rn[-1] > rn[0]
    assert np.all(np.diff(rn) >= -0.03)
