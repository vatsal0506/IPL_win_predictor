"""Tests for app helpers and a headless run of the Streamlit app via ``AppTest``."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.app_logic import (
    probability_label,
    situation_sentence,
    team_options,
    venue_options,
    what_if_next_over,
    what_if_wickets,
)
from src.predict import MatchState
from src.utils import UNKNOWN_VENUE

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
META = {"teams": {"Mumbai Indians": 200, "Deccan Chargers": 70, "Chennai Super Kings": 190},
        "venues": {"Eden Gardens": 30, "Wankhede Stadium": 80}}


def test_team_options_active_first_and_defunct_optional():
    assert team_options(META) == ["Chennai Super Kings", "Mumbai Indians"]
    assert team_options(META, include_defunct=True)[-1] == "Deccan Chargers"


def test_venue_options_sorted_by_frequency_with_fallback():
    assert venue_options(META) == ["Wankhede Stadium", "Eden Gardens", UNKNOWN_VENUE]


@pytest.mark.parametrize("p, label", [(0.95, "Strong favourite"), (0.5, "Evenly poised"), (0.05, "Long shot")])
def test_probability_label(p, label):
    assert probability_label(p) == label


def test_situation_sentence_and_what_ifs(synthetic_predictor):
    s = MatchState("Synthetic Lions", "Synthetic Tigers", "Test Ground A", 170, 80, 3, 10, 2)
    result = synthetic_predictor.predict(s)
    text = situation_sentence(s, result)
    assert "90 runs" in text and "58 balls" in text and "7 wickets" in text

    wk = what_if_wickets(synthetic_predictor, s)
    assert len(wk) == 10 and wk["win_probability"].between(0, 1).all()
    assert wk.loc[wk["is_current"], "wickets_lost"].item() == 3

    nxt = what_if_next_over(synthetic_predictor, s)
    assert nxt["win_probability"].between(0, 1).all()

    with pytest.raises(ValueError):
        what_if_next_over(synthetic_predictor, MatchState("Synthetic Lions", "Synthetic Tigers", "x", 170, 80, 3, 19, 1))


# ------------------------------------------------------------- Streamlit --- #
@pytest.fixture
def app_env(monkeypatch, synthetic_model_path):
    monkeypatch.setenv("IPL_MODEL_PATH", str(synthetic_model_path))
    monkeypatch.setenv("IPL_ALLOW_SYNTHETIC_MODEL", "1")


def _run_app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _markdown(at) -> str:
    return " ".join(m.value for m in at.markdown)


def test_app_renders_live_prediction(app_env):
    at = _run_app()
    assert any("SYNTHETIC" in w.value for w in at.warning)  # test-mode banner is shown
    assert "IPL Win Probability Estimator" in _markdown(at)
    assert 'class="prob-card bat"' in _markdown(at)
    assert not at.error


def test_app_completed_match_shows_no_probability(app_env):
    at = _run_app()
    at.number_input(key="wickets").set_value(10).run()
    assert not at.exception
    assert any("Match complete" in i.value for i in at.info)
    assert 'class="prob-card' not in _markdown(at)


def test_app_invalid_input_shows_error(app_env):
    at = _run_app()
    at.number_input(key="overs").set_value(20).run()
    at.number_input(key="balls").set_value(3).run()
    assert not at.exception
    assert any("Invalid match situation" in e.value for e in at.error)


def test_app_missing_model_shows_helpful_message(monkeypatch, tmp_path):
    monkeypatch.setenv("IPL_MODEL_PATH", str(tmp_path / "missing.joblib"))
    at = _run_app()
    assert any("not available" in e.value for e in at.error)
