"""Shared fixtures. A small SYNTHETIC dataset + quickly-trained model are built once
per test session so tests never depend on the network or on the real model."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_cricsheet  # noqa: E402
from src.feature_engineering import build_chase_states  # noqa: E402
from src.preprocessing import build_match_table, clean_deliveries, clean_matches  # noqa: E402
from src.synthetic import generate_synthetic_cricsheet  # noqa: E402
from src.utils import MODEL_PATH  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory) -> Path:
    return generate_synthetic_cricsheet(tmp_path_factory.mktemp("synthetic_cricsheet"), n_matches=120, seed=7)


@pytest.fixture(scope="session")
def synthetic_pipeline(synthetic_dir):
    raw_deliveries, raw_matches = load_cricsheet(synthetic_dir)
    deliveries = clean_deliveries(raw_deliveries)
    matches = clean_matches(raw_matches)
    clean, excluded = build_match_table(deliveries, matches)
    deliveries = deliveries[deliveries["match_id"].isin(clean["match_id"])]
    states = build_chase_states(deliveries, clean)
    return {"raw_deliveries": raw_deliveries, "raw_matches": raw_matches, "deliveries": deliveries,
            "clean": clean, "excluded": excluded, "states": states}


@pytest.fixture(scope="session")
def synthetic_model_path(synthetic_pipeline, tmp_path_factory) -> Path:
    from src.train import train_models

    out = tmp_path_factory.mktemp("model")
    path = out / "synthetic_model.joblib"
    train_models(synthetic_pipeline["states"], test_seasons=1, n_splits=2, quick=True, n_jobs=1,
                 model_path=path, artifacts_dir=out / "artifacts", data_kind="synthetic",
                 data_source="SYNTHETIC test data", make_plots=False)
    return path


@pytest.fixture(scope="session")
def synthetic_predictor(synthetic_model_path):
    from src.predict import WinProbabilityPredictor

    return WinProbabilityPredictor.load(synthetic_model_path, allow_synthetic=True)


@pytest.fixture(scope="session")
def real_predictor():
    """The real trained model, if it has been trained; tests using it are skipped otherwise."""
    from src.predict import ModelNotAvailableError, WinProbabilityPredictor

    try:
        return WinProbabilityPredictor.load(MODEL_PATH)
    except ModelNotAvailableError as exc:
        pytest.skip(f"Real model not available: {exc}")


def make_deliveries(rows: list[dict], match_id: str = "m1") -> pd.DataFrame:
    """Build canonical deliveries from compact dicts (innings, runs, extras, wide, noball, wicket)."""
    out = []
    for i, r in enumerate(rows):
        out.append({
            "match_id": match_id, "innings": r.get("innings", 2), "over": r.get("over", 0),
            "ball_label": r.get("ball_label", f"{r.get('innings', 2)}-{i}"), "seq": i,
            "batting_team": r.get("batting_team", "Mumbai Indians"),
            "bowling_team": r.get("bowling_team", "Chennai Super Kings"),
            "runs_off_bat": r.get("runs", 0), "extras": r.get("extras", 0),
            "is_wide": r.get("wide", False), "is_noball": r.get("noball", False), "wickets": r.get("wicket", 0),
        })
    return pd.DataFrame(out)
