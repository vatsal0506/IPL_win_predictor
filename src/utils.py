"""Shared paths, constants and small helpers used across the project.

All paths are resolved relative to the project root so the code works the same
on any machine (and on Streamlit Community Cloud) without absolute paths.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CRICSHEET_DIR = RAW_DATA_DIR / "cricsheet_ipl_csv2"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
EDA_DIR = ARTIFACTS_DIR / "eda"

MODEL_PATH = MODELS_DIR / "win_probability_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
CLEAN_MATCHES_PATH = PROCESSED_DATA_DIR / "matches_clean.csv"
CLEAN_DELIVERIES_PATH = PROCESSED_DATA_DIR / "deliveries_clean.csv.gz"
CHASE_STATES_PATH = PROCESSED_DATA_DIR / "chase_states.csv.gz"
EXCLUSIONS_PATH = PROCESSED_DATA_DIR / "excluded_matches.csv"

RANDOM_SEED = 42

# A standard T20 innings: 20 overs of 6 legal deliveries each.
OVERS_PER_INNINGS = 20
BALLS_PER_OVER = 6
BALLS_PER_INNINGS = OVERS_PER_INNINGS * BALLS_PER_OVER
MAX_WICKETS = 10

# The Impact Player substitution rule (from IPL 2023) lengthened batting line-ups
# and pushed scores up sharply - a structural change known before any match.
IMPACT_PLAYER_FIRST_SEASON = 2023

# Franchise renames: the same franchise under an older name is mapped to its
# current name so its history is not split across two categories.
# Defunct franchises (Deccan Chargers, Kochi Tuskers Kerala, Pune Warriors,
# Gujarat Lions, Rising Pune Supergiant) are distinct teams and kept as-is.
TEAM_NAME_MAP: dict[str, str] = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

# The ten franchises active in recent seasons (shown first in the app).
ACTIVE_TEAMS: list[str] = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
]

# Venue aliases applied *after* stripping the ", City" suffix.
VENUE_NAME_MAP: dict[str, str] = {
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",  # renamed in 2019
    "Punjab Cricket Association Stadium": "Punjab Cricket Association IS Bindra Stadium",
    "Sardar Patel Stadium": "Narendra Modi Stadium",  # same Motera ground, rebuilt/renamed
    "Sheikh Zayed Stadium": "Zayed Cricket Stadium",
}

UNKNOWN_VENUE = "Other / unknown venue"


def normalize_team(name: Any) -> Any:
    """Map historical franchise names to their current name."""
    if not isinstance(name, str):
        return name
    name = " ".join(name.split())
    return TEAM_NAME_MAP.get(name, name)


def normalize_venue(name: Any) -> Any:
    """Strip city suffixes ("Eden Gardens, Kolkata" -> "Eden Gardens") and apply aliases."""
    if not isinstance(name, str):
        return name
    base = " ".join(name.split(",")[0].split())
    return VENUE_NAME_MAP.get(base, base)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    return logging.getLogger(name)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def format_overs(legal_balls: int) -> str:
    """Convert legal balls to cricket overs notation, e.g. 75 -> '12.3'."""
    return f"{legal_balls // BALLS_PER_OVER}.{legal_balls % BALLS_PER_OVER}"
