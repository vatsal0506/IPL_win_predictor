"""Dataset acquisition and schema-aware loading.

Supported inputs
----------------
1. **Cricsheet "CSV2" IPL archive** (preferred, freely licensed under ODC-By):
   https://cricsheet.org/downloads/ipl_male_csv2.zip
   One ``<match_id>.csv`` (ball-by-ball) + ``<match_id>_info.csv`` (metadata)
   per match, plus an ``all_matches.csv`` with every delivery.
2. **Kaggle-style ``matches.csv`` + ``deliveries.csv``** (e.g. the popular
   "IPL Complete Dataset" uploads). Column names differ between uploads, so
   columns are resolved through alias lists instead of being hard-coded.

Both are converted to one *canonical* schema:

deliveries: match_id, innings, over, ball_label, seq, batting_team, bowling_team,
            runs_off_bat, extras, is_wide, is_noball, wickets
matches:    match_id, season, date, venue, city, team1, team2, winner,
            outcome, method, eliminator, target_runs, target_overs,
            scheduled_overs

Usage::

    python -m src.data_loader --download     # fetch Cricsheet archive
    python -m src.data_loader                # validate + summarise local data
"""

from __future__ import annotations

import argparse
import csv
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import CRICSHEET_DIR, RAW_DATA_DIR, get_logger

logger = get_logger(__name__)

CRICSHEET_URL = "https://cricsheet.org/downloads/ipl_male_csv2.zip"

# Dismissals that do not count towards a side being bowled out: the batter can
# return (retired hurt) or was never dismissed (retired not out).
NON_DISMISSALS = {"retired hurt", "retired not out"}

DELIVERY_COLUMNS = [
    "match_id", "innings", "over", "ball_label", "seq", "batting_team", "bowling_team",
    "runs_off_bat", "extras", "is_wide", "is_noball", "wickets",
]
MATCH_COLUMNS = [
    "match_id", "season", "date", "venue", "city", "team1", "team2", "winner",
    "outcome", "method", "eliminator", "target_runs", "target_overs", "scheduled_overs",
]


class DataValidationError(RuntimeError):
    """Raised when the dataset is missing or has an incompatible schema."""


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download_cricsheet(dest_dir: Path = CRICSHEET_DIR, url: str = CRICSHEET_URL) -> Path:
    """Download and extract the Cricsheet IPL CSV2 archive into ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s ...", url)
    request = urllib.request.Request(url, headers={"User-Agent": "ipl-win-probability-estimator"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - fixed https URL
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        bad = [n for n in archive.namelist() if n.startswith(("/", "\\")) or ".." in n]
        if bad:
            raise DataValidationError(f"Refusing to extract archive with unsafe paths: {bad[:3]}")
        archive.extractall(dest_dir)
    n_info = len(list(dest_dir.glob("*_info.csv")))
    logger.info("Extracted %d match files into %s", n_info, dest_dir)
    if n_info == 0:
        raise DataValidationError("Download finished but no *_info.csv match files were found.")
    return dest_dir


# --------------------------------------------------------------------------- #
# Cricsheet CSV2
# --------------------------------------------------------------------------- #
def _parse_cricsheet_info(path: Path) -> dict:
    """Parse one Cricsheet ``*_info.csv`` file (rows of ``info,key,value[,value2]``)."""
    record: dict = {"teams": []}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 3 or row[0] != "info":
                continue
            key, values = row[1], row[2:]
            if key == "team":
                record["teams"].append(values[0])
            elif key in ("target_runs", "target_overs") and len(values) >= 2:
                record[key] = values[1]  # values = [innings, value]
            elif key not in record:  # keep the first occurrence (e.g. first date)
                record[key] = values[0]
    record["match_id"] = record.get("match_id") or path.name.replace("_info.csv", "")
    return record


def load_cricsheet(directory: Path = CRICSHEET_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    info_files = sorted(directory.glob("*_info.csv"))
    if not info_files:
        raise DataValidationError(f"No Cricsheet *_info.csv files found in {directory}.")

    infos = [_parse_cricsheet_info(p) for p in info_files]
    matches = pd.DataFrame(
        {
            "match_id": [str(r["match_id"]) for r in infos],
            "season": [r.get("season") for r in infos],
            "date": [r.get("date") for r in infos],
            "venue": [r.get("venue") for r in infos],
            "city": [r.get("city") for r in infos],
            "team1": [r["teams"][0] if len(r["teams"]) > 0 else None for r in infos],
            "team2": [r["teams"][1] if len(r["teams"]) > 1 else None for r in infos],
            "winner": [r.get("winner") for r in infos],
            "outcome": [r.get("outcome", "result") for r in infos],
            "method": [r.get("method") for r in infos],
            "eliminator": [r.get("eliminator") for r in infos],
            "target_runs": [r.get("target_runs") for r in infos],
            "target_overs": [r.get("target_overs") for r in infos],
            "scheduled_overs": [r.get("overs") for r in infos],
        }
    )

    all_matches = directory / "all_matches.csv"
    if all_matches.exists():
        raw = pd.read_csv(all_matches, dtype={"match_id": str, "ball": str}, low_memory=False)
    else:
        ball_files = [p for p in directory.glob("*.csv") if not p.name.endswith("_info.csv")]
        if not ball_files:
            raise DataValidationError(f"No ball-by-ball CSV files found in {directory}.")
        raw = pd.concat(
            (pd.read_csv(p, dtype={"match_id": str, "ball": str}, low_memory=False) for p in ball_files),
            ignore_index=True,
        )

    required = {"match_id", "innings", "ball", "batting_team", "bowling_team", "runs_off_bat", "extras"}
    _require_columns(raw, required, "Cricsheet deliveries")

    def _count_dismissals(kind_col: str) -> pd.Series:
        if kind_col not in raw:
            return pd.Series(0, index=raw.index)
        kind = raw[kind_col].fillna("").astype(str).str.strip().str.lower()
        return ((kind != "") & ~kind.isin(NON_DISMISSALS)).astype(int)

    deliveries = pd.DataFrame(
        {
            "match_id": raw["match_id"].astype(str),
            "innings": pd.to_numeric(raw["innings"], errors="coerce"),
            # Cricsheet "ball" is "<over>.<delivery>", over is 0-based.
            "over": pd.to_numeric(raw["ball"].astype(str).str.split(".").str[0], errors="coerce"),
            "ball_label": raw["ball"].astype(str),
            "batting_team": raw["batting_team"],
            "bowling_team": raw["bowling_team"],
            "runs_off_bat": pd.to_numeric(raw["runs_off_bat"], errors="coerce"),
            "extras": pd.to_numeric(raw["extras"], errors="coerce"),
            "is_wide": pd.to_numeric(raw.get("wides"), errors="coerce").fillna(0).gt(0)
            if "wides" in raw else False,
            "is_noball": pd.to_numeric(raw.get("noballs"), errors="coerce").fillna(0).gt(0)
            if "noballs" in raw else False,
            "wickets": _count_dismissals("wicket_type") + _count_dismissals("other_wicket_type"),
        }
    )
    # File order is the true delivery order ("0.10" vs "0.1" would sort wrongly as floats).
    deliveries["seq"] = deliveries.groupby(["match_id", "innings"]).cumcount()
    return deliveries[DELIVERY_COLUMNS], matches[MATCH_COLUMNS]


# --------------------------------------------------------------------------- #
# Kaggle-style matches.csv + deliveries.csv
# --------------------------------------------------------------------------- #
def _pick(df: pd.DataFrame, candidates: list[str], what: str, required: bool = True) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    if required:
        raise DataValidationError(
            f"Could not find a column for '{what}'. Tried {candidates}; available: {list(df.columns)}"
        )
    return None


def load_kaggle(matches_path: Path, deliveries_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    m_raw = pd.read_csv(matches_path, low_memory=False)
    d_raw = pd.read_csv(deliveries_path, low_memory=False)

    mid = _pick(m_raw, ["id", "match_id"], "match id")
    m_cols = {
        "season": _pick(m_raw, ["season"], "season", required=False),
        "date": _pick(m_raw, ["date", "start_date"], "date"),
        "venue": _pick(m_raw, ["venue"], "venue"),
        "city": _pick(m_raw, ["city"], "city", required=False),
        "team1": _pick(m_raw, ["team1"], "team1"),
        "team2": _pick(m_raw, ["team2"], "team2"),
        "winner": _pick(m_raw, ["winner"], "winner"),
        "result": _pick(m_raw, ["result"], "result", required=False),
        "method": _pick(m_raw, ["method"], "method", required=False),
        "eliminator": _pick(m_raw, ["eliminator", "super_over"], "super over", required=False),
        "target_runs": _pick(m_raw, ["target_runs"], "target runs", required=False),
        "target_overs": _pick(m_raw, ["target_overs"], "target overs", required=False),
    }

    def col(name: str):
        return m_raw[m_cols[name]] if m_cols[name] else pd.Series([None] * len(m_raw))

    result = col("result").astype(str).str.lower()
    outcome = np.where(result.str.contains("tie"), "tie",
                       np.where(result.str.contains("no result"), "no result", "result"))
    eliminator = col("eliminator").astype(str).str.upper()
    dates = pd.to_datetime(col("date"), errors="coerce", dayfirst=False)
    season = col("season") if m_cols["season"] else dates.dt.year

    matches = pd.DataFrame(
        {
            "match_id": m_raw[mid].astype(str),
            "season": season.astype(str),
            "date": col("date"),
            "venue": col("venue"),
            "city": col("city"),
            "team1": col("team1"),
            "team2": col("team2"),
            "winner": col("winner"),
            "outcome": outcome,
            "method": col("method").where(col("method").notna(), None),
            # Kaggle "super_over" is a Y/N flag; mark ties so they are excluded later.
            "eliminator": np.where(eliminator.isin(["Y", "YES", "TRUE"]), "super over", None),
            "target_runs": col("target_runs"),
            "target_overs": col("target_overs"),
            "scheduled_overs": 20,
        }
    )
    matches.loc[matches["eliminator"].notna(), "outcome"] = "tie"

    d_mid = _pick(d_raw, ["match_id", "id"], "match id")
    inn = _pick(d_raw, ["inning", "innings"], "innings")
    over = _pick(d_raw, ["over", "overs"], "over")
    ball = _pick(d_raw, ["ball"], "ball")
    bat_runs = _pick(d_raw, ["batsman_runs", "batter_runs", "runs_off_bat"], "batter runs")
    extra_runs = _pick(d_raw, ["extra_runs", "extras"], "extra runs")
    extras_type = _pick(d_raw, ["extras_type", "extra_type"], "extras type")
    dismissal = _pick(d_raw, ["dismissal_kind", "wicket_type"], "dismissal kind", required=False)
    is_wicket = _pick(d_raw, ["is_wicket"], "is_wicket", required=False)
    bat_team = _pick(d_raw, ["batting_team"], "batting team")
    bowl_team = _pick(d_raw, ["bowling_team"], "bowling team")

    etype = d_raw[extras_type].fillna("").astype(str).str.lower()
    if dismissal:
        kind = d_raw[dismissal].fillna("").astype(str).str.strip().str.lower()
        wickets = ((kind != "") & (kind != "nan") & ~kind.isin(NON_DISMISSALS)).astype(int)
    elif is_wicket:
        wickets = pd.to_numeric(d_raw[is_wicket], errors="coerce").fillna(0).astype(int)
    else:
        raise DataValidationError("deliveries.csv needs either a dismissal_kind or is_wicket column.")

    over_num = pd.to_numeric(d_raw[over], errors="coerce")
    # Some uploads number overs 1-20 rather than 0-19.
    if over_num.min() == 1 and over_num.max() == 20:
        over_num = over_num - 1

    deliveries = pd.DataFrame(
        {
            "match_id": d_raw[d_mid].astype(str),
            "innings": pd.to_numeric(d_raw[inn], errors="coerce"),
            "over": over_num,
            "ball": pd.to_numeric(d_raw[ball], errors="coerce"),
            "batting_team": d_raw[bat_team],
            "bowling_team": d_raw[bowl_team],
            "runs_off_bat": pd.to_numeric(d_raw[bat_runs], errors="coerce"),
            "extras": pd.to_numeric(d_raw[extra_runs], errors="coerce"),
            "is_wide": etype.str.startswith("wide"),
            "is_noball": etype.str.startswith("noball") | etype.str.startswith("no ball"),
            "wickets": wickets,
        }
    )
    deliveries = deliveries.sort_values(["match_id", "innings", "over", "ball"], kind="stable")
    deliveries["ball_label"] = (
        deliveries["over"].astype("Int64").astype(str) + "." + deliveries["ball"].astype("Int64").astype(str)
    )
    deliveries["seq"] = deliveries.groupby(["match_id", "innings"]).cumcount()
    return deliveries[DELIVERY_COLUMNS].reset_index(drop=True), matches[MATCH_COLUMNS]


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"{label} is missing columns {sorted(missing)}; found {list(df.columns)}")


def detect_source(raw_dir: Path = RAW_DATA_DIR) -> tuple[str, dict[str, Path]]:
    """Find a supported dataset under ``raw_dir``."""
    for candidate in [raw_dir / CRICSHEET_DIR.name, raw_dir, *[p for p in raw_dir.glob("*") if p.is_dir()]]:
        if candidate.exists() and any(candidate.glob("*_info.csv")):
            return "cricsheet", {"dir": candidate}
    for candidate in [raw_dir, *[p for p in raw_dir.glob("*") if p.is_dir()]]:
        if (candidate / "matches.csv").exists() and (candidate / "deliveries.csv").exists():
            return "kaggle", {"matches": candidate / "matches.csv", "deliveries": candidate / "deliveries.csv"}
    raise DataValidationError(
        f"No IPL dataset found under {raw_dir}.\n"
        "  Option 1 (recommended): python -m src.data_loader --download\n"
        f"  Option 2: download {CRICSHEET_URL} and unzip it into {CRICSHEET_DIR}\n"
        "  Option 3: place a Kaggle-style matches.csv and deliveries.csv in data/raw/"
    )


def load_raw_data(raw_dir: Path = RAW_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load deliveries and matches in the canonical schema from whichever source is present."""
    source, paths = detect_source(raw_dir)
    logger.info("Detected %s dataset at %s", source, next(iter(paths.values())))
    if source == "cricsheet":
        deliveries, matches = load_cricsheet(paths["dir"])
    else:
        deliveries, matches = load_kaggle(paths["matches"], paths["deliveries"])

    if deliveries.empty or matches.empty:
        raise DataValidationError("Dataset loaded but contains no deliveries or no matches.")
    unmatched = set(deliveries["match_id"]) - set(matches["match_id"])
    if unmatched:
        logger.warning("%d match ids in deliveries have no metadata; they will be dropped.", len(unmatched))
    return deliveries, matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Download / validate the IPL dataset.")
    parser.add_argument("--download", action="store_true", help="Download the Cricsheet IPL archive.")
    args = parser.parse_args()
    if args.download:
        download_cricsheet()
    deliveries, matches = load_raw_data()
    print(f"Matches: {len(matches):,} | Deliveries: {len(deliveries):,}")
    print(f"Seasons: {sorted(matches['season'].astype(str).unique())}")
    print(matches["outcome"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
