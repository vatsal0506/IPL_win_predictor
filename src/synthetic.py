"""SYNTHETIC test-data generator - for exercising the pipeline ONLY.

Writes randomly simulated matches in the Cricsheet CSV2 layout so the loader,
preprocessing, feature engineering, training and prediction code can be
tested without network access. These are NOT real IPL matches and a model
trained on them must never be used for real predictions (models trained on
it are tagged ``data_kind="synthetic"`` and refused by the predictor by
default).
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np

SYNTHETIC_TEAMS = ["Synthetic Lions", "Synthetic Tigers", "Synthetic Eagles", "Synthetic Sharks"]
SYNTHETIC_VENUES = ["Test Ground A, City A", "Test Ground B, City B", "Test Ground C"]
BALL_COLUMNS = [
    "match_id", "season", "start_date", "venue", "innings", "ball", "actual_delivery", "batting_team",
    "bowling_team", "striker", "non_striker", "bowler", "runs_off_bat", "extras", "wides", "noballs", "byes",
    "legbyes", "penalty", "non_boundary", "wicket_type", "player_dismissed", "other_wicket_type",
    "other_player_dismissed", "fielder_1", "fielder_2", "fielder_3",
]


def _simulate_innings(rng, strength: float, target: int | None):
    """Return (rows, total, wickets). Stops when the target is reached in a chase."""
    rows, total, wickets, legal = [], 0, 0, 0
    over, delivery = 0, 0
    while legal < 120 and wickets < 10 and (target is None or total < target):
        delivery += 1
        wide = rng.random() < 0.03
        noball = (not wide) and rng.random() < 0.005
        runs = 0 if wide else int(rng.choice([0, 1, 2, 3, 4, 6], p=_run_probs(strength)))
        extras = (1 if wide or noball else 0)
        wicket = (not wide and not noball) and rng.random() < 0.045 / strength
        rows.append({
            "ball": f"{over}.{delivery}", "runs_off_bat": runs, "extras": extras,
            "wides": 1 if wide else "", "noballs": 1 if noball else "",
            "wicket_type": "caught" if wicket else "", "player_dismissed": "Batter" if wicket else "",
        })
        total += runs + extras
        wickets += int(wicket)
        if not (wide or noball):
            legal += 1
            if legal % 6 == 0:
                over, delivery = over + 1, 0
    return rows, total, wickets


def _run_probs(strength: float) -> list[float]:
    p = np.array([0.36, 0.36, 0.08, 0.01, 0.12 * strength, 0.07 * strength])
    return list(p / p.sum())


def generate_synthetic_cricsheet(out_dir: Path, n_matches: int = 120, seed: int = 0,
                                 start_year: int = 2015, matches_per_season: int = 20) -> Path:
    """Write ``n_matches`` synthetic matches (plus a tie and a no-result) to ``out_dir``."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    strengths = dict(zip(SYNTHETIC_TEAMS, [1.15, 1.0, 0.95, 0.85]))
    (out_dir / "README_SYNTHETIC.txt").write_text(
        "SYNTHETIC DATA FOR PIPELINE TESTS ONLY - NOT REAL IPL MATCHES.\n", encoding="utf-8"
    )

    for i in range(n_matches):
        match_id = str(900000 + i)
        season = start_year + i // matches_per_season
        day = date(season, 4, 1) + timedelta(days=i % matches_per_season)
        team_a, team_b = rng.choice(SYNTHETIC_TEAMS, size=2, replace=False)
        venue = SYNTHETIC_VENUES[i % len(SYNTHETIC_VENUES)]
        first_rows, first_total, _ = _simulate_innings(rng, strengths[team_a], None)
        target = first_total + 1
        second_rows, second_total, second_wkts = _simulate_innings(rng, strengths[team_b], target)

        info = [["version", "2.3.0"], ["info", "balls_per_over", "6"], ["info", "team", team_a],
                ["info", "team", team_b], ["info", "season", str(season)],
                ["info", "date", day.strftime("%Y/%m/%d")], ["info", "event", "SYNTHETIC TEST DATA"],
                ["info", "match_id", match_id], ["info", "overs", "20"], ["info", "venue", venue],
                ["info", "target_overs", "2", "20"], ["info", "target_runs", "2", str(target)]]
        if i == 5:  # one abandoned match
            info.append(["info", "outcome", "no result"])
            second_rows = []
        elif second_total >= target:
            info.append(["info", "winner", team_b])
        elif second_total == first_total:
            info += [["info", "outcome", "tie"], ["info", "eliminator", team_a]]
        else:
            info.append(["info", "winner", team_a])

        with open(out_dir / f"{match_id}_info.csv", "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(info)
        with open(out_dir / f"{match_id}.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=BALL_COLUMNS, restval="")
            writer.writeheader()
            for innings, (rows, bat, bowl) in enumerate(
                [(first_rows, team_a, team_b), (second_rows, team_b, team_a)], start=1
            ):
                for r in rows:
                    writer.writerow({**r, "match_id": match_id, "season": season, "start_date": day.isoformat(),
                                     "venue": venue, "innings": innings, "batting_team": bat,
                                     "bowling_team": bowl})
    return out_dir
