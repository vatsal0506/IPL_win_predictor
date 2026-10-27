"""Exploratory data analysis: summary tables and plots.

Usage::

    python -m src.eda      # writes artifacts/eda/*.png and artifacts/eda/eda_report.md
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_loader import load_raw_data
from src.preprocessing import build_match_table, clean_deliveries, clean_matches
from src.utils import EDA_DIR, get_logger

logger = get_logger(__name__)
sns.set_theme(style="whitegrid", context="notebook")


def load_for_eda() -> dict[str, pd.DataFrame]:
    raw_deliveries, raw_matches = load_raw_data()
    deliveries_all = clean_deliveries(raw_deliveries)
    matches_all = clean_matches(raw_matches)
    clean, excluded = build_match_table(deliveries_all, matches_all)
    return {
        "raw_deliveries": raw_deliveries,
        "raw_matches": raw_matches,
        "deliveries": deliveries_all,
        "matches": matches_all,
        "clean": clean,
        "excluded": excluded,
    }


def overview(d: dict[str, pd.DataFrame]) -> dict:
    m, dl = d["matches"], d["deliveries"]
    teams = sorted(set(m["team1"].dropna()) | set(m["team2"].dropna()))
    return {
        "matches": int(len(d["raw_matches"])),
        "deliveries_all_innings": int(len(d["raw_deliveries"])),
        "deliveries_regulation_innings": int(len(dl)),
        "super_over_deliveries": int((d["raw_deliveries"]["innings"] > 2).sum()),
        "seasons": f"{int(m['season_year'].min())}-{int(m['season_year'].max())} ({m['season_year'].nunique()} seasons)",
        "teams_after_normalisation": len(teams),
        "venues_after_normalisation": int(m["venue"].nunique()),
        "duplicate_deliveries": int(d["raw_deliveries"].duplicated(["match_id", "innings", "ball_label"]).sum()),
        "duplicate_match_ids": int(d["raw_matches"]["match_id"].duplicated().sum()),
        "total_runs": int(dl["total_runs"].sum()),
        "total_wickets": int(dl["wickets"].sum()),
        "eligible_matches_for_training": int(len(d["clean"])),
        "excluded_matches": int(len(d["excluded"])),
        "chasing_team_win_rate_eligible": round(float(d["clean"]["chase_won"].mean()), 4),
    }


def missing_values(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    m = d["raw_matches"]
    return (m.isna().sum().rename("missing").to_frame()
            .assign(pct=lambda x: (100 * x["missing"] / len(m)).round(1))
            .query("missing > 0").sort_values("missing", ascending=False))


def team_stats(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    m = d["matches"]
    played = pd.concat([m["team1"], m["team2"]]).value_counts().rename("matches")
    wins = m["winner"].value_counts().rename("wins")
    out = pd.concat([played, wins], axis=1).fillna(0).astype(int)
    out["win_pct"] = (100 * out["wins"] / out["matches"]).round(1)
    return out.sort_values("matches", ascending=False)


def season_stats(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    dl = d["deliveries"].merge(d["matches"][["match_id", "season_year"]], on="match_id")
    inn = dl.groupby(["season_year", "match_id", "innings"]).agg(runs=("total_runs", "sum"),
                                                                   wickets=("wickets", "sum")).reset_index()
    first = inn[inn["innings"] == 1].groupby("season_year").agg(avg_first_innings=("runs", "mean"),
                                                                 avg_wickets=("wickets", "mean"))
    matches = d["matches"].groupby("season_year").size().rename("matches")
    chase = d["clean"].groupby("season_year")["chase_won"].mean().rename("chase_win_rate")
    return pd.concat([matches, first.round(1), chase.round(3)], axis=1)


def outcome_distribution(d: dict[str, pd.DataFrame]) -> pd.Series:
    m = d["matches"].copy()
    clean = d["clean"].set_index("match_id")["chase_won"]
    label = pd.Series("Excluded (other reasons)", index=m["match_id"])
    excl = d["excluded"].set_index("match_id")["reason"]
    label.loc[excl.index] = "Excluded: " + excl
    label.loc[clean.index] = np.where(clean == 1, "Chasing team won", "Defending team won")
    return label.value_counts()


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_matches_and_scores(seasons: pd.DataFrame, out: Path) -> Path:
    fig, ax1 = plt.subplots(figsize=(10, 4.2))
    ax1.bar(seasons.index.astype(str), seasons["matches"], color="#9ecae1", label="Matches")
    ax1.set_ylabel("Matches")
    ax2 = ax1.twinx()
    ax2.plot(seasons.index.astype(str), seasons["avg_first_innings"], color="#d62728", marker="o",
             label="Avg 1st-innings total")
    ax2.set_ylabel("Average first-innings total")
    ax2.grid(False)
    ax1.set_title("Matches per season and average first-innings total")
    ax1.tick_params(axis="x", rotation=45)
    fig.legend(loc="upper left", bbox_to_anchor=(0.07, 0.9))
    return _save(fig, out)


def plot_team_stats(teams: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    t = teams.sort_values("matches")
    ax.barh(t.index, t["matches"], color="#c6dbef", label="Played")
    ax.barh(t.index, t["wins"], color="#2171b5", label="Won")
    for i, (mt, pct) in enumerate(zip(t["matches"], t["win_pct"])):
        ax.text(mt + 2, i, f"{pct:.0f}%", va="center", fontsize=8)
    ax.set_xlabel("Matches")
    ax.set_title("Matches played and won by franchise (names normalised)")
    ax.legend()
    return _save(fig, out)


def plot_winners(d: dict[str, pd.DataFrame], out: Path) -> Path:
    winners = d["matches"]["winner"].fillna("No winner (NR/tie w/o eliminator)").value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=winners.values, y=winners.index, ax=ax, color="#31a354")
    ax.set_xlabel("Matches won")
    ax.set_ylabel("")
    ax.set_title("Distribution of match winners")
    return _save(fig, out)


def plot_outcomes(outcomes: pd.Series, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(x=outcomes.values, y=outcomes.index, ax=ax, color="#756bb1")
    for i, v in enumerate(outcomes.values):
        ax.text(v + 3, i, str(v), va="center", fontsize=8)
    ax.set_xlabel("Matches")
    ax.set_ylabel("")
    ax.set_title("Match outcomes (class balance for the chase label + exclusions)")
    return _save(fig, out)


def plot_chase_by_target(d: dict[str, pd.DataFrame], out: Path) -> Path:
    c = d["clean"].copy()
    c["target_band"] = pd.cut(c["target"], bins=[0, 140, 160, 180, 200, 220, 400],
                              labels=["<=140", "141-160", "161-180", "181-200", "201-220", ">220"])
    g = c.groupby("target_band", observed=True)["chase_won"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(g.index.astype(str), g["mean"], color="#fd8d3c")
    for i, (rate, n) in enumerate(zip(g["mean"], g["size"])):
        ax.text(i, rate + 0.01, f"{rate:.0%}\n(n={n})", ha="center", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Target")
    ax.set_ylabel("Chasing team win rate")
    ax.set_title("Chase success by target (eligible matches)")
    return _save(fig, out)


def plot_runs_wickets_by_over(d: dict[str, pd.DataFrame], out: Path) -> Path:
    dl = d["deliveries"]
    per_over = dl.groupby(["innings", "over"]).agg(runs=("total_runs", "sum"), wickets=("wickets", "sum"))
    n_innings = dl.groupby("innings")["match_id"].nunique()
    per_over = per_over.div(n_innings, level="innings").reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for inn, g in per_over.groupby("innings"):
        axes[0].plot(g["over"] + 1, g["runs"], marker=".", label=f"Innings {inn}")
        axes[1].plot(g["over"] + 1, g["wickets"], marker=".", label=f"Innings {inn}")
    axes[0].set(title="Average runs per over", xlabel="Over", ylabel="Runs")
    axes[1].set(title="Average wickets per over", xlabel="Over", ylabel="Wickets")
    axes[0].legend()
    axes[1].legend()
    return _save(fig, out)


def run_eda(out_dir: Path = EDA_DIR) -> dict:
    d = load_for_eda()
    info = overview(d)
    seasons = season_stats(d)
    teams = team_stats(d)
    outcomes = outcome_distribution(d)
    missing = missing_values(d)

    plot_matches_and_scores(seasons, out_dir / "season_matches_scores.png")
    plot_team_stats(teams, out_dir / "team_matches_wins.png")
    plot_winners(d, out_dir / "match_winners.png")
    plot_outcomes(outcomes, out_dir / "match_outcomes.png")
    plot_chase_by_target(d, out_dir / "chase_success_by_target.png")
    plot_runs_wickets_by_over(d, out_dir / "runs_wickets_by_over.png")

    report = ["# EDA report (auto-generated by `python -m src.eda`)\n", "## Overview\n",
              "| Item | Value |", "|---|---|", *[f"| {k} | {v} |" for k, v in info.items()],
              "\n## Missing values in match metadata\n", missing.to_markdown() if len(missing) else "None",
              "\n## Outcomes\n", outcomes.to_frame("matches").to_markdown(),
              "\n## Season statistics\n", seasons.to_markdown(),
              "\n## Team statistics (all matches)\n", teams.to_markdown()]
    (out_dir / "eda_report.md").write_text("\n".join(report), encoding="utf-8")
    logger.info("EDA written to %s", out_dir)
    return info


if __name__ == "__main__":
    for k, v in run_eda().items():
        print(f"{k:35s} {v}")
