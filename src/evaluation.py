"""Metrics and plots for probabilistic classifiers.

For a win-probability model the headline metrics are **log loss**, **Brier
score** and **calibration** (do events predicted at 70% happen ~70% of the
time?). Accuracy / precision / recall / F1 at a 0.5 threshold are reported too,
but a well-calibrated 55% prediction that "loses" is not a model error.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: works in scripts, tests and CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PHASES = [(0, 36, "Powerplay (0-6 ov)"), (36, 90, "Middle (6-15 ov)"), (90, 120, "Death (15-20 ov)")]


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Weighted mean |observed win rate - mean predicted probability| over equal-width bins."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    bins = np.clip((y_prob * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)
    two_classes = len(np.unique(y_true)) == 2
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if two_classes else float("nan"),
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "brier": brier_score_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "n_states": int(len(y_true)),
    }


def metrics_by_phase(states: pd.DataFrame, y_prob: np.ndarray) -> pd.DataFrame:
    rows = []
    for lo, hi, label in PHASES:
        mask = (states["balls_bowled"] >= lo) & (states["balls_bowled"] < hi)
        if mask.any():
            m = classification_metrics(states.loc[mask, "chase_won"], np.asarray(y_prob)[mask.to_numpy()])
            rows.append({"phase": label, **{k: m[k] for k in ["n_states", "accuracy", "roc_auc", "log_loss", "brier"]}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_calibration(y_true, probs: dict[str, np.ndarray], path: Path, n_bins: int = 10) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfectly calibrated")
    for name, p in probs.items():
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="uniform")
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.6, label=name)
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Observed win rate")
    ax.set_title("Calibration on held-out test seasons")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_roc(y_true, probs: dict[str, np.ndarray], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, p in probs.items():
        if len(np.unique(p)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, linewidth=1.6, label=f"{name} (AUC {roc_auc_score(y_true, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on held-out test seasons")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_confusion_matrix(y_true, y_prob, path: Path, title: str) -> Path:
    cm = confusion_matrix(np.asarray(y_true).astype(int), (np.asarray(y_prob) >= 0.5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["Chase lost", "Chase won"])
    ax.set_yticks([0, 1], ["Chase lost", "Chase won"])
    ax.set_xlabel("Predicted (p >= 0.5)")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    return _save(fig, path)


def plot_metric_comparison(table: pd.DataFrame, path: Path) -> Path:
    metrics = ["log_loss", "brier", "ece", "roc_auc", "accuracy"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.8))
    for ax, metric in zip(axes, metrics):
        values = table.set_index("model")[f"test_{metric}"]
        ax.barh(values.index, values.to_numpy(), color="#1f6feb")
        ax.set_title(metric + (" (lower better)" if metric in {"log_loss", "brier", "ece"} else " (higher better)"),
                     fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        for i, v in enumerate(values.to_numpy()):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=7)
    return _save(fig, path)


def plot_match_worm(states: pd.DataFrame, y_prob: np.ndarray, match_ids: list[str], path: Path) -> Path:
    """Win-probability timeline for a few test matches (batting-side probability per legal ball)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    df = states.assign(prob=y_prob)
    for mid in match_ids:
        g = df[df["match_id"] == mid]
        if g.empty:
            continue
        r = g.iloc[0]
        outcome = "won" if r["chase_won"] == 1 else "lost"
        ax.plot(g["balls_bowled"] / 6, g["prob"], linewidth=1.5,
                label=f"{r['batting_team']} chasing {r['target']} vs {r['bowling_team']} ({outcome})")
    ax.axhline(0.5, color="grey", linestyle=":")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Overs completed in chase")
    ax.set_ylabel("Chasing team win probability")
    ax.set_title("Example test-season matches: predicted win probability through the chase")
    ax.legend(fontsize=7, loc="best")
    return _save(fig, path)


def plot_feature_importance(importances: pd.Series, path: Path, title: str) -> Path:
    importances = importances.sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(importances.index, importances.to_numpy(), color="#2da44e")
    ax.set_xlabel("Increase in log loss when feature is shuffled")
    ax.set_title(title, fontsize=10)
    return _save(fig, path)
