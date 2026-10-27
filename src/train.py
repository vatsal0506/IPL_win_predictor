"""Train, tune, compare and save win-probability models.

Methodology
-----------
1. **Chronological test split by match** - the most recent ``--test-seasons``
   IPL seasons are held out entirely. All states of a match stay together.
2. **GridSearchCV with expanding-window, season-based folds** - fold *k*
   trains on every season before validation season *k*. Matches never span
   folds and the model is never validated on the past using the future.
3. **Model selection on CV log loss only** - the test seasons are touched once,
   after every decision has been made, to report final metrics.
4. **Calibration check** - out-of-fold (OOF) CV predictions of the selected
   model are used to test Platt (sigmoid) and isotonic calibration, each fitted
   on earlier folds and scored on the next fold. Calibration is applied only
   if it measurably improves OOF log loss.

Usage::

    python -m src.train            # full grid
    python -m src.train --quick    # small grid (fast smoke run)
"""

from __future__ import annotations

import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import evaluation as ev
from src.calibration import CalibratedModel, apply_calibrator, fit_calibrator
from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_chase_states,
)
from src.utils import (
    ARTIFACTS_DIR,
    CHASE_STATES_PATH,
    CLEAN_DELIVERIES_PATH,
    CLEAN_MATCHES_PATH,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    get_logger,
    save_json,
)

logger = get_logger(__name__)

MIN_MATCHES_FOR_RELIABLE_MODEL = 300
# Calibration is only adopted if it lowers OOF log loss by at least this much.
CALIBRATION_MIN_GAIN = 0.002


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_chase_states(rebuild: bool = False) -> pd.DataFrame:
    if CHASE_STATES_PATH.exists() and not rebuild:
        states = pd.read_csv(CHASE_STATES_PATH, dtype={"match_id": str}, parse_dates=["date"])
        return states
    if not (CLEAN_MATCHES_PATH.exists() and CLEAN_DELIVERIES_PATH.exists()):
        from src.preprocessing import run_preprocessing

        run_preprocessing(save=True)
    matches = pd.read_csv(CLEAN_MATCHES_PATH, dtype={"match_id": str}, parse_dates=["date"])
    deliveries = pd.read_csv(CLEAN_DELIVERIES_PATH, dtype={"match_id": str})
    states = build_chase_states(deliveries, matches)
    states.to_csv(CHASE_STATES_PATH, index=False)
    logger.info("Built %d chase states from %d matches -> %s", len(states), states["match_id"].nunique(),
                CHASE_STATES_PATH)
    return states


def time_blocks(states: pd.DataFrame, n_blocks_min: int) -> pd.Series:
    """Chronological block id per row: the season if there are enough seasons, else match-order bins."""
    if states["season_year"].nunique() >= n_blocks_min:
        return states["season_year"].astype(int)
    order = states.groupby("match_id")["date"].first().sort_values()
    rank = pd.Series(np.arange(len(order)), index=order.index)
    bins = (rank * n_blocks_min // max(len(order), 1)).astype(int)
    return states["match_id"].map(bins)


def chronological_split(states: pd.DataFrame, test_seasons: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks = time_blocks(states, n_blocks_min=test_seasons + 3)
    test_blocks = sorted(blocks.unique())[-test_seasons:]
    is_test = blocks.isin(test_blocks)
    train, test = states[~is_test].reset_index(drop=True), states[is_test].reset_index(drop=True)
    assert not set(train["match_id"]) & set(test["match_id"]), "match leakage between train and test"
    return train, test


def temporal_cv_splits(train: pd.DataFrame, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window splits: validate on block k, train on all earlier blocks."""
    blocks = time_blocks(train, n_blocks_min=n_splits + 2).to_numpy()
    uniq = sorted(np.unique(blocks))
    splits = []
    for val_block in uniq[-n_splits:]:
        tr = np.flatnonzero(blocks < val_block)
        va = np.flatnonzero(blocks == val_block)
        if len(tr) and len(va):
            splits.append((tr, va))
    return splits


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_pipeline(estimator, scale_numeric: bool) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                # Rare teams/venues (<0.5% of states) share an "infrequent" column;
                # unseen ones at inference time map to that same column.
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005, sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler() if scale_numeric else "passthrough", NUMERIC_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("clf", estimator)])


def model_specs(quick: bool) -> dict[str, tuple[Pipeline, dict]]:
    if quick:
        return {
            "Logistic Regression": (
                build_pipeline(LogisticRegression(max_iter=3000), scale_numeric=True),
                {"clf__C": [0.1, 1.0]},
            ),
            "Random Forest": (
                build_pipeline(RandomForestClassifier(n_estimators=40, random_state=RANDOM_SEED, n_jobs=1), False),
                {"clf__max_depth": [8], "clf__min_samples_leaf": [20, 60]},
            ),
            "Gradient Boosting": (
                build_pipeline(HistGradientBoostingClassifier(max_iter=100, random_state=RANDOM_SEED), False),
                {"clf__learning_rate": [0.1], "clf__max_leaf_nodes": [15, 31]},
            ),
        }
    return {
        "Logistic Regression": (
            build_pipeline(LogisticRegression(max_iter=3000), scale_numeric=True),
            {"clf__C": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]},
        ),
        "Random Forest": (
            build_pipeline(RandomForestClassifier(n_estimators=150, random_state=RANDOM_SEED, n_jobs=1), False),
            {"clf__max_depth": [12, 16, None], "clf__min_samples_leaf": [100, 250, 500]},
        ),
        "Gradient Boosting": (
            build_pipeline(
                HistGradientBoostingClassifier(max_iter=300, early_stopping=False, random_state=RANDOM_SEED), False
            ),
            {
                "clf__learning_rate": [0.03, 0.08],
                "clf__max_leaf_nodes": [7, 15],
                "clf__min_samples_leaf": [200, 1000],
                "clf__l2_regularization": [1.0, 10.0],
            },
        ),
    }


SCORING = {"log_loss": "neg_log_loss", "brier": "neg_brier_score", "roc_auc": "roc_auc"}


def oof_predictions(estimator, X, y, splits) -> list[tuple[np.ndarray, np.ndarray]]:
    """Out-of-fold (y_true, p) per CV fold for a fixed hyper-parameter setting."""
    out = []
    for tr, va in splits:
        model = clone(estimator).fit(X.iloc[tr], y.iloc[tr])
        out.append((y.iloc[va].to_numpy(), model.predict_proba(X.iloc[va])[:, 1]))
    return out


def choose_calibration(oof: list[tuple[np.ndarray, np.ndarray]]) -> tuple[str, dict[str, float]]:
    """Score none/sigmoid/isotonic: calibrator fitted on folds < k, evaluated on fold k."""
    scores: dict[str, list[float]] = {"none": [], "sigmoid": [], "isotonic": []}
    weights = []
    for k in range(1, len(oof)):
        y_k, p_k = oof[k]
        y_prev = np.concatenate([o[0] for o in oof[:k]])
        p_prev = np.concatenate([o[1] for o in oof[:k]])
        weights.append(len(y_k))
        scores["none"].append(log_loss(y_k, p_k, labels=[0, 1]))
        for method in ("sigmoid", "isotonic"):
            cal = fit_calibrator(method, p_prev, y_prev)
            scores[method].append(log_loss(y_k, apply_calibrator(cal, p_k), labels=[0, 1]))
    if not weights:
        return "none", {}
    mean = {m: float(np.average(v, weights=weights)) for m, v in scores.items()}
    best = min(("sigmoid", "isotonic"), key=mean.get)
    chosen = best if mean["none"] - mean[best] >= CALIBRATION_MIN_GAIN else "none"
    return chosen, mean


def permutation_importance_logloss(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 3,
                                   max_rows: int = 15000) -> pd.Series:
    rng = np.random.default_rng(RANDOM_SEED)
    if len(X) > max_rows:
        idx = rng.choice(len(X), size=max_rows, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    base = log_loss(y, model.predict_proba(X)[:, 1], labels=[0, 1])
    result = {}
    for col in X.columns:
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            deltas.append(log_loss(y, model.predict_proba(Xp)[:, 1], labels=[0, 1]) - base)
        result[col] = float(np.mean(deltas))
    return pd.Series(result)


# --------------------------------------------------------------------------- #
# Main training routine
# --------------------------------------------------------------------------- #
def train_models(
    states: pd.DataFrame,
    *,
    test_seasons: int = 2,
    n_splits: int = 4,
    quick: bool = False,
    n_jobs: int = -1,
    model_path: Path = MODEL_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
    data_kind: str = "real",
    data_source: str = "Cricsheet IPL ball-by-ball data (https://cricsheet.org, ODC-By 1.0)",
    make_plots: bool = True,
) -> dict:
    figures_dir = artifacts_dir / FIGURES_DIR.name
    train, test = chronological_split(states, test_seasons)
    n_train_matches, n_test_matches = train["match_id"].nunique(), test["match_id"].nunique()
    logger.info("Train: %d states / %d matches (seasons %s-%s) | Test: %d states / %d matches (seasons %s)",
                len(train), n_train_matches, train["season_year"].min(), train["season_year"].max(),
                len(test), n_test_matches, sorted(test["season_year"].unique()))
    if n_train_matches < MIN_MATCHES_FOR_RELIABLE_MODEL:
        logger.warning("Only %d training matches - results will be noisy and should not be over-interpreted.",
                       n_train_matches)

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN].astype(int)
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN].astype(int)
    splits = temporal_cv_splits(train, n_splits)
    if len(splits) < 2:
        raise ValueError("Not enough chronological blocks in the training data for temporal cross-validation.")
    logger.info("Temporal CV: %d expanding-window folds", len(splits))

    rows, fitted, test_probs = [], {}, {}

    # Baseline: always predict the training base rate - any useful model must beat this.
    dummy = DummyClassifier(strategy="prior")
    cv_dummy = [log_loss(y_train.iloc[va], clone(dummy).fit(X_train.iloc[tr], y_train.iloc[tr])
                         .predict_proba(X_train.iloc[va])[:, 1], labels=[0, 1]) for tr, va in splits]
    dummy.fit(X_train, y_train)
    test_probs["Baseline (base rate)"] = dummy.predict_proba(X_test)[:, 1]
    rows.append({"model": "Baseline (base rate)", "best_params": {}, "cv_log_loss": float(np.mean(cv_dummy)),
                 "cv_brier": np.nan, "cv_roc_auc": np.nan})

    for name, (pipeline, grid) in model_specs(quick).items():
        logger.info("GridSearchCV: %s over %s", name, grid)
        search = GridSearchCV(pipeline, grid, scoring=SCORING, refit="log_loss", cv=splits, n_jobs=n_jobs,
                              error_score="raise")
        search.fit(X_train, y_train)
        best = search.best_index_
        cv = search.cv_results_
        rows.append({
            "model": name,
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_log_loss": float(-cv["mean_test_log_loss"][best]),
            "cv_brier": float(-cv["mean_test_brier"][best]),
            "cv_roc_auc": float(cv["mean_test_roc_auc"][best]),
        })
        fitted[name] = search.best_estimator_
        test_probs[name] = search.best_estimator_.predict_proba(X_test)[:, 1]
        logger.info("  best %s -> CV log loss %.4f", search.best_params_, rows[-1]["cv_log_loss"])

    # ---- Select on CV log loss (test set not consulted) --------------------
    ml_rows = [r for r in rows if r["model"] in fitted]
    selected = min(ml_rows, key=lambda r: r["cv_log_loss"])["model"]
    logger.info("Selected model by CV log loss: %s", selected)

    # ---- Calibration check on out-of-fold predictions ----------------------
    oof = oof_predictions(fitted[selected], X_train, y_train, splits)
    calibration, calibration_scores = choose_calibration(oof)
    logger.info("Calibration OOF log loss: %s -> using '%s'", calibration_scores, calibration)
    final_model = fitted[selected]
    final_name = selected
    if calibration != "none":
        y_oof = np.concatenate([o[0] for o in oof])
        p_oof = np.concatenate([o[1] for o in oof])
        final_model = CalibratedModel(fitted[selected], fit_calibrator(calibration, p_oof, y_oof), calibration)
        final_name = f"{selected} + {calibration} calibration"
        test_probs[final_name] = final_model.predict_proba(X_test)[:, 1]
        rows.append({**next(r for r in rows if r["model"] == selected), "model": final_name})

    # ---- Final, one-time evaluation on the held-out test seasons -----------
    for r in rows:
        m = ev.classification_metrics(y_test, test_probs[r["model"]])
        r.update({f"test_{k}": v for k, v in m.items() if k != "n_states"})
    table = pd.DataFrame(rows)
    table["selected"] = table["model"] == final_name
    final_metrics = ev.classification_metrics(y_test, test_probs[final_name])
    phase_table = ev.metrics_by_phase(test, test_probs[final_name])

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(artifacts_dir / "model_comparison.csv", index=False)
    phase_table.to_csv(artifacts_dir / "metrics_by_phase.csv", index=False)

    importance = None
    if make_plots:
        ev.plot_calibration(y_test, {k: v for k, v in test_probs.items() if not k.startswith("Baseline")},
                            figures_dir / "calibration_curves.png")
        ev.plot_roc(y_test, {k: v for k, v in test_probs.items() if not k.startswith("Baseline")},
                    figures_dir / "roc_curves.png")
        ev.plot_confusion_matrix(y_test, test_probs[final_name], figures_dir / "confusion_matrix.png",
                                 f"{final_name}\n(test seasons, threshold 0.5)")
        ev.plot_metric_comparison(table, figures_dir / "model_comparison.png")
        rng = np.random.default_rng(RANDOM_SEED)
        sample_ids = list(rng.choice(test["match_id"].unique(), size=min(4, n_test_matches), replace=False))
        ev.plot_match_worm(test, test_probs[final_name], sample_ids, figures_dir / "win_probability_worms.png")
        importance = permutation_importance_logloss(final_model, X_test, y_test)
        ev.plot_feature_importance(importance, figures_dir / "feature_importance.png",
                                   f"Permutation importance - {final_name}")

    team_counts = pd.concat([train.drop_duplicates("match_id")["batting_team"],
                             train.drop_duplicates("match_id")["bowling_team"]]).value_counts()
    venue_counts = train.drop_duplicates("match_id")["venue"].value_counts()

    metadata = {
        "model_name": final_name,
        "base_model": selected,
        "calibration": calibration,
        "calibration_oof_log_loss": calibration_scores,
        "data_kind": data_kind,
        "data_source": data_source,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "teams": {k: int(v) for k, v in team_counts.items()},
        "venues": {k: int(v) for k, v in venue_counts.items()},
        "train_seasons": [int(train["season_year"].min()), int(train["season_year"].max())],
        "test_seasons": sorted(int(s) for s in test["season_year"].unique()),
        "n_train_matches": int(n_train_matches),
        "n_test_matches": int(n_test_matches),
        "n_train_states": int(len(train)),
        "n_test_states": int(len(test)),
        "train_chase_win_rate": float(y_train.mean()),
        "cv_folds": len(splits),
        "test_metrics": final_metrics,
        "comparison": table.drop(columns=["best_params"]).to_dict(orient="records"),
        "best_params": {r["model"]: r["best_params"] for r in rows},
        "metrics_by_phase": phase_table.to_dict(orient="records"),
        "permutation_importance": importance.sort_values(ascending=False).to_dict() if importance is not None else None,
        "quick_mode": quick,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "metadata": metadata}, model_path, compress=3)
    save_json({**metadata, "comparison": table.to_dict(orient="records")}, artifacts_dir / METRICS_PATH.name)
    logger.info("Saved model -> %s (%.1f KB)", model_path, model_path.stat().st_size / 1e3)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IPL win-probability models.")
    parser.add_argument("--quick", action="store_true", help="Use a small grid for a fast run.")
    parser.add_argument("--test-seasons", type=int, default=2, help="Most recent seasons held out for testing.")
    parser.add_argument("--cv-folds", type=int, default=4, help="Expanding-window CV folds (seasons).")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for GridSearchCV.")
    parser.add_argument("--rebuild-states", action="store_true", help="Rebuild chase states from cleaned data.")
    args = parser.parse_args()

    states = load_chase_states(rebuild=args.rebuild_states)
    meta = train_models(states, test_seasons=args.test_seasons, n_splits=args.cv_folds, quick=args.quick,
                        n_jobs=args.n_jobs)
    table = pd.DataFrame(meta["comparison"])
    cols = ["model", "cv_log_loss", "test_log_loss", "test_brier", "test_ece", "test_roc_auc", "test_accuracy",
            "test_f1"]
    print("\n" + table[cols].round(4).to_string(index=False))
    print(f"\nSelected: {meta['model_name']}")


if __name__ == "__main__":
    main()
