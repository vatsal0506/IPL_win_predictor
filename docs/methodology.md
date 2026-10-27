# Methodology notes

This document explains the cricket rules, data decisions and leakage safeguards behind the model in more
detail than the README.

## 1. Data source

* **Cricsheet IPL archive, CSV2 format** — <https://cricsheet.org/downloads/ipl_male_csv2.zip>
* Licence: **Open Data Commons Attribution License (ODC-By 1.0)**. Attribution: "Data provided by Cricsheet
  (https://cricsheet.org)". The raw data is *not* committed to this repository; it is downloaded by
  `python -m src.data_loader --download`.
* Each match has a ball-by-ball file (`<id>.csv`) and a metadata file (`<id>_info.csv`, rows of
  `info,<key>,<value>`), and the archive includes `all_matches.csv` with every delivery.
* Kaggle-style `matches.csv` + `deliveries.csv` files are also supported (columns resolved through alias
  lists, e.g. `inning`/`innings`, `batsman_runs`/`batter_runs`).

## 2. Cricket rules implemented

| Concept | Rule used |
|---|---|
| Legal delivery | Not a wide and not a no-ball. Byes and leg-byes are legal. |
| Runs | Runs off the bat + all extras (wides, no-balls, byes, leg-byes, penalty). |
| Wickets | Every dismissal except *retired hurt* / *retired not out* (the batter may return / was not out). Both `wicket_type` and `other_wicket_type` are counted. |
| Innings | Only innings 1 and 2. Innings 3+ are super overs and are discarded. |
| Target | First-innings total + 1. Cricsheet's recorded `target_runs` is cross-checked against this. |
| Balls remaining | 120 − legal balls bowled (full 20-over chase). |
| Current run rate | score × 6 / legal balls bowled (0 before the first ball). |
| Required run rate | runs remaining × 6 / legal balls remaining, clipped at 36 (6 per ball) as a model feature. |
| Chase finished | Target reached → batting side won. All out or 120 legal balls bowled → bowling side won, unless the scores are level (runs remaining = 1) → tie. |

Simplifications: matches with D/L (DLS) revisions or reduced overs are excluded rather than modelled, and the
order of deliveries within an over follows the source file order.

## 3. Excluded matches

| Reason | Why it is excluded |
|---|---|
| No result | No winner. |
| Tie (super over) | The regulation chase neither succeeded nor failed; labelling it either way would be arbitrary. |
| D/L method / reduced overs | Target or available balls changed, so the 20-over state features are wrong. |
| Inconsistent metadata | Missing innings or winner, winner not one of the teams, same team batting twice, recorded target ≠ first-innings total + 1, or recorded winner contradicting the ball-by-ball totals. |

Every exclusion and its reason is written to `data/processed/excluded_matches.csv`.

## 4. Match-state dataset

For each eligible match, one row is created for the state **before the first ball of the chase** and one
**after every delivery** of the chase, built only from cumulative totals up to that delivery. Rows where
the result is already known (target reached, all out, overs complete) are dropped, because the app handles
them with rules instead of the model.

Features: `batting_team`, `bowling_team`, `venue` (one-hot encoded) and `target`, `current_score`,
`runs_left`, `wickets_lost`, `wickets_left`, `balls_bowled`, `balls_left`, `current_run_rate`,
`required_run_rate`, `impact_player_era`.

Label: `chase_won` = 1 if the chasing side won the match.

`derive_state_features()` in `src/feature_engineering.py` computes the derived features for both training and
inference, so the two cannot drift apart. A unit test feeds a historical state through the inference path and
checks that the features are identical.

### Leakage safeguards

* No feature uses the final score, the result, or any later delivery.
* No historical team "form" or win-rate features are used, because computing them without future
  leakage would require careful point-in-time snapshots. Team identity is one-hot encoded instead.
* Train/test split is **by match and chronological**: the latest two seasons form the test set.
* Cross-validation uses **expanding-window season folds**: each fold trains only on seasons before its
  validation season. Matches never appear in two folds.
* The model is selected on cross-validated log loss. The calibration method is chosen on out-of-fold
  predictions, with each calibrator fitted on earlier folds and evaluated on the next. The test set is used
  only for the final report.

## 5. The era feature (`impact_player_era`)

The EDA shows average first-innings totals rising from ~165 to ~190 from 2023, when the Impact Player rule
was introduced. In the first training run (without this feature), every model **under-estimated** how
often chasing sides won in the 2025-26 test seasons (calibration curves above the diagonal). A binary flag
for the rules era (season ≥ 2023) was then added. It is known before any match is played, so it is not
leakage. Because this change was made **after looking at the first run's test results**, the test seasons
are no longer a perfectly untouched hold-out for the final model. Both runs are reported in the README.
First-run outputs are kept in `artifacts/run1_baseline_features/`.

## 6. Known limitations

* Team identity is a weak proxy for squad strength, which changes every season.
* No player-level, pitch, toss, dew or weather information.
* States within a match are strongly correlated. The effective sample size is roughly the number of
  matches (about 1,000 for training), not the number of rows.
* Only 2023-2024 data represents the current era in training, so era effects are estimated from relatively
  few matches.
