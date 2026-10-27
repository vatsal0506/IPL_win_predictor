"""Reusable inference: validate a chase situation and return a win probability.

The UI (``app.py``) only collects inputs and displays results; every cricket
rule and model call lives here so it can be unit-tested.

Example::

    from src.predict import MatchState, WinProbabilityPredictor
    predictor = WinProbabilityPredictor.load()
    result = predictor.predict(MatchState("Mumbai Indians", "Chennai Super Kings",
                                          "Wankhede Stadium", target=181, current_score=96,
                                          wickets_lost=3, overs_completed=11, balls_in_over=2))
    print(result.batting_win_probability)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import BASE_NUMERIC_INPUTS, FEATURE_COLUMNS, derive_state_features
from src.utils import BALLS_PER_INNINGS, BALLS_PER_OVER, MAX_WICKETS, MODEL_PATH, OVERS_PER_INNINGS, UNKNOWN_VENUE

MAX_TARGET = 400  # far above the highest IPL total; guards against typos


class InvalidMatchStateError(ValueError):
    """The supplied match situation is impossible or inconsistent."""


class ModelNotAvailableError(RuntimeError):
    """The trained model artifact is missing, unreadable or not trained on real data."""


@dataclass(frozen=True)
class MatchState:
    batting_team: str
    bowling_team: str
    venue: str
    target: int
    current_score: int
    wickets_lost: int
    overs_completed: int
    balls_in_over: int = 0
    impact_player_era: bool = True  # current IPL rules; False simulates pre-2023 conditions

    @property
    def balls_bowled(self) -> int:
        """Legal balls bowled. Overs notation 12.3 means 12 overs + 3 legal balls = 75 balls."""
        return self.overs_completed * BALLS_PER_OVER + self.balls_in_over


@dataclass
class SituationSummary:
    runs_left: int
    balls_left: int
    wickets_left: int
    current_run_rate: float
    required_run_rate: float | None


@dataclass
class PredictionResult:
    """``status`` is 'in_progress', 'batting_won', 'bowling_won' or 'tie'."""

    status: str
    summary: SituationSummary
    batting_win_probability: float | None = None
    message: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def bowling_win_probability(self) -> float | None:
        if self.batting_win_probability is None:
            return None
        return 1.0 - self.batting_win_probability

    @property
    def is_live(self) -> bool:
        return self.status == "in_progress"


def _as_int(name: str, value) -> int:
    if isinstance(value, bool) or value is None:
        raise InvalidMatchStateError(f"{name} must be a whole number.")
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        raise InvalidMatchStateError(f"{name} must be a whole number.") from None
    if not np.isfinite(as_float) or as_float != int(as_float):
        raise InvalidMatchStateError(f"{name} must be a whole number.")
    return int(as_float)


def validate_state(state: MatchState) -> MatchState:
    """Return a normalised copy of ``state`` or raise ``InvalidMatchStateError`` with a clear message."""
    errors = []
    bat = (state.batting_team or "").strip()
    bowl = (state.bowling_team or "").strip()
    if not bat or not bowl:
        errors.append("Both the batting and the bowling team must be selected.")
    elif bat == bowl:
        errors.append("The batting and bowling teams must be different.")

    try:
        target = _as_int("Target", state.target)
        score = _as_int("Current score", state.current_score)
        wickets = _as_int("Wickets lost", state.wickets_lost)
        overs = _as_int("Overs completed", state.overs_completed)
        balls = _as_int("Balls in the current over", state.balls_in_over)
    except InvalidMatchStateError as exc:
        raise InvalidMatchStateError(str(exc)) from None

    if not 1 <= target <= MAX_TARGET:
        errors.append(f"Target must be between 1 and {MAX_TARGET} runs.")
    if score < 0:
        errors.append("Current score cannot be negative.")
    if not 0 <= wickets <= MAX_WICKETS:
        errors.append(f"Wickets lost must be between 0 and {MAX_WICKETS}.")
    if not 0 <= overs <= OVERS_PER_INNINGS:
        errors.append(f"Overs completed must be between 0 and {OVERS_PER_INNINGS}.")
    if not 0 <= balls < BALLS_PER_OVER:
        errors.append(f"Balls in the current over must be between 0 and {BALLS_PER_OVER - 1}.")
    if overs == OVERS_PER_INNINGS and balls > 0:
        errors.append("An innings has only 20 overs - balls must be 0 once 20 overs are complete.")
    # The chase stops the moment the target is reached, so a score can exceed
    # the target by at most one scoring shot (max 6 off the bat + extras).
    if target >= 1 and score >= target + 7:
        errors.append("Current score is too far beyond the target - the chase ends as soon as the target is reached.")
    if errors:
        raise InvalidMatchStateError(" ".join(errors))

    return MatchState(bat, bowl, (state.venue or UNKNOWN_VENUE).strip() or UNKNOWN_VENUE,
                      target, score, wickets, overs, balls, bool(state.impact_player_era))


def summarize(state: MatchState) -> SituationSummary:
    """Cricket arithmetic for display; mirrors ``derive_state_features``."""
    row = derive_state_features(pd.DataFrame([{
        "target": state.target, "current_score": state.current_score,
        "wickets_lost": state.wickets_lost, "balls_bowled": state.balls_bowled,
    }])).iloc[0]
    balls_left = int(row["balls_left"])
    runs_left = int(row["runs_left"])
    rrr = None
    if balls_left > 0 and runs_left > 0:
        rrr = runs_left * BALLS_PER_OVER / balls_left  # unclipped for display
    return SituationSummary(
        runs_left=runs_left,
        balls_left=balls_left,
        wickets_left=int(row["wickets_left"]),
        current_run_rate=float(row["current_run_rate"]),
        required_run_rate=rrr,
    )


def resolve_completed(state: MatchState, summary: SituationSummary) -> PredictionResult | None:
    """Return a final result if the chase is already decided, else ``None``."""
    if summary.runs_left <= 0:
        return PredictionResult("batting_won", summary, message=f"{state.batting_team} have reached the target "
                                f"and won by {summary.wickets_left} wicket(s).")
    if summary.wickets_left == 0 or summary.balls_left == 0:
        how = "were bowled out" if summary.wickets_left == 0 else "used all 20 overs"
        if summary.runs_left == 1:
            # Scores level (target - 1): a tie, decided by a super over in the IPL.
            return PredictionResult("tie", summary, message=f"{state.batting_team} {how} with the scores level - "
                                    "the match is tied and goes to a super over.")
        return PredictionResult("bowling_won", summary, message=f"{state.batting_team} {how}. "
                                f"{state.bowling_team} won by {summary.runs_left - 1} run(s).")
    return None


class WinProbabilityPredictor:
    def __init__(self, model, metadata: dict):
        self.model = model
        self.metadata = metadata
        missing = set(FEATURE_COLUMNS) - set(metadata.get("feature_columns", []))
        if missing:
            raise ModelNotAvailableError(
                f"Model artifact feature schema is incompatible (missing {sorted(missing)}). Retrain with "
                "`python -m src.train`."
            )

    @classmethod
    def load(cls, path: Path = MODEL_PATH, allow_synthetic: bool = False) -> "WinProbabilityPredictor":
        path = Path(path)
        if not path.exists():
            raise ModelNotAvailableError(
                f"No trained model found at '{path}'. Download the data and train first:\n"
                "  python -m src.data_loader --download\n  python -m src.preprocessing\n  python -m src.train"
            )
        try:
            bundle = joblib.load(path)
            model, metadata = bundle["model"], bundle["metadata"]
        except Exception as exc:  # corrupted file, version mismatch, wrong format ...
            raise ModelNotAvailableError(
                f"Could not load the model artifact at '{path}' ({type(exc).__name__}: {exc}). "
                "Retrain it with `python -m src.train` using the versions in requirements.txt."
            ) from exc
        if metadata.get("data_kind") != "real" and not allow_synthetic:
            raise ModelNotAvailableError(
                "The saved model was trained on SYNTHETIC test data, not real IPL data, so its probabilities "
                "are meaningless. Train on the real dataset with `python -m src.train`."
            )
        return cls(model, metadata)

    @property
    def known_teams(self) -> list[str]:
        return sorted(self.metadata.get("teams", {}))

    @property
    def known_venues(self) -> list[str]:
        return list(self.metadata.get("venues", {}))

    def feature_frame(self, states: list[MatchState]) -> pd.DataFrame:
        raw = pd.DataFrame([{
            "batting_team": s.batting_team, "bowling_team": s.bowling_team, "venue": s.venue,
            "target": s.target, "current_score": s.current_score,
            "wickets_lost": s.wickets_lost, "balls_bowled": s.balls_bowled,
            "impact_player_era": int(s.impact_player_era),
        } for s in states])
        raw[BASE_NUMERIC_INPUTS] = raw[BASE_NUMERIC_INPUTS].astype(int)
        return derive_state_features(raw)[FEATURE_COLUMNS]

    def predict(self, state: MatchState) -> PredictionResult:
        state = validate_state(state)
        summary = summarize(state)
        finished = resolve_completed(state, summary)
        if finished is not None:
            return finished

        warnings = []
        if state.batting_team not in self.metadata.get("teams", {}):
            warnings.append(f"'{state.batting_team}' was not in the training data; it is treated as a generic team.")
        if state.bowling_team not in self.metadata.get("teams", {}):
            warnings.append(f"'{state.bowling_team}' was not in the training data; it is treated as a generic team.")
        if state.venue not in self.metadata.get("venues", {}):
            warnings.append("Venue not seen in training; venue effects are treated as generic.")

        prob = float(self.model.predict_proba(self.feature_frame([state]))[0, 1])
        if not np.isfinite(prob) or not 0.0 <= prob <= 1.0:
            raise RuntimeError(f"Model returned an invalid probability ({prob}).")
        return PredictionResult("in_progress", summary, batting_win_probability=prob, warnings=warnings)

    def predict_many(self, states: list[MatchState]) -> np.ndarray:
        """Batch probabilities for *live* states (used for what-if charts)."""
        states = [validate_state(s) for s in states]
        return self.model.predict_proba(self.feature_frame(states))[:, 1]


def predict_win_probability(state: MatchState, path: Path = MODEL_PATH) -> PredictionResult:
    """Convenience one-shot helper."""
    return WinProbabilityPredictor.load(path).predict(state)


__all__ = [
    "BALLS_PER_INNINGS", "InvalidMatchStateError", "MatchState", "ModelNotAvailableError", "PredictionResult",
    "SituationSummary", "WinProbabilityPredictor", "predict_win_probability", "resolve_completed",
    "summarize", "validate_state",
]
