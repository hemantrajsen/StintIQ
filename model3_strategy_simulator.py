"""StintIQ Model 3: Monte Carlo race-strategy simulator.

The runtime keeps the project dependency set small. Model 2 CatBoost and Model 4
joblib artifacts are optional; both have deterministic interfaces and fallbacks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

logger = logging.getLogger("StintIQ.Model3")

FUEL_KG_PER_LAP = 1.75
FUEL_SEC_PER_KG = 0.030
DRIVER_NOISE_STD = 0.35
MODEL2_RESIDUAL_STD = 0.5811
PIT_STOP_DWELL_MEAN = 2.4
PIT_STOP_DWELL_STD = 0.3
SLOW_STOP_PROB = 0.05
SLOW_STOP_PENALTY = 2.5
SC_PIT_LOSS_FRACTION = 0.55
BASE_NEUTRAL_PROB_PER_LAP = 0.035
Q10_NORMAL = 1.2815515655446004

CLIFF_PRIORS: Dict[str, int] = {"SOFT": 14, "MEDIUM": 20, "HARD": 27}
CIRCUIT_ABRASIVENESS: Dict[str, float] = {
    "Barcelona": 3.0, "Bahrain": 3.0, "Great Britain": 3.0, "Japan": 3.0,
    "Spain": 2.8, "Australia": 2.5, "Hungary": 2.5, "Netherlands": 2.5,
    "Singapore": 2.5, "Austria": 2.5, "Belgium": 2.5, "Miami": 2.5,
    "China": 2.0, "Italy": 2.0, "Canada": 1.5, "Azerbaijan": 1.5,
    "Monaco": 1.0,
}
CIRCUIT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "Australia": {"total_laps": 58, "base_lap": 82.091, "pit_green": 16.2, "pit_sc": 8.9, "abrasiveness": 2.5, "temp": 36.0},
    "Austria": {"total_laps": 71, "base_lap": 70.374, "pit_green": 16.9, "pit_sc": 9.3, "abrasiveness": 2.5, "temp": 32.0},
    "Azerbaijan": {"total_laps": 51, "base_lap": 104.916, "pit_green": 35.0, "pit_sc": 19.3, "abrasiveness": 1.5, "temp": 30.0},
    "Bahrain": {"total_laps": 55, "base_lap": 98.220, "pit_green": 24.5, "pit_sc": 13.5, "abrasiveness": 3.0, "temp": 35.0},
    "Barcelona": {"total_laps": 66, "base_lap": 80.122, "pit_green": 20.2, "pit_sc": 11.1, "abrasiveness": 3.0, "temp": 38.0},
    "Belgium": {"total_laps": 45, "base_lap": 108.890, "pit_green": 19.9, "pit_sc": 10.9, "abrasiveness": 2.5, "temp": 28.0},
    "Canada": {"total_laps": 68, "base_lap": 74.210, "pit_green": 18.5, "pit_sc": 10.2, "abrasiveness": 1.5, "temp": 30.0},
    "China": {"total_laps": 56, "base_lap": 95.275, "pit_green": 22.0, "pit_sc": 12.1, "abrasiveness": 2.0, "temp": 29.0},
    "Great Britain": {"total_laps": 52, "base_lap": 91.777, "pit_green": 20.5, "pit_sc": 11.3, "abrasiveness": 3.0, "temp": 31.0},
    "Hungary": {"total_laps": 70, "base_lap": 82.000, "pit_green": 18.5, "pit_sc": 10.2, "abrasiveness": 2.5, "temp": 42.0},
    "Italy": {"total_laps": 53, "base_lap": 83.504, "pit_green": 24.0, "pit_sc": 13.2, "abrasiveness": 2.0, "temp": 34.0},
    "Japan": {"total_laps": 53, "base_lap": 92.432, "pit_green": 22.5, "pit_sc": 12.4, "abrasiveness": 3.0, "temp": 30.0},
    "Miami": {"total_laps": 57, "base_lap": 91.869, "pit_green": 16.5, "pit_sc": 9.1, "abrasiveness": 2.5, "temp": 38.0},
    "Monaco": {"total_laps": 78, "base_lap": 73.481, "pit_green": 21.0, "pit_sc": 11.6, "abrasiveness": 1.0, "temp": 35.0},
    "Netherlands": {"total_laps": 72, "base_lap": 74.230, "pit_green": 16.5, "pit_sc": 9.1, "abrasiveness": 2.5, "temp": 29.0},
    "Singapore": {"total_laps": 62, "base_lap": 95.202, "pit_green": 22.0, "pit_sc": 12.1, "abrasiveness": 2.5, "temp": 33.0},
    "Spain": {"total_laps": 57, "base_lap": 95.587, "pit_green": 21.5, "pit_sc": 11.8, "abrasiveness": 2.8, "temp": 36.0},
}
STREET_LEVEL = {"Azerbaijan": 1.0, "Miami": 1.0, "Monaco": 1.0, "Singapore": 1.0}
WALL_RISK = {"Azerbaijan": 3.0, "Monaco": 3.0, "Singapore": 3.0, "Miami": 2.0}


@dataclass
class CircuitProfile:
    """Circuit conditions and the race-level Model 1 pace reference."""

    name: str
    total_laps: int
    base_lap_time: float
    pit_loss_green: float
    pit_loss_sc: float
    abrasiveness: float = 2.5
    track_temp: float = 35.0
    air_temp: float = 25.0
    humidity: float = 50.0
    driver_delta_seconds: float = 0.0
    street_level: float = 0.0
    wall_risk: float = 1.0
    rain_prev: float = 0.0
    wet_frac_prev: float = 0.0
    rain_change: float = 0.0

    @classmethod
    def from_name(cls, circuit_name: str) -> "CircuitProfile":
        key = circuit_name.replace("2026_", "").strip()
        data = CIRCUIT_DEFAULTS.get(key, {
            "total_laps": 57, "base_lap": 90.0, "pit_green": 20.0, "pit_sc": 11.0,
            "abrasiveness": CIRCUIT_ABRASIVENESS.get(key, 2.5), "temp": 35.0,
        })
        return cls(
            name=key,
            total_laps=int(data["total_laps"]),
            base_lap_time=float(data["base_lap"]),
            pit_loss_green=float(data["pit_green"]),
            pit_loss_sc=float(data["pit_sc"]),
            abrasiveness=float(data.get("abrasiveness", CIRCUIT_ABRASIVENESS.get(key, 2.5))),
            track_temp=float(data.get("temp", 35.0)),
            street_level=STREET_LEVEL.get(key, 0.0),
            wall_risk=WALL_RISK.get(key, 1.0),
        )

    @classmethod
    def from_raw_data(
        cls, raw_csv_path: str | Path | pd.DataFrame, circuit_name: str
    ) -> "CircuitProfile":
        """Use the event's fastest accurate green lap as its reference pace."""
        profile = cls.from_name(circuit_name)
        try:
            frame = raw_csv_path.copy() if isinstance(raw_csv_path, pd.DataFrame) else pd.read_csv(raw_csv_path, low_memory=False)
            key = circuit_name.replace("2026_", "").strip().casefold()
            mask = pd.Series(False, index=frame.index)
            if "CircuitName" in frame:
                mask |= frame["CircuitName"].astype(str).str.strip().str.casefold().eq(key)
            if "EventName" in frame:
                event_names = frame["EventName"].astype(str).str.replace(r"^2026_", "", regex=True).str.strip().str.casefold()
                mask |= event_names.eq(key)
            subset = frame.loc[mask].copy()
            if "Year" in subset:
                years = pd.to_numeric(subset["Year"], errors="coerce")
                subset = subset.loc[years.eq(2026)]
            if subset.empty:
                return profile

            if "LapTimeSeconds" in subset:
                subset["LapTimeSeconds"] = pd.to_numeric(subset["LapTimeSeconds"], errors="coerce")
            else:
                subset["LapTimeSeconds"] = pd.to_timedelta(subset.get("LapTime"), errors="coerce").dt.total_seconds()
            clean = subset.loc[subset["LapTimeSeconds"].gt(0)].copy()
            if "IsAccurate" in clean:
                accurate = clean["IsAccurate"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
                clean = clean.loc[accurate]
            if "TrackStatus" in clean:
                clean = clean.loc[clean["TrackStatus"].astype(str).str.strip().eq("1")]
            for col in ("PitInTime", "PitOutTime"):
                if col in clean:
                    has_time = clean[col].notna() & clean[col].astype(str).str.strip().ne("")
                    clean = clean.loc[~has_time]
            if clean.empty:
                return profile

            total_laps = pd.to_numeric(subset.get("TotalLaps"), errors="coerce").dropna()
            def median_col(column: str, fallback: float) -> float:
                values = pd.to_numeric(clean.get(column), errors="coerce").dropna()
                return float(values.median()) if not values.empty else fallback

            return cls(
                name=profile.name,
                total_laps=int(total_laps.max()) if not total_laps.empty else profile.total_laps,
                base_lap_time=float(clean["LapTimeSeconds"].min()),
                pit_loss_green=profile.pit_loss_green,
                pit_loss_sc=profile.pit_loss_green * SC_PIT_LOSS_FRACTION,
                abrasiveness=profile.abrasiveness,
                track_temp=median_col("TrackTemp", profile.track_temp),
                air_temp=median_col("AirTemp", profile.air_temp),
                humidity=median_col("Humidity", profile.humidity),
                driver_delta_seconds=profile.driver_delta_seconds,
                street_level=profile.street_level,
                wall_risk=profile.wall_risk,
            )
        except Exception as exc:
            logger.warning("Could not derive %s profile from raw data (%s); using defaults.", circuit_name, exc)
            return profile


def driver_pace_delta(
    model_ready: str | Path | pd.DataFrame,
    driver: str,
    team: Optional[str] = None,
    exclude_event: Optional[str] = None,
) -> float:
    """Median Model 1 target delta, weighted equally by event; team fallback if needed."""
    try:
        frame = model_ready.copy() if isinstance(model_ready, pd.DataFrame) else pd.read_csv(model_ready, low_memory=False)
        if "lap_time_delta" not in frame:
            return 0.0
        frame["lap_time_delta"] = pd.to_numeric(frame["lap_time_delta"], errors="coerce")
        if exclude_event and "EventName" in frame:
            frame = frame.loc[frame["EventName"].astype(str).ne(exclude_event)]
        if "Driver" in frame:
            driver_rows = frame.loc[frame["Driver"].astype(str).str.upper().eq(str(driver).upper())]
            if not driver_rows.empty:
                if "EventName" in driver_rows:
                    event_medians = driver_rows.groupby("EventName")["lap_time_delta"].median()
                    value = event_medians.median()
                else:
                    value = driver_rows["lap_time_delta"].median()
                if pd.notna(value):
                    return float(value)
        if team and "Team" in frame:
            team_rows = frame.loc[frame["Team"].astype(str).str.casefold().eq(str(team).casefold())]
            if not team_rows.empty:
                if "EventName" in team_rows:
                    value = team_rows.groupby("EventName")["lap_time_delta"].median().median()
                else:
                    value = team_rows["lap_time_delta"].median()
                if pd.notna(value):
                    return float(value)
    except Exception as exc:
        logger.warning("Could not load Model 1 pace benchmark (%s); using zero driver delta.", exc)
    return 0.0


@dataclass
class Stint:
    """One continuous tire stint. Lap numbers are one-indexed and inclusive."""

    compound: str
    start_lap: int
    end_lap: int

    @property
    def length(self) -> int:
        return max(0, self.end_lap - self.start_lap + 1)

    @property
    def cliff_lap(self) -> int:
        return CLIFF_PRIORS.get(self.compound.upper(), 22)

    @property
    def exceeds_cliff(self) -> bool:
        return self.length > self.cliff_lap

    @property
    def cliff_overage(self) -> int:
        return max(0, self.length - self.cliff_lap)


@dataclass
class Strategy:
    """End-to-end dry strategy."""

    stints: List[Stint]

    @property
    def total_stops(self) -> int:
        return max(0, len(self.stints) - 1)

    @property
    def compounds_used(self) -> Set[str]:
        return {stint.compound.upper() for stint in self.stints}

    @property
    def is_fia_legal(self) -> bool:
        return len(self.compounds_used) >= 2

    @property
    def has_cliff_warning(self) -> bool:
        return any(stint.exceeds_cliff for stint in self.stints)

    @property
    def pit_laps(self) -> List[int]:
        return [stint.end_lap for stint in self.stints[:-1]]

    def summary(self) -> str:
        return " ➔ ".join(
            f"{stint.compound[0]} (L{stint.start_lap}–L{stint.end_lap})"
            for stint in self.stints
        )

    def full_name(self) -> str:
        compounds = " ➔ ".join(stint.compound.upper() for stint in self.stints)
        pits = ",".join(map(str, self.pit_laps)) or "none"
        return f"{compounds} (Pit L{pits})"


@dataclass
class SimulationResult:
    """Race-time distribution for one candidate strategy."""

    strategy: Strategy
    mean_time: float
    median_time: float
    std_time: float
    q10_time: float
    q90_time: float
    sc_benefit_pct: float
    best_time: float
    worst_time: float
    lap_pace_deterministic: np.ndarray = field(repr=False)
    lap_time_samples: Optional[np.ndarray] = field(default=None, repr=False)
    stint_trajectories: List[Dict[str, Any]] = field(default_factory=list, repr=False)


class Model2TireDegradationBridge:
    """Model 2 q10/q50/q90 bridge with the roadmap's parametric fallback."""

    def __init__(self, model_path: Optional[str | Path] = None):
        self.model = None
        self.has_catboost = False
        self.model_path = Path(model_path) if model_path else (
            Path(__file__).resolve().parent / "Model Development" / "stintiq_model2_catboost.cbm"
        )
        self._cache: Dict[tuple, np.ndarray] = {}
        if self.model_path.exists():
            try:
                from catboost import CatBoostRegressor
                model = CatBoostRegressor()
                model.load_model(str(self.model_path))
                self.model = model
                self.has_catboost = True
                logger.info("Loaded Model 2 CatBoost artifact from %s", self.model_path)
            except Exception as exc:
                logger.info("Model 2 CatBoost unavailable (%s); using the specified parametric fallback.", exc)
        else:
            logger.info("Model 2 artifact not found at %s; using parametric fallback.", self.model_path)

    def _parametric_quantiles(
        self, compound: str, stint_length: int, abrasiveness: float, track_temp: float
    ) -> np.ndarray:
        ages = np.arange(1, stint_length + 1, dtype=np.float64)
        cliff = CLIFF_PRIORS.get(compound, 22)
        abrasion_factor = max(0.1, abrasiveness / 2.5)
        if compound == "SOFT":
            median = 0.045 * ages + 0.0025 * ages**2 + 0.15 * np.maximum(0.0, ages - cliff)
        elif compound == "MEDIUM":
            median = 0.032 * ages + 0.0012 * ages**2 + 0.09 * np.maximum(0.0, ages - cliff)
        else:
            median = 0.018 * ages + 0.0006 * ages**2 + 0.06 * np.maximum(0.0, ages - cliff)
        temp_factor = 1.0 + max(0.0, track_temp - 35.0) * 0.015
        median = np.maximum.accumulate(median * abrasion_factor * temp_factor)
        spread = Q10_NORMAL * MODEL2_RESIDUAL_STD
        q10 = np.maximum(-0.25, median - spread)
        q90 = median + spread
        return np.column_stack([q10, median, q90]).astype(np.float64)

    def get_stint_quantiles(
        self,
        compound: str,
        stint_length: int,
        abrasiveness: float = 2.5,
        track_temp: float = 35.0,
        air_temp: float = 25.0,
        humidity: float = 50.0,
        driver: str = "UNK",
        team: str = "UNK",
    ) -> np.ndarray:
        """Return per-age q10, q50, q90 degradation predictions."""
        length = int(stint_length)
        if length <= 0:
            return np.empty((0, 3), dtype=np.float64)
        comp = compound.upper()
        key = (comp, length, round(abrasiveness, 3), round(track_temp, 2),
               round(air_temp, 2), round(humidity, 2), driver, team, self.has_catboost)
        if key in self._cache:
            return self._cache[key].copy()

        quantiles: Optional[np.ndarray] = None
        if self.has_catboost and self.model is not None:
            try:
                cmap = {"SOFT": 1.0, "MEDIUM": 2.0, "HARD": 3.0}
                rows: List[List[float]] = []
                lag_history: List[float] = []
                for age in range(1, length + 1):
                    lag1 = lag_history[-1] if lag_history else 0.0
                    lag2 = lag_history[-2] if len(lag_history) >= 2 else 0.0
                    roll3 = float(np.mean(lag_history[-3:])) if lag_history else 0.0
                    row = pd.DataFrame([{
                        "TyreLife": float(age),
                        "TyreLifeSq": float(age**2),
                        "TyreLifeXCompound": float(age * cmap.get(comp, 2.0)),
                        "TyreLifeXTrackTemp": float(age * track_temp),
                        "TyreLifeXAbrasive": float(age * abrasiveness),
                        "TrackAbrasiveness": float(abrasiveness),
                        "TrackTemp": float(track_temp),
                        "AirTemp": float(air_temp),
                        "Humidity": float(humidity),
                        "StintNum": 1,
                        "SessionProgress": min(1.0, age / 60.0),
                        "lag_1_deg": lag1,
                        "lag_2_deg": lag2,
                        "rolling_deg_3": roll3,
                        "Compound": comp,
                        "Driver": str(driver),
                        "Team": str(team),
                    }])
                    prediction = np.asarray(self.model.predict(row), dtype=np.float64).reshape(-1)
                    if prediction.size >= 3:
                        q10, q50, q90 = np.sort(prediction[:3])
                    else:
                        q50 = float(prediction[0])
                        q10, q90 = q50 - 0.15, q50 + 0.15
                    q50 = max(0.0, float(q50))
                    q10 = min(q50, max(-0.25, float(q10)))
                    q90 = max(q50, float(q90))
                    rows.append([q10, q50, q90])
                    lag_history.append(q50)
                quantiles = np.asarray(rows, dtype=np.float64)
                quantiles = np.maximum.accumulate(quantiles, axis=0)
            except Exception as exc:
                logger.warning("Model 2 inference failed (%s); using parametric fallback.", exc)

        if quantiles is None:
            quantiles = self._parametric_quantiles(comp, length, abrasiveness, track_temp)
        self._cache[key] = quantiles
        return quantiles.copy()

    def get_stint_trajectory(self, compound: str, stint_length: int, **kwargs: Any) -> np.ndarray:
        """Backward-compatible median trajectory used by earlier Model 3 callers."""
        return self.get_stint_quantiles(compound, stint_length, **kwargs)[:, 1]


class Model4HazardBridge:
    """Model 4 SC/VSC simulator with a no-dependency empirical fallback."""

    def __init__(self, joblib_path: Optional[str | Path] = None):
        self.predictor = None
        self.has_joblib = False
        self.model_path = Path(joblib_path) if joblib_path else (
            Path(__file__).resolve().parent / "Model Development" / "models" / "model4_sc_vsc.joblib"
        )
        if self.model_path.exists():
            try:
                import sys
                import joblib
                model_dir = str(Path(__file__).resolve().parent / "Model Development")
                if model_dir not in sys.path:
                    sys.path.insert(0, model_dir)
                self.predictor = joblib.load(self.model_path)
                self.has_joblib = True
                logger.info("Loaded Model 4 hazard model from %s", self.model_path)
            except Exception as exc:
                logger.info("Model 4 artifact unavailable (%s); using constant-hazard fallback.", exc)

    def sample_neutralisations(
        self,
        total_laps: int,
        n_sims: int,
        rng: np.random.Generator,
        context: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Boolean (simulation, lap) matrix, with True on neutralized laps."""
        if self.has_joblib and self.predictor is not None:
            try:
                ctx = {"total_laps": int(total_laps)}
                ctx.update(context or {})
                sim = self.predictor.simulate(ctx=ctx, n_sims=n_sims, rng=rng)
                neutral = np.asarray(sim["neutral"], dtype=bool)
                if neutral.shape == (n_sims, total_laps):
                    return neutral
            except Exception as exc:
                logger.warning("Model 4 simulation failed (%s); using constant hazard.", exc)

        neutral = np.zeros((n_sims, total_laps), dtype=bool)
        durations_vsc = np.array([2, 3, 4, 4, 4, 6], dtype=int)
        remaining = np.zeros(n_sims, dtype=int)
        for lap in range(total_laps):
            ongoing = remaining > 0
            neutral[ongoing, lap] = True
            remaining[ongoing] -= 1
            at_risk = ~ongoing
            onset = at_risk & (rng.random(n_sims) < BASE_NEUTRAL_PROB_PER_LAP)
            if onset.any():
                indices = np.flatnonzero(onset)
                is_sc = rng.random(len(indices)) < 0.20
                vsc_durations = rng.choice(durations_vsc, size=len(indices))
                duration = np.where(is_sc, 4, vsc_durations)
                remaining[indices] = np.maximum(duration - 1, 0)
                neutral[indices, lap] = True
        return neutral


@dataclass
class _Worlds:
    lap_noise: np.ndarray
    neutral: np.ndarray
    pit_transit: np.ndarray
    dwell: np.ndarray
    slow_stop: np.ndarray
    degradation_samples: Dict[str, np.ndarray]
    degradation_cumulative: Dict[str, np.ndarray]
    quantiles: Dict[str, np.ndarray]


class StrategySimulator:
    """Vectorized candidate search using common Monte Carlo scenarios per circuit."""

    def __init__(
        self,
        deg_bridge: Optional[Model2TireDegradationBridge] = None,
        hazard_bridge: Optional[Model4HazardBridge] = None,
    ):
        self.deg_bridge = deg_bridge or Model2TireDegradationBridge()
        self.hazard_bridge = hazard_bridge or Model4HazardBridge()

    def generate_candidate_strategies(
        self, total_laps: int, max_stops: int = 2, min_stint: int = 10, step: int = 2
    ) -> List[Strategy]:
        if total_laps < 2 * min_stint or max_stops < 1:
            return []
        candidates: List[Strategy] = []
        compounds = ("SOFT", "MEDIUM", "HARD")
        for first in compounds:
            for second in compounds:
                if first == second:
                    continue
                for pit_lap in range(min_stint, total_laps - min_stint + 1, step):
                    candidates.append(Strategy([
                        Stint(first, 1, pit_lap), Stint(second, pit_lap + 1, total_laps)
                    ]))
        if max_stops >= 2:
            step_two = max(3, step)
            for first in compounds:
                for second in compounds:
                    for third in compounds:
                        if len({first, second, third}) < 2:
                            continue
                        for pit_one in range(min_stint, total_laps - 2 * min_stint + 1, step_two):
                            for pit_two in range(pit_one + min_stint, total_laps - min_stint + 1, step_two):
                                candidates.append(Strategy([
                                    Stint(first, 1, pit_one),
                                    Stint(second, pit_one + 1, pit_two),
                                    Stint(third, pit_two + 1, total_laps),
                                ]))
        return candidates

    @staticmethod
    def _model4_context(profile: CircuitProfile) -> Dict[str, float]:
        return {
            "total_laps": float(profile.total_laps),
            "street_level": float(profile.street_level),
            "wall_risk": float(profile.wall_risk),
            "track_temp": float(profile.track_temp),
            "air_temp": float(profile.air_temp),
            "humidity": float(profile.humidity),
            "rain_prev": float(profile.rain_prev),
            "wet_frac_prev": float(profile.wet_frac_prev),
            "rain_change": float(profile.rain_change),
        }

    def _prepare_worlds(
        self,
        profile: CircuitProfile,
        n_sims: int,
        rng: np.random.Generator,
        max_stops: int,
        driver: str = "UNK",
        team: str = "UNK",
    ) -> _Worlds:
        laps = profile.total_laps
        lap_noise = rng.normal(0.0, DRIVER_NOISE_STD, size=(n_sims, laps))
        neutral = self.hazard_bridge.sample_neutralisations(
            laps, n_sims, rng, context=self._model4_context(profile)
        )
        # The same synthetic world is used for every candidate to reduce ranking noise.
        stop_scenarios = max(max_stops, profile.total_laps - 1)
        dwell = rng.normal(PIT_STOP_DWELL_MEAN, PIT_STOP_DWELL_STD, size=(stop_scenarios, n_sims))
        slow = (rng.random((stop_scenarios, n_sims)) < SLOW_STOP_PROB) * SLOW_STOP_PENALTY
        pit_transit = np.where(neutral, profile.pit_loss_sc, profile.pit_loss_green)

        max_stint = max(1, laps)
        age_z = rng.standard_normal((n_sims, max_stint))
        quantiles: Dict[str, np.ndarray] = {}
        samples: Dict[str, np.ndarray] = {}
        cumulative: Dict[str, np.ndarray] = {}
        for compound in ("SOFT", "MEDIUM", "HARD"):
            q = self.deg_bridge.get_stint_quantiles(
                compound=compound,
                stint_length=max_stint,
                abrasiveness=profile.abrasiveness,
                track_temp=profile.track_temp,
                air_temp=profile.air_temp,
                humidity=profile.humidity,
                driver=driver,
                team=team,
            )
            sigma = np.maximum(0.05, (q[:, 2] - q[:, 0]) / (2.0 * Q10_NORMAL))
            sampled = np.clip(q[:, 1][None, :] + age_z * sigma[None, :], q[:, 0][None, :], q[:, 2][None, :])
            sampled = np.maximum.accumulate(sampled, axis=1)
            quantiles[compound] = q
            samples[compound] = sampled
            cumulative[compound] = np.cumsum(sampled, axis=1)
        return _Worlds(
            lap_noise=lap_noise,
            neutral=neutral,
            pit_transit=pit_transit,
            dwell=dwell,
            slow_stop=slow,
            degradation_samples=samples,
            degradation_cumulative=cumulative,
            quantiles=quantiles,
        )

    @staticmethod
    def _base_lap_vector(profile: CircuitProfile) -> np.ndarray:
        starting_fuel = min(100.0, profile.total_laps * FUEL_KG_PER_LAP)
        remaining_fuel = np.linspace(starting_fuel, 0.0, profile.total_laps, dtype=np.float64)
        return (
            profile.base_lap_time + profile.driver_delta_seconds
            + remaining_fuel * FUEL_SEC_PER_KG
        )

    def _evaluate_with_worlds(
        self, profile: CircuitProfile, strategy: Strategy, worlds: _Worlds
    ) -> SimulationResult:
        if not strategy.stints:
            raise ValueError("A strategy must contain at least one stint.")
        if sum(stint.length for stint in strategy.stints) != profile.total_laps:
            raise ValueError("Strategy stints must cover every race lap exactly once.")
        if not strategy.is_fia_legal:
            raise ValueError("A dry strategy must use at least two distinct compounds.")

        base_lap = self._base_lap_vector(profile)
        median_segments = [
            worlds.quantiles[stint.compound.upper()][:stint.length, 1]
            for stint in strategy.stints
        ]
        deterministic = base_lap + np.concatenate(median_segments)
        total_times = deterministic.sum() + worlds.lap_noise.sum(axis=1)
        degradation_adjustment = np.zeros_like(total_times)
        for stint in strategy.stints:
            sampled_total = worlds.degradation_cumulative[stint.compound.upper()][:, stint.length - 1]
            median_total = float(worlds.quantiles[stint.compound.upper()][:stint.length, 1].sum())
            degradation_adjustment += sampled_total - median_total
        total_times += degradation_adjustment

        sc_benefit_flags = np.zeros(len(total_times), dtype=bool)
        for stop_index, pit_lap in enumerate(strategy.pit_laps):
            pit_index = pit_lap - 1
            under_neutral = worlds.neutral[:, pit_index]
            sc_benefit_flags |= under_neutral
            total_times += worlds.pit_transit[:, pit_index] + worlds.dwell[stop_index] + worlds.slow_stop[stop_index]

        return SimulationResult(
            strategy=strategy,
            mean_time=float(np.mean(total_times)),
            median_time=float(np.median(total_times)),
            std_time=float(np.std(total_times)),
            q10_time=float(np.percentile(total_times, 10)),
            q90_time=float(np.percentile(total_times, 90)),
            sc_benefit_pct=float(np.mean(sc_benefit_flags) * 100.0),
            best_time=float(np.min(total_times)),
            worst_time=float(np.max(total_times)),
            lap_pace_deterministic=deterministic,
        )

    def evaluate_strategy(
        self,
        profile: CircuitProfile,
        strategy: Strategy,
        n_sims: int = 1000,
        rng: Optional[np.random.Generator] = None,
        max_stops: int = 2,
        driver: str = "UNK",
        team: str = "UNK",
    ) -> SimulationResult:
        """Evaluate one candidate; pass the same seed to compare matched scenarios."""
        rng = rng or np.random.default_rng(42)
        worlds = self._prepare_worlds(profile, n_sims, rng, max_stops, driver, team)
        return self._evaluate_with_worlds(profile, strategy, worlds)

    def _materialize_lap_samples(
        self, profile: CircuitProfile, strategy: Strategy, worlds: _Worlds
    ) -> np.ndarray:
        deg_segments = [
            worlds.degradation_samples[stint.compound.upper()][:, :stint.length]
            for stint in strategy.stints
        ]
        degradation = np.concatenate(deg_segments, axis=1)
        lap_samples = self._base_lap_vector(profile)[None, :] + degradation + worlds.lap_noise
        for stop_index, pit_lap in enumerate(strategy.pit_laps):
            idx = pit_lap - 1
            lap_samples[:, idx] += (
                worlds.pit_transit[:, idx] + worlds.dwell[stop_index] + worlds.slow_stop[stop_index]
            )
        return lap_samples

    @staticmethod
    def _stint_trajectory_details(strategy: Strategy, worlds: _Worlds) -> List[Dict[str, Any]]:
        details = []
        for index, stint in enumerate(strategy.stints, 1):
            q = worlds.quantiles[stint.compound.upper()][:stint.length]
            details.append({
                "stint_number": index,
                "compound": stint.compound.upper(),
                "start_lap": stint.start_lap,
                "end_lap": stint.end_lap,
                "age": np.arange(1, stint.length + 1),
                "q10": q[:, 0].copy(),
                "q50": q[:, 1].copy(),
                "q90": q[:, 2].copy(),
            })
        return details

    def optimize(
        self,
        profile: CircuitProfile,
        max_stops: int = 2,
        n_sims: int = 1000,
        top_k: int = 3,
        rng: Optional[np.random.Generator] = None,
        driver: str = "UNK",
        team: str = "UNK",
    ) -> List[Dict[str, Any]]:
        """Rank strategies with shared worlds; Plan B/C are alternate contingencies."""
        if n_sims < 1:
            raise ValueError("n_sims must be positive.")
        rng = rng or np.random.default_rng(42)
        candidates = self.generate_candidate_strategies(profile.total_laps, max_stops=max_stops)
        if not candidates:
            raise ValueError(f"No legal strategies available for a {profile.total_laps}-lap race.")
        worlds = self._prepare_worlds(profile, n_sims, rng, max_stops, driver, team)
        results = [self._evaluate_with_worlds(profile, strategy, worlds) for strategy in candidates]
        results.sort(key=lambda result: result.mean_time)
        plan_a = results[0]

        # Plan B: an alternate compound sequence or an earlier undercut with the same stop count.
        plan_b_pool = [
            result for result in results[1:]
            if result.strategy.total_stops == plan_a.strategy.total_stops
            and (
                [s.compound for s in result.strategy.stints] != [s.compound for s in plan_a.strategy.stints]
                or (result.strategy.pit_laps and plan_a.strategy.pit_laps
                    and result.strategy.pit_laps[0] < plan_a.strategy.pit_laps[0])
            )
        ]
        plan_b = plan_b_pool[0] if plan_b_pool else next((r for r in results[1:] if r is not plan_a), None)

        # Plan C: favor a later overcut / two-stop strategy with a neutralization opportunity.
        selected = {id(plan_a)}
        if plan_b is not None:
            selected.add(id(plan_b))
        tolerance = max(15.0, plan_a.mean_time * 0.003)
        plan_c_pool = [
            result for result in results[1:]
            if id(result) not in selected
            and result.mean_time <= plan_a.mean_time + tolerance
            and (result.strategy.total_stops == 2
                 or (result.strategy.pit_laps and plan_a.strategy.pit_laps
                     and result.strategy.pit_laps[-1] > plan_a.strategy.pit_laps[-1]))
        ]
        if not plan_c_pool:
            plan_c_pool = [result for result in results[1:] if id(result) not in selected]
        plan_c = max(
            plan_c_pool,
            key=lambda result: (
                result.sc_benefit_pct,
                result.strategy.pit_laps[-1] if result.strategy.pit_laps else 0,
                -result.mean_time,
            ),
            default=None,
        )
        chosen: List[tuple[str, SimulationResult]] = [
            ("Plan A · Baseline optimum", plan_a),
            ("Plan B · Undercut / alternate compound", plan_b),
            ("Plan C · SC contingency / extended overcut", plan_c),
        ]
        if len([result for _, result in chosen if result is not None]) < min(3, top_k):
            used = {id(result) for _, result in chosen if result is not None}
            for result in results:
                if id(result) not in used:
                    chosen.append(("Alternate strategy", result))
                    used.add(id(result))
                    if len([item for item in chosen if item[1] is not None]) >= top_k:
                        break

        output: List[Dict[str, Any]] = []
        for label, result in chosen:
            if result is None:
                continue
            if len(output) >= top_k:
                break
            result.lap_time_samples = self._materialize_lap_samples(profile, result.strategy, worlds)
            result.stint_trajectories = self._stint_trajectory_details(result.strategy, worlds)
            output.append({
                "label": label,
                "strategy": result.strategy,
                "strategy_summary": result.strategy.summary(),
                "full_name": result.strategy.full_name(),
                "stops": result.strategy.total_stops,
                "mean_time": result.mean_time,
                "delta_to_best": result.mean_time - plan_a.mean_time,
                "median_time": result.median_time,
                "std_time": result.std_time,
                "q10_time": result.q10_time,
                "q90_time": result.q90_time,
                "sc_benefit_pct": result.sc_benefit_pct,
                "has_cliff_warning": result.strategy.has_cliff_warning,
                "lap_pace": result.lap_pace_deterministic,
                "lap_time_samples": result.lap_time_samples,
                "stint_trajectories": result.stint_trajectories,
                "result": result,
            })
        return output


def _cli_demo() -> None:
    import time
    profile = CircuitProfile.from_name("Bahrain")
    simulator = StrategySimulator()
    started = time.perf_counter()
    plans = simulator.optimize(profile, max_stops=2, n_sims=1000, top_k=3)
    elapsed = time.perf_counter() - started
    print(f"StintIQ Model 3 · {profile.name} · {profile.total_laps} laps · 1,000 simulations")
    print(f"Completed in {elapsed:.3f}s")
    for plan in plans:
        print(
            f"{plan['label']}: {plan['full_name']} | "
            f"mean {plan['mean_time']:.1f}s | q90 {plan['q90_time']:.1f}s | "
            f"σ {plan['std_time']:.2f}s | SC pit opportunity {plan['sc_benefit_pct']:.1f}% | "
            f"{'cliff warning' if plan['has_cliff_warning'] else 'within tire limits'}"
        )


if __name__ == "__main__":
    _cli_demo()
