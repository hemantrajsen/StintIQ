"""Five-event 2026 historical validation for StintIQ Model 3.

Reads the raw lap snapshot and Model 1 pace-delta table. Final classifications
come from the existing Model 5 result-label export when available, with the raw
Position column used as a fallback. Actual stint choices always come from raw laps.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from model3_strategy_simulator import (
    CLIFF_PRIORS,
    CircuitProfile,
    Stint,
    Strategy,
    StrategySimulator,
    driver_pace_delta,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_PATH = PROJECT_ROOT / "lap_data_raw.csv"
DEFAULT_MODEL_READY_PATH = PROJECT_ROOT / "lap_data_model_ready.csv"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "model5_outputs" / "model5_2026_race_features_and_labels.csv"
DEFAULT_EVENTS = ("Barcelona", "Monaco", "Azerbaijan", "Austria", "Great Britain")
EVENT_ALIASES = {
    "australian": "australia", "chinese": "china", "japanese": "japan",
    "canadian": "canada", "austrian": "austria", "british": "greatbritain",
    "belgian": "belgium", "hungarian": "hungary", "dutch": "netherlands",
    "italian": "italy", "spanish": "spain",
}
DRY_COMPOUNDS = set(CLIFF_PRIORS)


def normalize_event_name(value: object) -> str:
    text = re.sub(r"^2026[_\s-]*", "", str(value).strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s+Grand Prix$", "", text, flags=re.IGNORECASE)
    token = re.sub(r"[^a-z0-9]", "", text.casefold())
    return EVENT_ALIASES.get(token, token)


def _raw_event_rows(raw: pd.DataFrame, circuit: str) -> Tuple[str, pd.DataFrame]:
    if "EventName" not in raw:
        raise ValueError("lap_data_raw.csv must contain EventName.")
    event_names = raw["EventName"].dropna().astype(str).unique()
    matches = [name for name in event_names if normalize_event_name(name) == normalize_event_name(circuit)]
    if not matches:
        raise ValueError(f"No raw lap event matches {circuit!r}.")
    event_name = matches[0]
    rows = raw.loc[raw["EventName"].astype(str).eq(event_name)].copy()
    return event_name, rows


def _positions_from_raw(event_rows: pd.DataFrame) -> Optional[pd.DataFrame]:
    if not {"Driver", "Position"}.issubset(event_rows.columns):
        return None
    work = event_rows.copy()
    work["Position"] = pd.to_numeric(work["Position"], errors="coerce")
    if "LapNumber" in work:
        work["LapNumber"] = pd.to_numeric(work["LapNumber"], errors="coerce")
        work = work.sort_values("LapNumber")
    positions = (
        work.dropna(subset=["Position"])
        .groupby("Driver", as_index=False)
        .tail(1)[["Driver", "Position"]]
        .rename(columns={"Position": "FinishPosition"})
    )
    positions = positions.loc[positions["FinishPosition"].gt(0)]
    if positions["FinishPosition"].nunique() < 3 or not positions["FinishPosition"].eq(1).any():
        return None
    return positions.sort_values("FinishPosition").reset_index(drop=True)


def _positions_from_results(results: Optional[pd.DataFrame], circuit: str) -> Optional[pd.DataFrame]:
    if results is None or not {"EventName", "Driver", "FinishPosition"}.issubset(results.columns):
        return None
    normalized = results["EventName"].map(normalize_event_name)
    subset = results.loc[normalized.eq(normalize_event_name(circuit)), ["Driver", "FinishPosition"]].copy()
    subset["FinishPosition"] = pd.to_numeric(subset["FinishPosition"], errors="coerce")
    subset = subset.dropna(subset=["Driver", "FinishPosition"])
    subset = subset.sort_values("FinishPosition").drop_duplicates("Driver")
    if subset["FinishPosition"].nunique() < 3 or not subset["FinishPosition"].eq(1).any():
        return None
    return subset.reset_index(drop=True)


def actual_finishers(
    event_rows: pd.DataFrame, results: Optional[pd.DataFrame], circuit: str
) -> Tuple[pd.DataFrame, str]:
    positions = _positions_from_results(results, circuit)
    if positions is not None:
        return positions, "Model 5 final-result labels"
    positions = _positions_from_raw(event_rows)
    if positions is not None:
        return positions, "lap_data_raw.Position fallback"
    raise ValueError(
        f"No classified top-three results are available for {circuit}. "
        "Populate Position in lap_data_raw.csv or supply the Model 5 results export."
    )


def actual_driver_strategy(event_rows: pd.DataFrame, driver: str) -> Strategy:
    required = {"Driver", "Stint", "Compound", "LapNumber"}
    if not required.issubset(event_rows.columns):
        raise ValueError(f"Raw lap rows lack required strategy columns: {sorted(required - set(event_rows.columns))}")
    rows = event_rows.loc[event_rows["Driver"].astype(str).eq(str(driver))].copy()
    rows["LapNumber"] = pd.to_numeric(rows["LapNumber"], errors="coerce")
    rows["Stint"] = pd.to_numeric(rows["Stint"], errors="coerce")
    rows = rows.dropna(subset=["LapNumber", "Stint", "Compound"])
    if rows.empty:
        raise ValueError(f"No stint records found for {driver}.")
    stints = []
    for _, stint_rows in rows.groupby("Stint", sort=True):
        compound = str(stint_rows["Compound"].dropna().iloc[0]).upper()
        start_lap = int(stint_rows["LapNumber"].min())
        end_lap = int(stint_rows["LapNumber"].max())
        stints.append(Stint(compound, start_lap, end_lap))
    stints.sort(key=lambda stint: stint.start_lap)
    return Strategy(stints)


def _pit_window_metrics(predicted: Sequence[int], actual: Sequence[int]) -> Tuple[float, Optional[int]]:
    if not actual:
        return (1.0 if not predicted else 0.0), None
    remaining = list(predicted)
    matched = 0
    deltas = []
    for actual_lap in actual:
        if not remaining:
            continue
        chosen = min(remaining, key=lambda lap: abs(lap - actual_lap))
        remaining.remove(chosen)
        delta = abs(chosen - actual_lap)
        deltas.append(delta)
        matched += int(delta <= 3)
    return matched / len(actual), min(deltas) if deltas else None


def _strategy_string(strategy: Strategy) -> str:
    return " → ".join(
        f"{stint.compound[0]} ({stint.start_lap}–{stint.end_lap})"
        for stint in strategy.stints
    )


def run_backtest(
    raw_path: Path = DEFAULT_RAW_PATH,
    model_ready_path: Path = DEFAULT_MODEL_READY_PATH,
    results_path: Path = DEFAULT_RESULTS_PATH,
    events: Sequence[str] = DEFAULT_EVENTS,
    n_sims: int = 1000,
    simulator: Optional[StrategySimulator] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run the fixed five-event holdout and return event rows plus aggregate metrics."""
    unique_events = list(dict.fromkeys(events))
    if len(unique_events) != 5:
        raise ValueError(f"The roadmap backtest requires five distinct events; received {len(unique_events)}.")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw lap dataset not found: {raw_path}")
    raw = pd.read_csv(raw_path, low_memory=False)
    model_ready = pd.read_csv(model_ready_path, low_memory=False) if model_ready_path.exists() else pd.DataFrame()
    results = pd.read_csv(results_path, low_memory=False) if results_path.exists() else None
    engine = simulator or StrategySimulator()

    rows: List[Dict[str, object]] = []
    for event_index, circuit in enumerate(unique_events):
        event_name, event_rows = _raw_event_rows(raw, circuit)
        finishers, result_source = actual_finishers(event_rows, results, circuit)
        podium = finishers.sort_values("FinishPosition").head(3)
        winner = podium.iloc[0]
        winner_driver = str(winner["Driver"])
        winner_team = None
        if "Team" in event_rows:
            team_values = event_rows.loc[event_rows["Driver"].astype(str).eq(winner_driver), "Team"].dropna()
            if not team_values.empty:
                winner_team = str(team_values.mode().iloc[0])

        profile = CircuitProfile.from_raw_data(event_rows, circuit)
        profile.driver_delta_seconds = driver_pace_delta(
            model_ready, winner_driver, team=winner_team, exclude_event=event_name
        )
        plan_a = engine.optimize(
            profile,
            max_stops=2,
            n_sims=n_sims,
            top_k=1,
            rng=np.random.default_rng(2026 + event_index),
            driver=winner_driver,
            team=winner_team or "UNK",
        )[0]

        podium_strategies = []
        for _, finisher in podium.iterrows():
            strategy = actual_driver_strategy(event_rows, str(finisher["Driver"]))
            podium_strategies.append(
                f"P{int(finisher['FinishPosition'])} {finisher['Driver']}: {_strategy_string(strategy)}"
            )

        actual = actual_driver_strategy(event_rows, winner_driver)
        actual_model_result = None
        regret = np.nan
        if (
            all(stint.compound in DRY_COMPOUNDS for stint in actual.stints)
            and actual.is_fia_legal
            and sum(stint.length for stint in actual.stints) == profile.total_laps
        ):
            # Replay the executed strategy under the same seeded race scenarios as Plan A.
            actual_model_result = engine.evaluate_strategy(
                profile,
                actual,
                n_sims=n_sims,
                rng=np.random.default_rng(2026 + event_index),
                max_stops=2,
                driver=winner_driver,
                team=winner_team or "UNK",
            )
            regret = actual_model_result.mean_time - plan_a["mean_time"]

        pit_overlap, min_pit_delta = _pit_window_metrics(plan_a["strategy"].pit_laps, actual.pit_laps)
        predicted_stints = plan_a["strategy"].stints
        rows.append({
            "Event": event_name,
            "Circuit": circuit,
            "Result source": result_source,
            "Actual winner": winner_driver,
            "Winner actual strategy": _strategy_string(actual),
            "Podium actual strategies": " | ".join(podium_strategies),
            "Model 3 Plan A": plan_a["strategy"].summary(),
            "Actual stops": actual.total_stops,
            "Plan A stops": plan_a["stops"],
            "Stop count match": actual.total_stops == plan_a["stops"],
            "Start compound match": predicted_stints[0].compound == actual.stints[0].compound,
            "Finish compound match": predicted_stints[-1].compound == actual.stints[-1].compound,
            "Pit window overlap ±3": pit_overlap,
            "Closest pit delta (laps)": min_pit_delta,
            "Plan A expected time (s)": plan_a["mean_time"],
            "Executed strategy model time (s)": actual_model_result.mean_time if actual_model_result else np.nan,
            "Model-implied regret (s)": regret,
            "Cliff warning": plan_a["has_cliff_warning"],
        })

    report = pd.DataFrame(rows)
    if len(report) != 5 or report["Event"].nunique() != 5:
        raise RuntimeError("Backtest did not produce five distinct event results.")
    metrics = {
        "events": float(len(report)),
        "stop_count_accuracy_pct": float(report["Stop count match"].mean() * 100.0),
        "start_compound_match_pct": float(report["Start compound match"].mean() * 100.0),
        "finish_compound_match_pct": float(report["Finish compound match"].mean() * 100.0),
        "pit_window_overlap_pct": float(report["Pit window overlap ±3"].mean() * 100.0),
        "mean_model_implied_regret_s": float(report["Model-implied regret (s)"].mean()),
        "mean_closest_pit_delta_laps": float(report["Closest pit delta (laps)"].dropna().mean()),
    }
    return report, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the StintIQ Model 3 five-race 2026 backtest.")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH, help="Path to lap_data_raw.csv")
    parser.add_argument("--model-ready", type=Path, default=DEFAULT_MODEL_READY_PATH, help="Path to lap_data_model_ready.csv")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH, help="Optional Model 5 race-result labels")
    parser.add_argument("--events", nargs=5, default=list(DEFAULT_EVENTS), help="Exactly five circuit names")
    parser.add_argument("--n-sims", type=int, default=1000, help="Monte Carlo runs per strategy (default: 1,000)")
    args = parser.parse_args()

    report, metrics = run_backtest(args.raw, args.model_ready, args.results, args.events, args.n_sims)
    display_columns = [
        "Event", "Actual winner", "Winner actual strategy", "Model 3 Plan A",
        "Actual stops", "Plan A stops", "Closest pit delta (laps)", "Model-implied regret (s)",
    ]
    print("STINTIQ MODEL 3 — FIVE-EVENT HISTORICAL BACKTEST")
    print(report[display_columns].to_string(index=False, formatters={
        "Model-implied regret (s)": lambda value: f"{value:+.2f}" if pd.notna(value) else "n/a",
        "Closest pit delta (laps)": lambda value: f"{value:.0f}" if pd.notna(value) else "n/a",
    }))
    print("\nPODIUM STRATEGIES")
    for _, row in report.iterrows():
        print(f"{row['Event']}: {row['Podium actual strategies']}")
    print("\nSUMMARY")
    print(f"Stop count accuracy: {metrics['stop_count_accuracy_pct']:.1f}%")
    print(f"Start compound match: {metrics['start_compound_match_pct']:.1f}%")
    print(f"Finish compound match: {metrics['finish_compound_match_pct']:.1f}%")
    print(f"Pit window overlap within ±3 laps: {metrics['pit_window_overlap_pct']:.1f}%")
    print(f"Mean model-implied hindsight regret: {metrics['mean_model_implied_regret_s']:+.2f}s")
    print(f"Mean closest pit-lap difference: {metrics['mean_closest_pit_delta_laps']:.2f} laps")
    print("\nResult-source note: final classifications use the existing Model 5 labels when available; raw Position is the fallback. Strategies come from raw laps.")
    print("Regret compares both strategies under matched Model 3 Monte Carlo scenarios; it is not a measured real-world time delta.")


if __name__ == "__main__":
    main()
