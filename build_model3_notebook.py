"""Build the roadmap-aligned Model 3 notebook from the production simulator."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap

PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_ROOT / "Model Development" / "model3_strategy_sim.ipynb"
cells: list[dict] = []


def add_markdown(source: str) -> None:
    source = textwrap.dedent(source).strip()
    cells.append({
        "cell_type": "markdown",
        "id": f"m3-{len(cells) + 1:03d}",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    })


def add_code(source: str) -> None:
    source = textwrap.dedent(source).strip()
    cells.append({
        "cell_type": "code",
        "id": f"m3-{len(cells) + 1:03d}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    })


add_markdown("""
# StintIQ Model 3 — Strategy Simulator

This notebook is the analysis companion to the roadmap implementation. It imports the same simulator used by the Streamlit Strategy Simulator and runs the same five-event backtest. Core logic lives in `model3_strategy_simulator.py`; this notebook does not maintain a second copy of the model.

**Use:** explore race conditions, compare three dry strategies, inspect uncertainty, and review historical validation. Results are a research preview, not live pit-wall advice.

**Clean run flow:** from the project root, install `requirements.txt`, then launch the app with `streamlit run app.py`. To use this notebook, open it with the project virtual environment as its kernel and run the cells from top to bottom.
""")

add_markdown("""
## Roadmap coverage

| Roadmap need | Notebook section |
|---|---|
| All 17 circuit defaults, a selected driver, stop limit, track temperature, and 500–2,000 simulations | Inputs and run |
| Model 1 pace delta, Model 2 tire quantiles and cliffs, Model 4 neutralization risk | Data and bridge status |
| Plan A, alternate/undercut Plan B, and contingency/overcut Plan C | Strategy results |
| Stint timeline, cumulative time-delta fan, and degradation curves | Three interactive charts |
| Five-race results, podium strategies, stop/compound/pit-window matches, and model-implied regret | Historical backtest |

A fallback is shown when a trained upstream artifact cannot be loaded. The backtest uses Model 5 final-result labels when available and extracts actual strategies from raw lap records.
""")

add_markdown("""
## 1. Environment and project data

Run the notebook from the project or its `Model Development` folder. The project root is located from the simulator and raw lap-data files.
""")

add_code("""
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = next(
    (path for path in (Path.cwd(), *Path.cwd().parents)
     if (path / "model3_strategy_simulator.py").exists()),
    None,
)
if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Could not find model3_strategy_simulator.py above the notebook directory."
    )
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model3_strategy_simulator import (
    CIRCUIT_DEFAULTS,
    CLIFF_PRIORS,
    CircuitProfile,
    StrategySimulator,
    driver_pace_delta,
)

RAW_PATH = PROJECT_ROOT / "lap_data_raw.csv"
MODEL_READY_PATH = PROJECT_ROOT / "lap_data_model_ready.csv"
MODEL2_PATH = PROJECT_ROOT / "Model Development" / "stintiq_model2_catboost.cbm"
MODEL4_PATH = PROJECT_ROOT / "Model Development" / "models" / "model4_sc_vsc.joblib"
RESULTS_PATH = PROJECT_ROOT / "model5_outputs" / "model5_2026_race_features_and_labels.csv"

raw_df = pd.read_csv(RAW_PATH, low_memory=False) if RAW_PATH.exists() else None
model_ready_df = (
    pd.read_csv(MODEL_READY_PATH, low_memory=False)
    if MODEL_READY_PATH.exists() else pd.DataFrame()
)
results_df = pd.read_csv(RESULTS_PATH, low_memory=False) if RESULTS_PATH.exists() else None

inventory = pd.DataFrame([
    {"Input": "Raw lap data", "Path": str(RAW_PATH), "Status": f"{len(raw_df):,} rows" if raw_df is not None else "Not found"},
    {"Input": "Model 1 pace table", "Path": str(MODEL_READY_PATH), "Status": f"{len(model_ready_df):,} rows" if not model_ready_df.empty else "Not found"},
    {"Input": "Model 2 CatBoost artifact", "Path": str(MODEL2_PATH), "Status": "Present" if MODEL2_PATH.exists() else "Not found; fallback will be used"},
    {"Input": "Model 4 hazard artifact", "Path": str(MODEL4_PATH), "Status": "Present" if MODEL4_PATH.exists() else "Not found; fallback will be used"},
    {"Input": "Model 5 result labels", "Path": str(RESULTS_PATH), "Status": f"{len(results_df):,} rows" if results_df is not None else "Not found; raw positions are fallback"},
])
print(f"Project: {PROJECT_ROOT}")
print(f"Circuit defaults available: {len(CIRCUIT_DEFAULTS)}")
inventory
""")

add_markdown("""
## 2. Upstream model bridge status

Model 1 contributes a driver pace delta from the event-balanced lap-time-delta table. Model 2 provides q10/q50/q90 tire-degradation curves and empirical cliff thresholds. Model 4 supplies circuit/weather-aware neutralization scenarios when its artifact can be loaded; otherwise the specified constant-hazard fallback is used.
""")

add_code("""
simulator = StrategySimulator()
bridge_status = pd.DataFrame([
    {"Bridge": "Model 1 · pace", "Status": "Historical driver/team pace delta" if not model_ready_df.empty else "Zero-delta fallback"},
    {"Bridge": "Model 2 · tire degradation", "Status": "CatBoost loaded" if simulator.deg_bridge.has_catboost else "Parametric q10/q50/q90 fallback"},
    {"Bridge": "Model 4 · SC/VSC hazard", "Status": "Saved hazard model loaded" if simulator.hazard_bridge.has_joblib else "Constant-hazard fallback"},
])
bridge_status
""")

add_markdown("""
## 3. Choose a simulation

Edit the values in the next cell, then run the following cells in order. The optimizer evaluates dry one-stop and two-stop options (within the selected maximum), uses common Monte Carlo scenarios to compare candidates, and enforces two distinct dry compounds.
""")

add_code("""
CIRCUIT = "Bahrain"       # Any of the 17 entries in CIRCUIT_DEFAULTS
DRIVER = "NOR"            # A driver in the project data; otherwise the benchmark falls back
MAX_STOPS = 2              # 1 or 2
N_SIMS = 1000              # Roadmap range: 500–2,000
TRACK_TEMP_OVERRIDE = None # Celsius; None uses the event/default value
RANDOM_SEED = 42

assert CIRCUIT in CIRCUIT_DEFAULTS, f"Choose one of: {list(CIRCUIT_DEFAULTS)}"
assert MAX_STOPS in (1, 2)
assert 500 <= N_SIMS <= 2000
""")

add_code("""
profile_source = raw_df if raw_df is not None else RAW_PATH
profile = CircuitProfile.from_raw_data(profile_source, CIRCUIT)

pace_source = model_ready_df if not model_ready_df.empty else raw_df
team = None
if pace_source is not None and {"Driver", "Team"}.issubset(pace_source.columns):
    driver_rows = pace_source.loc[
        pace_source["Driver"].astype(str).str.upper().eq(DRIVER.upper()), "Team"
    ].dropna()
    if not driver_rows.empty:
        team = str(driver_rows.mode().iloc[0])

if TRACK_TEMP_OVERRIDE is not None:
    profile.track_temp = float(TRACK_TEMP_OVERRIDE)
profile.driver_delta_seconds = driver_pace_delta(
    model_ready_df if not model_ready_df.empty else pd.DataFrame(),
    DRIVER,
    team=team,
)

input_summary = pd.DataFrame([
    {"Setting": "Circuit", "Value": profile.name},
    {"Setting": "Race laps", "Value": profile.total_laps},
    {"Setting": "Driver", "Value": DRIVER},
    {"Setting": "Team benchmark", "Value": team or "Unavailable"},
    {"Setting": "Driver pace delta", "Value": f"{profile.driver_delta_seconds:+.3f} s/lap"},
    {"Setting": "Reference lap", "Value": f"{profile.base_lap_time:.3f} s"},
    {"Setting": "Track temperature", "Value": f"{profile.track_temp:.1f} °C"},
    {"Setting": "Maximum stops / simulations", "Value": f"{MAX_STOPS} / {N_SIMS:,}"},
])
input_summary
""")

add_code("""
started = time.perf_counter()
plans = simulator.optimize(
    profile,
    max_stops=MAX_STOPS,
    n_sims=N_SIMS,
    top_k=3,
    rng=np.random.default_rng(RANDOM_SEED),
    driver=DRIVER,
    team=team or "UNK",
)
simulation_seconds = time.perf_counter() - started

plan_table = pd.DataFrame([
    {
        "Plan": plan["label"],
        "Strategy": plan["strategy_summary"],
        "Stops": plan["stops"],
        "Expected time (s)": plan["mean_time"],
        "Downside q90 (s)": plan["q90_time"],
        "Race-time σ (s)": plan["std_time"],
        "SC/VSC pit opportunity (%)": plan["sc_benefit_pct"],
        "Cliff warning": plan["has_cliff_warning"],
    }
    for plan in plans
]).round(2)
print(f"Completed {N_SIMS:,} shared scenarios in {simulation_seconds:.3f} seconds.")
plan_table
""")

add_markdown("""
## 4. Interactive strategy charts

The timeline marks the empirical cliff point inside any stint that exceeds its compound threshold. The fan chart uses matched simulations, so each alternative is compared against Plan A under the same lap-noise, tire, pit-stop, and neutralization scenarios.
""")

add_code("""
COMPOUND_COLORS = {"SOFT": "#E53935", "MEDIUM": "#E5A600", "HARD": "#7A8797"}
PLAN_COLORS = ["#E10600", "#3766A5", "#238A78"]

fig_gantt = go.Figure()
seen_compounds = set()
for plan_index, plan in enumerate(plans):
    for stint in plan["strategy"].stints:
        compound = stint.compound.upper()
        fig_gantt.add_trace(go.Bar(
            x=[stint.length],
            y=[plan["label"]],
            base=[stint.start_lap - 1],
            orientation="h",
            name=compound,
            legendgroup=compound,
            showlegend=compound not in seen_compounds,
            marker_color=COMPOUND_COLORS.get(compound, "#697386"),
            text=[f"{compound} · {stint.length} laps"],
            textposition="inside",
            hovertemplate=(f"{compound}<br>Lap {stint.start_lap}–{stint.end_lap}"
                           f"<br>Stint length: {stint.length}<extra></extra>"),
        ))
        seen_compounds.add(compound)
        if stint.exceeds_cliff:
            cliff_lap = stint.start_lap + stint.cliff_lap - 1
            fig_gantt.add_trace(go.Scatter(
                x=[cliff_lap], y=[plan["label"]], mode="markers",
                marker={"symbol": "x", "size": 11, "color": "#B42318"},
                name="Cliff threshold exceeded", showlegend=False,
                hovertemplate=f"{compound} cliff threshold · lap {cliff_lap}<extra></extra>",
            ))
fig_gantt.update_layout(
    title=f"{profile.name} · recommended stint timelines",
    xaxis_title="Race lap",
    yaxis_title="Strategy",
    barmode="overlay",
    height=390,
    legend_title="Dry compound",
)
fig_gantt.update_xaxes(range=[0, profile.total_laps], dtick=max(5, profile.total_laps // 10))
fig_gantt
""")

add_code("""
plan_a_cumulative = np.cumsum(plans[0]["lap_time_samples"], axis=1)
laps = np.arange(1, profile.total_laps + 1)
fig_fan = go.Figure()

for plan_index, plan in enumerate(plans):
    cumulative = np.cumsum(plan["lap_time_samples"], axis=1)
    delta = cumulative - plan_a_cumulative
    q10, median, q90 = np.percentile(delta, [10, 50, 90], axis=0)
    color = PLAN_COLORS[plan_index % len(PLAN_COLORS)]
    if plan_index > 0:
        fig_fan.add_trace(go.Scatter(
            x=laps, y=q10, mode="lines", line={"width": 0},
            showlegend=False, hoverinfo="skip",
        ))
        fig_fan.add_trace(go.Scatter(
            x=laps, y=q90, mode="lines", line={"width": 0},
            fill="tonexty", fillcolor=(
                f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.14)"
            ),
            name=f"{plan['label']} · q10–q90", hoverinfo="skip",
        ))
    fig_fan.add_trace(go.Scatter(
        x=laps, y=median, mode="lines",
        line={"color": color, "width": 2.5},
        name=f"{plan['label']} · median",
        hovertemplate="Lap %{x}<br>Time delta: %{y:.1f}s<extra></extra>",
    ))
fig_fan.add_hline(y=0, line_dash="dash", line_color="#697386")
fig_fan.update_layout(
    title="Cumulative race-time delta versus Plan A",
    xaxis_title="Race lap",
    yaxis_title="Cumulative time delta (s)",
    height=430,
)
fig_fan
""")

add_code("""
def as_rgba(hex_color: str, alpha: float) -> str:
    rgb = tuple(int(hex_color[index:index + 2], 16) for index in (1, 3, 5))
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"

fig_tires = go.Figure()
for stint_index, stint in enumerate(plans[0]["stint_trajectories"], start=1):
    compound = stint["compound"]
    color = COMPOUND_COLORS.get(compound, "#697386")
    age = stint["age"]
    fig_tires.add_trace(go.Scatter(
        x=age, y=stint["q10"], mode="lines", line={"width": 0},
        showlegend=False, hoverinfo="skip",
    ))
    fig_tires.add_trace(go.Scatter(
        x=age, y=stint["q90"], mode="lines", line={"width": 0},
        fill="tonexty", fillcolor=as_rgba(color, 0.16),
        showlegend=False, hoverinfo="skip",
    ))
    fig_tires.add_trace(go.Scatter(
        x=age, y=stint["q50"], mode="lines+markers",
        line={"color": color, "width": 2.5}, marker={"size": 4},
        name=f"Stint {stint_index} · {compound} q50",
        customdata=np.column_stack([stint["q10"], stint["q90"]]),
        hovertemplate=("Tire age %{x}<br>Median degradation %{y:.2f}s"
                       "<br>q10 %{customdata[0]:.2f}s · q90 %{customdata[1]:.2f}s<extra></extra>"),
    ))
fig_tires.update_layout(
    title="Plan A · Model 2 stint degradation pace curves",
    xaxis_title="Tire age (laps on compound)",
    yaxis_title="Pace loss from stint baseline (s/lap)",
    height=430,
    legend_title="Stint trajectory",
)
fig_tires
""")

add_markdown("""
## 5. Five-event historical backtest

The fixed event set spans different circuit types. The report includes actual podium strategies, winner stop and compound matches, pit-window overlap within ±3 laps, and regret under matched Model 3 scenarios. “Model-implied regret” is a simulator counterfactual, not an observed timing-system loss.
""")

add_code("""
from model3_backtest import DEFAULT_EVENTS, run_backtest

backtest_report, backtest_metrics = run_backtest(
    raw_path=RAW_PATH,
    model_ready_path=MODEL_READY_PATH,
    results_path=RESULTS_PATH,
    events=DEFAULT_EVENTS,
    n_sims=N_SIMS,
)

backtest_view = backtest_report[[
    "Event", "Result source", "Actual winner", "Winner actual strategy",
    "Podium actual strategies", "Model 3 Plan A", "Actual stops", "Plan A stops",
    "Stop count match", "Start compound match", "Finish compound match",
    "Pit window overlap ±3", "Closest pit delta (laps)", "Model-implied regret (s)",
]].copy()
backtest_view["Pit window overlap ±3"] *= 100
backtest_view = backtest_view.rename(columns={"Pit window overlap ±3": "Pit overlap within ±3 (%)"})
backtest_view.round(2)
""")

add_code("""
backtest_summary = pd.DataFrame([
    {"Metric": "Stop-count accuracy", "Result": f"{backtest_metrics['stop_count_accuracy_pct']:.1f}%"},
    {"Metric": "Start-compound match", "Result": f"{backtest_metrics['start_compound_match_pct']:.1f}%"},
    {"Metric": "Finish-compound match", "Result": f"{backtest_metrics['finish_compound_match_pct']:.1f}%"},
    {"Metric": "Pit-window overlap within ±3 laps", "Result": f"{backtest_metrics['pit_window_overlap_pct']:.1f}%"},
    {"Metric": "Mean model-implied regret", "Result": f"{backtest_metrics['mean_model_implied_regret_s']:+.2f} s"},
    {"Metric": "Mean closest pit-lap difference", "Result": f"{backtest_metrics['mean_closest_pit_delta_laps']:.2f} laps"},
])
backtest_summary
""")

add_markdown("""
## 6. How to read the result

A successful run confirms the notebook is calling the shared Model 3 implementation; it does not establish that the strategy recommendations are accurate enough for operational use. Review the backtest matches and fallback status before interpreting an optimum. If the Model 2 CatBoost and Model 4 hazard artifacts are unavailable in the runtime, the fallback paths are active and their status is shown above.

The five-event report is a small diagnostic sample. Low agreement with actual strategies is a model-quality signal to investigate, not a reason to tune the model to these five outcomes. Keep this notebook and the app on the shared simulator module so future model changes are evaluated consistently.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14.3", "file_extension": ".py", "mimetype": "text/x-python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "codemirror_mode": {"name": "ipython", "version": 3}},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH} with {len(cells)} cells.")
