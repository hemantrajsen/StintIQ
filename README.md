
# StintIQ

**Real-Time F1 Telemetry & Predictive Race Strategy (WIP)**

StintIQ is an exploration project built on top of the **FastF1** Python API to pull Formula 1 session data, work with lap telemetry, and prototype the building blocks needed for **race strategy modeling** (stints, tyre behavior, pace, traffic, and pit windows).

Today, this repository focuses on:
- Loading F1 sessions via FastF1 (with caching)
- Extracting lap telemetry (speed/throttle/brake/gear)
- Visualizing telemetry and track maps (including speed heatmaps along the lap)

The “real-time” and “predictive strategy” components are the intended direction of the project and are not fully implemented yet.

## What’s in this repo

- `FastF1_basics.ipynb` — hands-on notebook showing FastF1 basics: caching, session loading, telemetry extraction, circuit maps, and speed-mapped track visualization.
- `load_data.py` — placeholder script (currently just imports) for future data-loading utilities.
- `cache/` — FastF1 cache directory (created/used to avoid repeated downloads).

## Setup

### Prerequisites
- Python 3.9+ recommended

### Install

```bash
pip install fastf1 pandas numpy matplotlib
```

FastF1 uses a local cache to speed up repeated analysis runs. This project expects a `cache/` directory in the repo root.

## Quickstart

### Run the notebook

1. Open `FastF1_basics.ipynb` in VS Code (or Jupyter).
2. Run the cells from top to bottom.

The notebook includes examples like:

```python
import fastf1

fastf1.Cache.enable_cache('cache')

session = fastf1.get_session(2026, 'Australia', 'R')
session.load(telemetry=True)

laps = session.laps.pick_driver('HAM')
fastest = laps.pick_fastest()
telemetry = fastest.get_telemetry()

telemetry[['Distance', 'Speed', 'Throttle', 'Brake', 'nGear']].head()
```

### Notes on seasons & sessions

FastF1 data availability depends on the session type and what the upstream sources provide.
- Typical session identifiers include: `FP1`, `FP2`, `FP3`, `Q`, `R`.
- If a specific event/year is unavailable, try a different year or session.

## Roadmap (project direction)

Planned milestones for “Predictive Race Strategy”:
- Build a clean dataset layer: lap-by-lap pace, tyre compound/age, pit events, safety car/VSC segments, weather
- Stint model: degradation curves, fuel correction, traffic effects
- Strategy simulator: evaluate pit windows, undercut/overcut scenarios, and “plan A/B/C” outcomes
- (Optional) Live-ish pipeline: periodic refresh during a session and near-real-time dashboards

## Disclaimer

This project is **not affiliated with Formula 1**. FastF1 is an unofficial API wrapper that relies on publicly available data sources.

