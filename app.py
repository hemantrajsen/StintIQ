"""StintIQ - an interactive, data-aware F1 analytics showcase."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# App setup and visual system
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "lap_data_raw.csv"
MODEL_PATH = ROOT / "lap_data_model_ready.csv"

RED = "#E10600"
INK = "#171923"
MUTED = "#697386"
GRID = "#E8EAF0"
COMPOUND_COLORS = {"SOFT": "#E53935", "MEDIUM": "#E5A600", "HARD": "#7A8797", "INTERMEDIATE": "#2E9B75", "WET": "#2878C8"}
TEAM_COLORS = {
    "Red Bull Racing": "#3671C6", "Ferrari": "#E8002D", "McLaren": "#FF8000",
    "Mercedes": "#27F4D2", "Aston Martin": "#229971", "Alpine": "#0093CC",
    "Williams": "#64C4FF", "Racing Bulls": "#6692FF", "Haas F1 Team": "#B6BABD",
    "Kick Sauber": "#52E252", "Audi": "#00A19B", "Cadillac": "#B7A36A",
}

st.set_page_config(
    page_title="StintIQ | F1 race analytics",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp { background: #F5F6F8; color: #171923; }
      [data-testid="stHeader"] { background: rgba(245,246,248,.9); }
      [data-testid="stSidebar"] { background: #171923; border-right: 1px solid #292C37; }
      [data-testid="stSidebar"] * { color: #F4F5F7; }
      [data-testid="stSidebar"] [data-testid="stRadio"] label { color: #D9DCE5; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #B8BDCA; }
      [data-testid="stSidebar"] hr { border-color: #343744; }
      [data-testid="stMetric"] { background: #FFFFFF; padding: 18px 20px; border: 1px solid #E7E9EF; border-radius: 14px; box-shadow: 0 3px 12px rgba(23,25,35,.035); }
      [data-testid="stMetricLabel"] { color: #697386; }
      [data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; color: #171923; }
      div[data-testid="stPlotlyChart"] { background: #FFFFFF; border: 1px solid #E7E9EF; border-radius: 14px; padding: 8px; }
      div[data-testid="stDataFrame"] { border: 1px solid #E7E9EF; border-radius: 12px; overflow: hidden; }
      .hero { background: linear-gradient(115deg, #171923 0%, #252936 65%, #333744 100%); color: #FFFFFF; border-radius: 20px; padding: 30px 34px; margin: 2px 0 22px; position: relative; overflow: hidden; }
      .hero:after { content: ''; position: absolute; right: -40px; top: -90px; width: 320px; height: 320px; border: 1px solid rgba(255,255,255,.10); border-radius: 50%; box-shadow: 0 0 0 28px rgba(255,255,255,.025), 0 0 0 58px rgba(255,255,255,.02); }
      .hero-kicker { color: #FF6B64; text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; font-weight: 700; margin-bottom: 9px; }
      .hero h1 { font-family: 'Space Grotesk', sans-serif; font-size: clamp(2rem, 4vw, 3rem); line-height: 1.05; margin: 0 0 10px; color: #FFFFFF; }
      .hero p { margin: 0; color: #C5C9D3; font-size: 1rem; max-width: 760px; }
      .hero-tag { display: inline-block; margin-top: 18px; padding: 6px 10px; border: 1px solid rgba(255,255,255,.22); border-radius: 99px; color: #E3E5EB; font-size: .72rem; letter-spacing: .06em; text-transform: uppercase; }
      .section-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.18rem; font-weight: 700; color: #171923; margin: 20px 0 4px; }
      .section-copy { color: #697386; font-size: .9rem; margin: 0 0 12px; }
      .surface { background: #FFFFFF; border: 1px solid #E7E9EF; border-radius: 14px; padding: 18px 20px; }
      .status-pill { display: inline-block; border-radius: 99px; padding: 5px 10px; font-weight: 700; font-size: .72rem; letter-spacing: .03em; }
      .status-active { background: #FFF0EF; color: #B51B14; }
      .status-explore { background: #FFF5D9; color: #8B6200; }
      .status-planned { background: #EEF1F5; color: #566174; }
      .muted { color: #697386; }
      .small-label { color: #8A91A0; text-transform: uppercase; letter-spacing: .1em; font-size: .68rem; font-weight: 700; }
      .stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data access: use files in this repository and tolerate missing snapshots.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _read_csv(path_string: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns  # Included in the cache key so a changed local file reloads.
    return pd.read_csv(path_string, low_memory=False)


def load_csv(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.exists():
        return None, None
    try:
        frame = _read_csv(str(path), path.stat().st_mtime_ns)
        return frame, None
    except Exception as exc:  # A broken optional snapshot should not hide the app.
        return None, f"Could not read {path.name}: {exc}"


def prepare_raw_laps(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None:
        return None
    result = frame.copy()
    if "LapTimeSeconds" in result.columns:
        result["LapTimeSeconds"] = pd.to_numeric(result["LapTimeSeconds"], errors="coerce")
    elif "LapTime" in result.columns:
        result["LapTimeSeconds"] = pd.to_timedelta(result["LapTime"], errors="coerce").dt.total_seconds()
    if "LapTimeSeconds" not in result.columns:
        return result

    result = result[result["LapTimeSeconds"].gt(0)].copy()
    if "IsAccurate" in result.columns:
        accurate = result["IsAccurate"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        result = result[accurate].copy()
    if "TrackStatus" in result.columns:
        result = result[result["TrackStatus"].astype(str).str.strip().eq("1")].copy()
    for pit_column in ("PitInTime", "PitOutTime"):
        if pit_column in result.columns:
            has_pit_time = result[pit_column].notna() & result[pit_column].astype(str).str.strip().ne("")
            result = result[~has_pit_time].copy()

    if {"EventName", "LapTimeSeconds"}.issubset(result.columns):
        reference = result.groupby("EventName")["LapTimeSeconds"].transform("min")
        result["lap_time_delta"] = result["LapTimeSeconds"] - reference
    return result


def numeric_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


raw_df, raw_error = load_csv(RAW_PATH)
model_df, model_error = load_csv(MODEL_PATH)
raw_laps = prepare_raw_laps(raw_df)

if model_df is not None:
    model_df = numeric_columns(model_df, ["TyreLife", "LapNumber", "Stint", "lap_time_delta", "EstimatedFuelKg", "AirTemp", "Humidity", "TimeGapAhead", "lag_1_delta", "lag_2_delta", "rolling_mean_3_delta"])
    if "EventName" in model_df.columns:
        model_df["EventName"] = model_df["EventName"].astype(str)


def display_event(value: Any) -> str:
    return str(value).replace("_", " ").strip()


def options_for(frame: pd.DataFrame | None, column: str) -> list[str]:
    if frame is None or column not in frame.columns:
        return []
    return sorted(frame[column].dropna().astype(str).unique().tolist())


def frame_for_exploration() -> pd.DataFrame | None:
    if model_df is not None and "lap_time_delta" in model_df.columns:
        return model_df
    if raw_laps is not None and "lap_time_delta" in raw_laps.columns:
        return raw_laps
    return None


def team_color_map(frame: pd.DataFrame, drivers: list[str]) -> dict[str, str]:
    colors: dict[str, str] = {}
    if "Team" not in frame.columns or "Driver" not in frame.columns:
        return colors
    for driver in drivers:
        team = frame.loc[frame["Driver"].astype(str).eq(driver), "Team"].dropna()
        if not team.empty:
            colors[driver] = TEAM_COLORS.get(str(team.iloc[0]), "#697386")
    return colors


def style_figure(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family="DM Sans, sans-serif", color=INK, size=12),
        title=dict(font=dict(family="Space Grotesk, sans-serif", size=16, color=INK)),
        margin=dict(l=24, r=18, t=54, b=28),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor=INK, font_color="white"),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def hero(kicker: str, title: str, subtitle: str, tag: str | None = None) -> None:
    tag_markup = f'<span class="hero-tag">{tag}</span>' if tag else ""
    st.markdown(
        f'<div class="hero"><div class="hero-kicker">{kicker}</div><h1>{title}</h1><p>{subtitle}</p>{tag_markup}</div>',
        unsafe_allow_html=True,
    )


def section(title: str, copy: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if copy:
        st.markdown(f'<div class="section-copy">{copy}</div>', unsafe_allow_html=True)


def show_data_notice() -> None:
    if raw_error or model_error:
        for message in (raw_error, model_error):
            if message:
                st.warning(message)
    if model_df is None and raw_laps is None:
        st.warning("No lap dataset was found. Add `lap_data_model_ready.csv` or `lap_data_raw.csv` beside `app.py` to enable the analysis pages.")


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div style='font-family:Space Grotesk,sans-serif;font-size:1.6rem;font-weight:700;color:white'>STINT<span style='color:#FF514A'>IQ</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='small-label' style='color:#858B99;margin:2px 0 16px'>Formula 1 race analytics</div>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Workspace",
        ["Overview", "Race Explorer", "Telemetry", "Stint Lab", "Model 01", "Roadmap & Files"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if model_df is not None:
        st.markdown(f"<div class='small-label'>LOCAL DATA</div><div style='font-size:.9rem;margin:6px 0 12px'>{len(model_df):,} engineered laps</div>", unsafe_allow_html=True)
    elif raw_laps is not None:
        st.markdown(f"<div class='small-label'>LOCAL DATA</div><div style='font-size:.9rem;margin:6px 0 12px'>{len(raw_laps):,} usable raw laps</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='small-label'>LOCAL DATA</div><div style='font-size:.9rem;margin:6px 0 12px'>No lap CSV loaded</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.75rem;color:#858B99;line-height:1.5'>Research preview<br>Model outputs are labeled by source and stage.</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    hero(
        "Race intelligence, built from public data",
        "Read the race. Understand the stint.",
        "StintIQ brings lap-level pace, tire context and model research into one interactive workspace. This is a research preview: the pace benchmark exists in notebooks; the full strategy stack is still being built.",
        "Model 01 research preview",
    )
    show_data_notice()

    events = options_for(model_df if model_df is not None else raw_laps, "EventName")
    drivers = options_for(model_df if model_df is not None else raw_laps, "Driver")
    metric_row_one = st.columns(2)
    metric_row_two = st.columns(2)
    metric_row_one[0].metric("Model-ready laps", f"{len(model_df):,}" if model_df is not None else "—")
    metric_row_one[1].metric("Raw lap records", f"{len(raw_df):,}" if raw_df is not None else "—")
    metric_row_two[0].metric("Grand Prix covered", f"{len(events)}" if events else "—")
    metric_row_two[1].metric("Drivers in data", f"{len(drivers)}" if drivers else "—")

    left, right = st.columns([1.25, 1])
    with left:
        section("Coverage at a glance", "Lap counts by event in the selected local snapshot.")
        source = model_df if model_df is not None else raw_laps
        if source is not None and {"EventName", "Driver"}.issubset(source.columns):
            coverage = source.groupby("EventName").size().rename("Laps").reset_index()
            coverage["Grand Prix"] = coverage["EventName"].map(display_event)
            coverage = coverage.sort_values("Laps", ascending=True)
            fig = px.bar(coverage, x="Laps", y="Grand Prix", orientation="h", color_discrete_sequence=[RED], text="Laps")
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(showlegend=False, xaxis_title="Lap records", yaxis_title="", height=max(340, len(coverage) * 23), margin=dict(l=12, r=44, t=20, b=30))
            st.plotly_chart(style_figure(fig, max(340, len(coverage) * 23)), use_container_width=True)
        else:
            st.info("Event coverage will appear when a lap dataset is available.")

    with right:
        section("Project status", "A clear view of what the repository supports today.")
        status_rows = [
            ("M1", "Lap pace", "Notebook benchmark recorded", "active"),
            ("M2", "Tire degradation", "Exploration / implementation needed", "explore"),
            ("M3", "Strategy simulation", "Planned", "planned"),
            ("M4", "SC / VSC risk", "Planned", "planned"),
            ("M5", "Race outcome", "Exploratory notebook only", "explore"),
        ]
        for code, name, state, css in status_rows:
            st.markdown(
                f"<div class='surface' style='padding:12px 14px;margin:8px 0;display:flex;align-items:center;gap:12px'>"
                f"<span style='font-family:Space Grotesk,sans-serif;font-weight:700;color:{RED};min-width:28px'>{code}</span>"
                f"<span style='font-weight:700;min-width:132px'>{name}</span><span class='muted' style='font-size:.82rem;flex:1'>{state}</span>"
                f"<span class='status-pill status-{css}'>{'RESEARCH' if css == 'active' else 'IN PROGRESS' if css == 'explore' else 'PLANNED'}</span></div>",
                unsafe_allow_html=True,
            )

    section("How StintIQ is designed", "Predictions are intended to feed a race simulator; the simulator is not presented as complete.")
    flow = st.columns(5)
    flow_items = [
        ("01", "Collect", "FastF1 + public race data"),
        ("02", "Prepare", "Clean laps and engineer features"),
        ("03", "Estimate", "Pace, tire wear and event risk"),
        ("04", "Simulate", "Propagate uncertainty through a race"),
        ("05", "Compare", "Evaluate strategy outcomes"),
    ]
    for col, (number, title, desc) in zip(flow, flow_items):
        with col:
            st.markdown(f"<div class='surface' style='height:126px'><div class='small-label'>{number}</div><div style='font-weight:700;margin:8px 0 3px'>{title}</div><div class='muted' style='font-size:.82rem'>{desc}</div></div>", unsafe_allow_html=True)

    st.caption("Data totals above are calculated from the CSV snapshots available beside this app. Model metrics on the Model 01 page are notebook-reported experiment results.")


# ---------------------------------------------------------------------------
# Race Explorer
# ---------------------------------------------------------------------------
elif page == "Race Explorer":
    hero("Explore the data", "Race Explorer", "Filter a Grand Prix, follow driver pace across laps and compare tire compounds using the data already in the repository.", "Historical data")
    show_data_notice()
    explore_df = frame_for_exploration()
    if explore_df is None or "EventName" not in explore_df.columns:
        st.info("Race Explorer needs a lap dataset with an event column.")
    else:
        events = options_for(explore_df, "EventName") or ["No event data"]
        selected_event = st.selectbox("Grand Prix", events, format_func=display_event)
        event_df = explore_df[explore_df["EventName"].astype(str).eq(selected_event)].copy()

        filter_cols = st.columns([1.25, 1, 1])
        available_drivers = options_for(event_df, "Driver")
        with filter_cols[0]:
            selected_drivers = st.multiselect("Drivers", available_drivers, default=available_drivers[: min(5, len(available_drivers))])
        compounds = options_for(event_df, "Compound")
        with filter_cols[1]:
            selected_compounds = st.multiselect("Compounds", compounds, default=compounds)
        with filter_cols[2]:
            lap_values = pd.to_numeric(event_df.get("LapNumber", pd.Series(dtype=float)), errors="coerce").dropna()
            if not lap_values.empty:
                low, high = int(lap_values.min()), int(lap_values.max())
                lap_range = st.slider("Lap range", min_value=low, max_value=max(low, high), value=(low, max(low, high)))
            else:
                lap_range = None

        filtered = event_df.copy()
        if "Driver" in filtered.columns:
            filtered = filtered[filtered["Driver"].astype(str).isin(selected_drivers)]
        if "Compound" in filtered.columns and selected_compounds:
            filtered = filtered[filtered["Compound"].astype(str).isin(selected_compounds)]
        elif "Compound" in filtered.columns and not selected_compounds:
            filtered = filtered.iloc[0:0]
        if lap_range and "LapNumber" in filtered.columns:
            filtered = filtered[filtered["LapNumber"].between(lap_range[0], lap_range[1])]

        metric_row_one = st.columns(2)
        metric_row_two = st.columns(2)
        metric_row_one[0].metric("Laps in view", f"{len(filtered):,}")
        metric_row_one[1].metric("Drivers", f"{filtered['Driver'].nunique()}" if "Driver" in filtered.columns else "—")
        metric_row_two[0].metric("Tire compounds", f"{filtered['Compound'].nunique()}" if "Compound" in filtered.columns else "—")
        best_delta = pd.to_numeric(filtered.get("lap_time_delta", pd.Series(dtype=float)), errors="coerce").min()
        metric_row_two[1].metric("Fastest relative pace", f"{best_delta:.3f}s" if pd.notna(best_delta) else "—")

        section("Pace by lap", "Pace delta is seconds off the event reference; a smaller value is faster.")
        if filtered.empty:
            st.info("No laps match these filters. Widen the lap range or select more drivers and compounds.")
        elif {"LapNumber", "lap_time_delta", "Driver"}.issubset(filtered.columns):
            color_map = team_color_map(filtered, selected_drivers)
            fig = px.line(
                filtered.sort_values("LapNumber"), x="LapNumber", y="lap_time_delta", color="Driver",
                color_discrete_map=color_map or None, markers=False,
                hover_data=[c for c in ["Team", "Compound", "TyreLife", "LapTimeSeconds"] if c in filtered.columns],
                labels={"LapNumber": "Lap", "lap_time_delta": "Pace delta (s)", "Driver": "Driver"},
            )
            fig.update_yaxes(autorange="reversed", title="Pace delta (s) · lower is faster")
            fig.update_layout(xaxis_title="Lap number", height=430)
            st.plotly_chart(style_figure(fig, 430), use_container_width=True)
        else:
            st.info("This snapshot does not contain the columns needed to plot lap pace.")

        tire_tab, compound_tab = st.tabs(["Tire age and pace", "Compound use"])
        with tire_tab:
            section("Tire age and pace", "Observed laps only; this view does not estimate a future degradation curve.")
            if {"TyreLife", "lap_time_delta", "Compound"}.issubset(filtered.columns):
                fig_tire = px.scatter(
                    filtered, x="TyreLife", y="lap_time_delta", color="Compound",
                    color_discrete_map=COMPOUND_COLORS, opacity=.58,
                    hover_data=[c for c in ["Driver", "LapNumber", "Team"] if c in filtered.columns],
                    labels={"TyreLife": "Tire age (laps)", "lap_time_delta": "Pace delta (s)"},
                )
                fig_tire.update_layout(height=330, xaxis_title="Tire age (laps)", yaxis_title="Pace delta (s)")
                st.plotly_chart(style_figure(fig_tire, 330), use_container_width=True)
            else:
                st.info("Tire age or compound columns are not present in this snapshot.")
        with compound_tab:
            section("Compound use", "Lap records in the selected view.")
            if {"Driver", "Compound"}.issubset(filtered.columns):
                compound_counts = filtered.groupby(["Driver", "Compound"]).size().rename("Laps").reset_index()
                fig_compound = px.bar(compound_counts, x="Driver", y="Laps", color="Compound", barmode="stack", color_discrete_map=COMPOUND_COLORS)
                fig_compound.update_layout(height=330, xaxis_title="Driver", yaxis_title="Laps")
                st.plotly_chart(style_figure(fig_compound, 330), use_container_width=True)
            else:
                st.info("Driver and compound columns are required for this chart.")

        with st.expander("View filtered lap records"):
            display_columns = [c for c in ["LapNumber", "Driver", "Team", "Compound", "TyreLife", "LapTimeSeconds", "lap_time_delta", "AirTemp", "TrackTemp", "Humidity"] if c in filtered.columns]
            sort_columns = [c for c in ["LapNumber", "Driver"] if c in display_columns]
            lap_table = filtered[display_columns]
            if sort_columns:
                lap_table = lap_table.sort_values(sort_columns)
            st.dataframe(lap_table, use_container_width=True, hide_index=True)
            st.download_button("Download filtered laps", filtered.to_csv(index=False).encode("utf-8"), file_name=f"stintiq_{selected_event}_laps.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# Telemetry: speed-trap and sector measurements present in the raw CSV.
# ---------------------------------------------------------------------------
elif page == "Telemetry":
    hero("FastF1 data view", "Telemetry & sectors", "Compare recorded speed traps and sector pace for a selected event. The local CSV contains lap-level speed measurements; distance-indexed telemetry remains in the research notebooks.", "Raw lap fields")
    show_data_notice()
    if raw_laps is None:
        st.info("Telemetry view needs `lap_data_raw.csv` beside the app.")
    else:
        speed_columns = [column for column in ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"] if column in raw_laps.columns]
        sector_columns = [column for column in ["Sector1Time", "Sector2Time", "Sector3Time"] if column in raw_laps.columns]
        if not speed_columns and not sector_columns:
            st.info("No speed-trap or sector-time columns were found in the raw lap file.")
        else:
            tele_events = options_for(raw_laps, "EventName")
            if not tele_events:
                st.info("The raw lap file does not contain event labels.")
            else:
                selected_event = st.selectbox("Grand Prix", tele_events, key="telemetry_event", format_func=display_event)
                event_data = raw_laps[raw_laps["EventName"].astype(str).eq(selected_event)].copy()
                drivers = options_for(event_data, "Driver")
                selected_drivers = st.multiselect("Drivers", drivers, default=drivers[: min(8, len(drivers))], key="telemetry_drivers")
                if "Driver" in event_data.columns:
                    event_data = event_data[event_data["Driver"].astype(str).isin(selected_drivers)]

                for column in speed_columns:
                    event_data[column] = pd.to_numeric(event_data[column], errors="coerce")
                for column in sector_columns:
                    event_data[f"{column}_seconds"] = pd.to_timedelta(event_data[column], errors="coerce").dt.total_seconds()

                metric_cols = st.columns(2)
                metric_cols[0].metric("Laps in view", f"{len(event_data):,}")
                speed_values = event_data[speed_columns].max().dropna() if speed_columns else pd.Series(dtype=float)
                metric_cols[1].metric("Top recorded speed", f"{speed_values.max():.0f} km/h" if not speed_values.empty else "—")

                speed_tab, sector_tab = st.tabs(["Speed traps", "Sector times"])
                with speed_tab:
                    section("Speed-trap comparison", "Average recorded speed by driver and measurement point.")
                    if speed_columns and not event_data.empty:
                        speed_summary = event_data.groupby("Driver")[speed_columns].mean().reset_index().melt(id_vars="Driver", var_name="Measurement", value_name="Speed (km/h)")
                        speed_summary["Measurement"] = speed_summary["Measurement"].replace({"SpeedI1": "Intermediate 1", "SpeedI2": "Intermediate 2", "SpeedFL": "Finish line", "SpeedST": "Start / finish trap"})
                        fig = px.bar(speed_summary, x="Driver", y="Speed (km/h)", color="Measurement", barmode="group", labels={"Driver": "Driver"})
                        fig.update_layout(height=390, yaxis_title="Average speed (km/h)")
                        st.plotly_chart(style_figure(fig, 390), use_container_width=True)
                    else:
                        st.info("No speed-trap readings are available for these filters.")
                with sector_tab:
                    section("Sector pace", "Mean sector duration for the selected drivers.")
                    parsed_sectors = [f"{column}_seconds" for column in sector_columns]
                    if parsed_sectors and not event_data.empty:
                        sector_summary = event_data.groupby("Driver")[parsed_sectors].mean().reset_index().melt(id_vars="Driver", var_name="Sector", value_name="Time (s)")
                        sector_summary["Sector"] = sector_summary["Sector"].replace({"Sector1Time_seconds": "Sector 1", "Sector2Time_seconds": "Sector 2", "Sector3Time_seconds": "Sector 3"})
                        fig = px.bar(sector_summary, x="Driver", y="Time (s)", color="Sector", barmode="group", color_discrete_sequence=["#E10600", "#4B69A8", "#919AAA"])
                        fig.update_layout(height=390, yaxis_title="Average sector time (s)")
                        st.plotly_chart(style_figure(fig, 390), use_container_width=True)
                    else:
                        st.info("No sector timing fields are available for these filters.")

                st.info("Speed-trap and sector summaries are derived from the local raw CSV. Full speed, throttle, brake, gear and distance traces are demonstrated in `FastF1_basics.ipynb`; those channels are not stored as a dataset for this app.")


# ---------------------------------------------------------------------------
# Stint Lab: historical scenario exploration, not an untrained live predictor.
# ---------------------------------------------------------------------------
elif page == "Stint Lab":
    hero("Compare observed stints", "Stint Lab", "Explore how historical pace varies with tire age, compound and driver. Every value here is calculated from observed rows in the local dataset.", "Descriptive analysis")
    show_data_notice()
    stint_data = frame_for_exploration()
    required = {"EventName", "Compound", "TyreLife", "lap_time_delta"}
    if stint_data is None or not required.issubset(stint_data.columns):
        st.info("Stint Lab needs event, compound, tire age and pace-delta columns. The current raw snapshot can still be browsed in Race Explorer.")
    else:
        col_a, col_b, col_c = st.columns([1.15, 1, 1])
        events = options_for(stint_data, "EventName") or ["No event data"]
        with col_a:
            selected_event = st.selectbox("Grand Prix", events, key="stint_event", format_func=display_event)
        event_data = stint_data[stint_data["EventName"].astype(str).eq(selected_event)].copy()
        with col_b:
            available_drivers = options_for(event_data, "Driver")
            selected_driver = st.selectbox("Driver", ["All drivers"] + available_drivers, key="stint_driver") if available_drivers else "All drivers"
        with col_c:
            window = st.selectbox("Comparison window", [1, 2, 3], index=1, format_func=lambda n: f"±{n} tire laps")

        if selected_driver != "All drivers" and "Driver" in event_data.columns:
            event_data = event_data[event_data["Driver"].astype(str).eq(selected_driver)]
        event_data = numeric_columns(event_data, ["TyreLife", "lap_time_delta"])
        event_data = event_data.dropna(subset=["TyreLife", "lap_time_delta", "Compound"])
        compounds = options_for(event_data, "Compound")
        if event_data.empty:
            st.info("There are no rows for this selection.")
        else:
            min_age = max(1, int(np.floor(event_data["TyreLife"].min())))
            max_age = max(min_age, int(np.ceil(event_data["TyreLife"].max())))
            target_age = st.slider("Tire age to inspect", min_value=min_age, max_value=max_age, value=min(max(min_age + 8, min_age), max_age), key="stint_target_age")
            band = event_data[event_data["TyreLife"].between(target_age - window, target_age + window)]
            comparison = band.groupby("Compound").agg(
                MedianDelta=("lap_time_delta", "median"),
                Laps=("lap_time_delta", "size"),
                MedianTyreAge=("TyreLife", "median"),
            ).reindex(compounds).dropna(subset=["MedianDelta"]).reset_index()

            top = st.columns(2)
            top[0].metric("Selected tire age", f"{target_age} laps")
            top[1].metric("Observed laps in window", f"{len(band):,}")
            if not comparison.empty:
                quickest = comparison.sort_values("MedianDelta").iloc[0]
                st.metric("Lowest median delta", f"{quickest['MedianDelta']:.3f}s", str(quickest["Compound"]))
            else:
                st.metric("Lowest median delta", "—")

            profile_tab, compound_tab = st.tabs(["Observed degradation profile", "Compound comparison"])
            with profile_tab:
                section("Observed degradation profile", "Median pace delta by tire age. Each point summarizes the selected event and driver scope.")
                curve = event_data.groupby(["Compound", "TyreLife"], as_index=False).agg(
                    MedianDelta=("lap_time_delta", "median"), Laps=("lap_time_delta", "size")
                )
                fig = px.line(curve, x="TyreLife", y="MedianDelta", color="Compound", markers=True, color_discrete_map=COMPOUND_COLORS, hover_data=["Laps"], labels={"TyreLife": "Tire age (laps)", "MedianDelta": "Median pace delta (s)"})
                fig.add_vline(x=target_age, line_dash="dot", line_color=RED, annotation_text=f"Age {target_age}", annotation_position="top")
                fig.update_layout(height=390, xaxis_title="Tire age (laps)", yaxis_title="Median pace delta (s)")
                st.plotly_chart(style_figure(fig, 390), use_container_width=True)
            with compound_tab:
                section("Compound comparison", f"Median pace in a ±{window}-lap window around tire age {target_age}.")
                if comparison.empty:
                    st.info("No compound has observations in this tire-age window. Try another age.")
                else:
                    fig = px.bar(comparison.sort_values("MedianDelta"), x="MedianDelta", y="Compound", orientation="h", color="Compound", color_discrete_map=COMPOUND_COLORS, text="MedianDelta", hover_data=["Laps", "MedianTyreAge"])
                    fig.update_traces(texttemplate="%{text:.3f}s", textposition="outside", cliponaxis=False)
                    fig.update_layout(showlegend=False, height=330, xaxis_title="Median pace delta (s)", yaxis_title="")
                    st.plotly_chart(style_figure(fig, 330), use_container_width=True)
                    st.dataframe(comparison.rename(columns={"MedianDelta": "Median Δ (s)", "MedianTyreAge": "Median tire age", "Laps": "Laps"}).round(3), use_container_width=True, hide_index=True)

            st.info("This is a historical comparison, not a machine-learning prediction or pit-stop recommendation. Sample sizes differ by compound; use the lap count when interpreting a median.")


# ---------------------------------------------------------------------------
# Model 01: clearly separate notebook results from app-served predictions.
# ---------------------------------------------------------------------------
elif page == "Model 01":
    hero("Model research", "Model 01 · Lap pace", "A cross-event pace-delta benchmark with group holdout validation and quantile uncertainty. Results below are transcribed from the Model 1 notebook; this app does not serve a trained estimator.", "Notebook benchmark")

    metric_row_one = st.columns(2)
    metric_row_two = st.columns(2)
    metric_row_one[0].metric("CatBoost MAE", "0.372 s", "median quantile")
    metric_row_one[1].metric("RMSE", "0.593 s")
    metric_row_two[0].metric("R²", "0.853")
    metric_row_two[1].metric("Lift over AR(1)", "+8.9%", "MAE reduction")

    st.markdown("<div class='surface' style='border-left:4px solid #E10600;margin:10px 0 18px'><strong>What is being predicted?</strong><br><span class='muted'>The notebook models pace delta: lap time minus the event reference. This makes pace deviations comparable across short and long circuits.</span></div>", unsafe_allow_html=True)

    benchmark = pd.DataFrame([
        {"Model": "CatBoost · MultiQuantile", "MAE (s)": 0.372085, "RMSE (s)": 0.592579, "R²": 0.853108},
        {"Model": "XGBoost", "MAE (s)": 0.390731, "RMSE (s)": 0.591545, "R²": 0.853620},
        {"Model": "Random Forest", "MAE (s)": 0.396390, "RMSE (s)": 0.600958, "R²": 0.848925},
        {"Model": "AR(1) persistence", "MAE (s)": 0.408367, "RMSE (s)": 0.670505, "R²": 0.811935},
        {"Model": "Lasso", "MAE (s)": 0.411691, "RMSE (s)": 0.626141, "R²": 0.835998},
        {"Model": "Elastic Net", "MAE (s)": 0.413925, "RMSE (s)": 0.622217, "R²": 0.838047},
        {"Model": "Ridge", "MAE (s)": 0.431180, "RMSE (s)": 0.628249, "R²": 0.834892},
    ]).sort_values("MAE (s)", ascending=True)

    benchmark_tab, validation_tab = st.tabs(["Benchmark results", "Validation design"])
    with benchmark_tab:
        section("Held-out benchmark", "Lower MAE is better. The notebook uses event-grouped training and test sets.")
        colors = [RED if "CatBoost" in name else "#A7AFBD" for name in benchmark["Model"]]
        fig = go.Figure(go.Bar(x=benchmark["MAE (s)"], y=benchmark["Model"], orientation="h", marker_color=colors, text=benchmark["MAE (s)"], texttemplate="%{text:.3f}s", textposition="outside", cliponaxis=False))
        fig.add_vline(x=.30, line_dash="dash", line_color="#2E9B75", annotation_text="Engineering goal · 0.30s", annotation_position="top")
        fig.update_layout(showlegend=False, height=370, xaxis_title="Mean absolute error (seconds)", yaxis_title="")
        st.plotly_chart(style_figure(fig, 370), use_container_width=True)
    with validation_tab:
        section("Validation design", "Entire events are held out to evaluate transfer to unseen circuits.")
        st.markdown("""
        <div class='surface'>
          <div class='small-label'>Group holdout · notebook run</div>
          <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:14px'>
            <div><div style='font-size:1.5rem;font-weight:700'>12</div><div class='muted'>training events</div></div>
            <div><div style='font-size:1.5rem;font-weight:700'>5</div><div class='muted'>held-out events</div></div>
            <div><div style='font-size:1.5rem;font-weight:700'>10,524</div><div class='muted'>training laps</div></div>
            <div><div style='font-size:1.5rem;font-weight:700'>4,669</div><div class='muted'>test laps</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top:12px'><strong>Reported held-out events</strong><br><span class='muted'>Australia · Austria · Belgium · Japan · Singapore</span></div>", unsafe_allow_html=True)

    section("Uncertainty check", "The q10–q90 interval targets 80% coverage; the notebook reports 72.9% on the test set.")
    coverage_col, note_col = st.columns([.38, 1])
    coverage_col.metric("Empirical interval coverage", "72.9%", "7.1 percentage points below nominal")
    with note_col:
        st.warning("The interval is under-covering in the recorded evaluation. Treat the uncertainty output as a prototype that needs calibration before strategy simulation.")

    section("Model boundary", "The repository contains notebook experiments and a model-ready CSV, but no trained estimator wired into this app.")
    st.markdown("""
    <div class='surface'>
      <b>Available here</b><br><span class='muted'>Notebook-reported model comparison, validation setup, and interactive exploration of observed lap data.</span>
      <hr style='border:0;border-top:1px solid #E8EAF0;margin:14px 0'>
      <b>Not served by this app</b><br><span class='muted'>A saved CatBoost model, live pace inference, calibrated confidence intervals, or pit-stop recommendations.</span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Full benchmark table"):
        st.dataframe(benchmark.round(3), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Roadmap and repository map
# ---------------------------------------------------------------------------
elif page == "Roadmap & Files":
    hero("Project map", "From telemetry to strategy", "Browse the intended model sequence and the files that currently support this research preview.", "Build status")

    section("Model roadmap", "The design document calls for environment models first, then simulation and outcome prediction.")
    roadmap = [
        ("01", "Lap pace", "Prototype benchmark in the main notebook; no model artifact connected to the app.", "active"),
        ("02", "Tire degradation", "Fuel-corrected compound curves and cliff detection remain to be implemented.", "planned"),
        ("04", "Safety Car / VSC risk", "Hazard estimation and calibration remain to be implemented.", "planned"),
        ("03", "Strategy simulator", "Monte Carlo pit and compound comparison depends on Models 1, 2 and 4.", "planned"),
        ("05", "Race outcomes", "A simple qualifying-position experiment exists; a validated ranking model remains future work.", "explore"),
    ]
    cols = st.columns(5)
    for col, (number, name, desc, state) in zip(cols, roadmap):
        with col:
            label = {"active": "RESEARCH", "explore": "EXPLORATORY", "planned": "PLANNED"}[state]
            st.markdown(f"<div class='surface' style='height:205px'><div class='small-label'>MODEL {number}</div><div style='font-weight:700;font-size:1.02rem;margin:8px 0'>{name}</div><div class='muted' style='font-size:.82rem;line-height:1.45'>{desc}</div><div style='margin-top:12px'><span class='status-pill status-{state}'>{label}</span></div></div>", unsafe_allow_html=True)

    section("Repository files", "The inventory is generated from files currently present in this project folder.")
    descriptions = {
        "app.py": "This Streamlit showcase and its interactive views.",
        "README.md": "Project purpose, FastF1 setup and the early telemetry-analysis workflow.",
        "load_data.py": "Placeholder for future reusable data-loading utilities; currently empty.",
        "StintIQ_Technical_Document.pdf": "System architecture, model specifications, evaluation and roadmap.",
        "Model1_LapTime_Code_Guide.md": "Detailed walkthrough and presentation notes for the Model 1 notebook.",
        "FastF1_basics.ipynb": "FastF1 session loading, lap timing, telemetry and circuit-map exploration.",
        "Barcelona_Analysis.ipynb": "Barcelona qualifying and race analysis notebook.",
        "Model1_LapTime.ipynb": "Primary pace-delta research, feature engineering, grouped validation and model benchmarks.",
        "Model Development/model_1_LTP.ipynb": "Earlier lap-time and telemetry prototype notebook.",
        "Model Development/winnerProbabModel.ipynb": "Exploratory qualifying-position to race-outcome classification notebook.",
        "lap_data_raw.csv": "Raw merged lap timing, tire, weather and OpenF1 fields.",
        "lap_data_model_ready.csv": "Engineered lap-level features and the pace-delta target used by Model 1.",
        ".streamlit/config.toml": "Local Streamlit server and theme settings.",
    }
    fixed_paths = list(descriptions)
    inventory: list[dict[str, Any]] = []
    for relative in fixed_paths:
        path = ROOT / relative
        if path.exists() and path.is_file():
            inventory.append({
                "File": relative,
                "Role": descriptions[relative],
                "Size": f"{path.stat().st_size / 1024:,.0f} KB",
                "Type": path.suffix.lstrip(".").upper() or "FILE",
            })
    cache_files = sorted((ROOT / "cache").glob("openf1_*.csv")) if (ROOT / "cache").exists() else []
    for path in cache_files:
        relative = path.relative_to(ROOT).as_posix()
        inventory.append({"File": relative, "Role": "Cached OpenF1 race snapshot available for local reference.", "Size": f"{path.stat().st_size / 1024:,.0f} KB", "Type": "CSV"})

    if inventory:
        inventory_df = pd.DataFrame(inventory)
        st.dataframe(inventory_df, use_container_width=True, hide_index=True, height=min(520, 38 * (len(inventory) + 1)))
        file_choices = [item["File"] for item in inventory if (ROOT / item["File"]).is_file()]
        selected_file = st.selectbox("Download a project reference", file_choices)
        selected_path = ROOT / selected_file
        mime = "application/pdf" if selected_path.suffix.lower() == ".pdf" else "text/plain" if selected_path.suffix.lower() in {".md", ".py", ".toml"} else "application/octet-stream"
        st.download_button("Download selected file", selected_path.read_bytes(), file_name=selected_path.name, mime=mime)
    else:
        st.info("No project files were found for the inventory.")

    st.caption(f"Cached OpenF1 CSV snapshots found: {len(cache_files)}. Cache files are listed individually above when present.")

