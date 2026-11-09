"""First-pass, leakage-aware Model 2 tire pace-loss pipeline.

Model 2 is trained on residuals from a separate non-tire baseline. The existing
Model 1 notebook and its benchmark remain untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


RANDOM_STATE = 42
REFERENCE_TIRE_AGE = 5
DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")

BASELINE_NUMERIC = [
    "EstimatedFuelKg",
    "SessionProgress",
    "TrackTemp",
    "AirTemp",
    "Humidity",
    "TimeGapAhead",
]
BASELINE_CATEGORICAL = ["EventName", "Driver", "Team"]
TIRE_NUMERIC = ["TyreLife", "TrackTemp"]
TIRE_CATEGORICAL = ["Compound", "CircuitName"]


def _as_bool(series: pd.Series) -> pd.Series:
    """Parse CSV booleans consistently across FastF1/OpenF1 rows."""
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _nonblank(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def _add_gap_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Use a supplied gap or approximate same-lap gap from elapsed race time."""
    out = df.copy()
    if "TimeGapAhead" in out.columns:
        gap = pd.to_numeric(out["TimeGapAhead"], errors="coerce")
    elif "Time" in out.columns:
        elapsed = pd.to_timedelta(out["Time"], errors="coerce").dt.total_seconds()
        work = out[["EventName", "LapNumber"]].copy()
        work["_elapsed"] = elapsed
        work["_original_index"] = np.arange(len(work))
        work = work.sort_values(["EventName", "LapNumber", "_elapsed"])
        gap_sorted = work.groupby(["EventName", "LapNumber"], sort=False)["_elapsed"].diff()
        work["_gap"] = gap_sorted
        gap = work.set_index("_original_index")["_gap"].reindex(np.arange(len(out)))
        gap.index = out.index
    else:
        gap = pd.Series(np.nan, index=out.index)

    # A large value denotes no usable car-ahead observation; clip extreme gaps.
    gap = gap.where(gap.ge(0), np.nan).fillna(99.0).clip(upper=99.0)
    out["TimeGapAhead"] = gap.astype(float)
    return out


def prepare_model2_data(csv_path: str | Path | pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load or reuse shared lap data and make a dry-slick modeling frame."""
    if isinstance(csv_path, pd.DataFrame):
        raw = csv_path.copy()
    else:
        raw = pd.read_csv(csv_path, low_memory=False)
    required = {
        "EventName", "CircuitName", "Driver", "Team", "LapNumber", "Stint",
        "Compound", "TyreLife", "LapTimeSeconds", "IsAccurate", "TrackStatus",
        "Rainfall", "TotalLaps", "TrackTemp", "AirTemp", "Humidity",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"lap_data_raw.csv is missing required columns: {sorted(missing)}")

    df = raw.copy()
    numeric_columns = [
        "LapNumber", "Stint", "TyreLife", "LapTimeSeconds", "TotalLaps",
        "TrackTemp", "AirTemp", "Humidity",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ("EventName", "CircuitName", "Driver", "Team"):
        df[column] = df[column].fillna("UNKNOWN").astype(str)
    df["Compound"] = df["Compound"].fillna("UNKNOWN").astype(str).str.upper().str.strip()

    keep = (
        df["LapTimeSeconds"].gt(0)
        & df["TyreLife"].ge(1)
        & df["Compound"].isin(DRY_COMPOUNDS)
        & _as_bool(df["IsAccurate"])
        & pd.to_numeric(df["TrackStatus"], errors="coerce").eq(1)
        & ~_as_bool(df["Rainfall"])
    )
    for column in ("PitInTime", "PitOutTime"):
        if column in df.columns:
            keep &= ~_nonblank(df[column])

    clean = df.loc[keep].copy()
    before_outlier_filter = len(clean)

    # Initial project-aligned incident guardrail. This is a retrospective data
    # cleaning rule only, not a live feature or a claim that 8% is universal.
    event_reference = clean.groupby("EventName")["LapTimeSeconds"].transform("min")
    clean = clean.loc[clean["LapTimeSeconds"].le(1.08 * event_reference)].copy()

    # Prevent duplicate joins from overweighting one observation.
    duplicate_key = ["EventName", "Driver", "LapNumber", "Stint"]
    duplicate_count = int(clean.duplicated(duplicate_key).sum())
    clean = clean.sort_values("LapTimeSeconds").drop_duplicates(duplicate_key, keep="first")

    clean["EstimatedFuelKg"] = (
        (clean["TotalLaps"] - clean["LapNumber"]).clip(lower=0) * 1.75
    )
    clean["SessionProgress"] = (
        clean["LapNumber"] / clean["TotalLaps"].replace(0, np.nan)
    ).fillna(0.0)
    clean = _add_gap_feature(clean)
    clean["StintKey"] = (
        clean["EventName"] + "|" + clean["Driver"] + "|" + clean["Stint"].astype(str)
    )
    clean = clean.dropna(subset=["LapTimeSeconds", "TyreLife", "EstimatedFuelKg"])
    clean = clean.reset_index(drop=True)

    summary = {
        "raw_rows": int(len(raw)),
        "after_quality_filters": int(before_outlier_filter),
        "after_event_outlier_guardrail": int(len(clean)),
        "removed_duplicate_keys": duplicate_count,
        "events": int(clean["EventName"].nunique()),
        "circuits": int(clean["CircuitName"].nunique()),
        "stints": int(clean["StintKey"].nunique()),
        "compound_laps": {str(k): int(v) for k, v in clean["Compound"].value_counts().items()},
        "tire_age_range": [float(clean["TyreLife"].min()), float(clean["TyreLife"].max())],
    }
    return clean, summary


def _baseline_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), BASELINE_NUMERIC),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                BASELINE_CATEGORICAL,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=140,
        learning_rate=0.06,
        max_leaf_nodes=15,
        min_samples_leaf=35,
        l2_regularization=2.0,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("prep", preprocess), ("model", model)])


def _tire_pipeline(alpha: float) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), TIRE_NUMERIC),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                TIRE_CATEGORICAL,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=180,
        learning_rate=0.04,
        max_depth=2,
        min_samples_leaf=25,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("prep", preprocess), ("model", model)])


def _crossfit_baseline(
    frame: pd.DataFrame,
    n_splits: int = 5,
) -> tuple[np.ndarray, Pipeline]:
    """Return out-of-stint-fold baseline predictions and a full-fit baseline."""
    x = frame[BASELINE_NUMERIC + BASELINE_CATEGORICAL]
    y = frame["LapTimeSeconds"].to_numpy(dtype=float)
    groups = frame["StintKey"].to_numpy()
    n_groups = len(np.unique(groups))
    folds = min(n_splits, n_groups)
    oof = np.full(len(frame), np.nan, dtype=float)

    splitter = GroupKFold(n_splits=folds)
    for fit_idx, val_idx in splitter.split(x, y, groups):
        baseline = _baseline_pipeline()
        baseline.fit(x.iloc[fit_idx], y[fit_idx])
        oof[val_idx] = baseline.predict(x.iloc[val_idx])

    full_baseline = _baseline_pipeline()
    full_baseline.fit(x, y)
    if np.isnan(oof).any():
        raise RuntimeError("Cross-fitted baseline left rows without predictions.")
    return oof, full_baseline


def _fit_tire_quantiles(frame: pd.DataFrame, residuals: np.ndarray) -> dict[float, Pipeline]:
    x = frame[TIRE_NUMERIC + TIRE_CATEGORICAL]
    models = {alpha: _tire_pipeline(alpha) for alpha in (0.10, 0.50, 0.90)}
    for alpha, model in models.items():
        model.fit(x, residuals)
    return models


def _linear_tire_design(frame: pd.DataFrame) -> pd.DataFrame:
    """Explicit compound-specific linear/quadratic age features."""
    age = frame["TyreLife"].astype(float) - REFERENCE_TIRE_AGE
    design = pd.DataFrame(index=frame.index)
    design["TrackTemp"] = pd.to_numeric(frame["TrackTemp"], errors="coerce")
    for compound in DRY_COMPOUNDS:
        is_compound = frame["Compound"].eq(compound).astype(float)
        design[f"AgeSlope_{compound}"] = age * is_compound
        design[f"AgeCurve_{compound}"] = age.pow(2) * is_compound
    design["Compound"] = frame["Compound"].astype(str)
    design["CircuitName"] = frame["CircuitName"].astype(str)
    return design


def _linear_tire_pipeline() -> Pipeline:
    numeric = [
        "TrackTemp",
        *[f"AgeSlope_{c}" for c in DRY_COMPOUNDS],
        *[f"AgeCurve_{c}" for c in DRY_COMPOUNDS],
    ]
    categorical = ["Compound", "CircuitName"]
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("prep", preprocess), ("model", Ridge(alpha=10.0))])


def evaluate_model2(
    frame: pd.DataFrame,
    test_size: float = 0.25,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate on fully held-out stints with nested cross-fitted baseline."""
    groups = frame["StintKey"].to_numpy()
    all_idx = np.arange(len(frame))
    split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(split.split(all_idx, groups=groups))
    train = frame.iloc[train_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)

    train_baseline_oof, _ = _crossfit_baseline(train)
    baseline_test = _baseline_pipeline()
    baseline_test.fit(
        train[BASELINE_NUMERIC + BASELINE_CATEGORICAL],
        train["LapTimeSeconds"].to_numpy(dtype=float),
    )
    test_baseline_pred = baseline_test.predict(test[BASELINE_NUMERIC + BASELINE_CATEGORICAL])

    train_residual = train["LapTimeSeconds"].to_numpy(dtype=float) - train_baseline_oof
    test_residual = test["LapTimeSeconds"].to_numpy(dtype=float) - test_baseline_pred
    linear_tire = _linear_tire_pipeline()
    linear_tire.fit(_linear_tire_design(train), train_residual)
    linear_pred = linear_tire.predict(_linear_tire_design(test))
    tire_models = _fit_tire_quantiles(train, train_residual)
    x_test_tire = test[TIRE_NUMERIC + TIRE_CATEGORICAL]
    q10 = tire_models[0.10].predict(x_test_tire)
    q50 = tire_models[0.50].predict(x_test_tire)
    q90 = tire_models[0.90].predict(x_test_tire)
    quantiles = np.sort(np.column_stack([q10, q50, q90]), axis=1)
    q10, q50, q90 = quantiles.T

    prediction_frame = test[
        ["EventName", "CircuitName", "Driver", "StintKey", "Compound", "TyreLife"]
    ].copy()
    prediction_frame["ActualTireResidualSec"] = test_residual
    prediction_frame["PredictedQ10Sec"] = q10
    prediction_frame["PredictedQ50Sec"] = q50
    prediction_frame["PredictedQ90Sec"] = q90
    prediction_frame["LinearQuadraticPredictionSec"] = linear_pred
    prediction_frame["ZeroEffectAbsErrorSec"] = np.abs(test_residual)
    prediction_frame["Model2AbsErrorSec"] = np.abs(test_residual - q50)
    per_compound = {}
    for compound, part in prediction_frame.groupby("Compound", observed=True):
        indices = part.index.to_numpy()
        per_compound[str(compound)] = {
            "test_laps": int(len(part)),
            "test_stints": int(part["StintKey"].nunique()),
            "zero_tire_effect_mae_sec": float(mean_absolute_error(test_residual[indices], np.zeros(len(indices)))),
            "linear_quadratic_tire_mae_sec": float(mean_absolute_error(test_residual[indices], linear_pred[indices])),
            "model2_median_mae_sec": float(mean_absolute_error(test_residual[indices], q50[indices])),
            "q10_q90_coverage_pct": float(np.mean((test_residual[indices] >= q10[indices]) & (test_residual[indices] <= q90[indices])) * 100.0),
        }

    metrics = {
        "train_laps": int(len(train)),
        "test_laps": int(len(test)),
        "train_stints": int(train["StintKey"].nunique()),
        "test_stints": int(test["StintKey"].nunique()),
        "train_events": int(train["EventName"].nunique()),
        "test_events": int(test["EventName"].nunique()),
        "zero_tire_effect_mae_sec": float(mean_absolute_error(test_residual, np.zeros(len(test_residual)))),
        "linear_quadratic_tire_mae_sec": float(mean_absolute_error(test_residual, linear_pred)),
        "model2_median_mae_sec": float(mean_absolute_error(test_residual, q50)),
        "model2_median_rmse_sec": float(np.sqrt(mean_squared_error(test_residual, q50))),
        "q10_q90_coverage_pct": float(np.mean((test_residual >= q10) & (test_residual <= q90)) * 100.0),
        "per_compound": per_compound,
        "warning": "Stint holdout is not a live rolling-origin backtest; event and driver context may be represented by other stints in the same race.",
    }
    evaluation = {
        "test_frame": test,
        "test_residual": test_residual,
        "q10": q10,
        "q50": q50,
        "q90": q90,
        "predictions": prediction_frame,
        "metrics": metrics,
    }
    return metrics, evaluation


def _curve_table(
    frame: pd.DataFrame,
    tire_models: dict[float, Pipeline],
    reference_age: float = REFERENCE_TIRE_AGE,
    min_stints: int = 3,
    min_laps: int = 40,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    support = frame.groupby(["CircuitName", "Compound"], observed=True).agg(
        n_laps=("TyreLife", "size"),
        n_stints=("StintKey", "nunique"),
        max_age=("TyreLife", "max"),
        median_track_temp=("TrackTemp", "median"),
    ).reset_index()

    for item in support.itertuples(index=False):
        if item.n_stints < min_stints or item.n_laps < min_laps or item.max_age < reference_age:
            continue
        ages = np.arange(1, int(np.floor(item.max_age)) + 1, dtype=int)
        query = pd.DataFrame(
            {
                "TyreLife": ages,
                "TrackTemp": float(item.median_track_temp),
                "Compound": item.Compound,
                "CircuitName": item.CircuitName,
            }
        )
        q_predictions = {}
        x_query = query[TIRE_NUMERIC + TIRE_CATEGORICAL]
        for alpha, model in tire_models.items():
            q_predictions[alpha] = model.predict(x_query)

        ref_idx = int(np.argmin(np.abs(ages - reference_age)))
        ref_median = q_predictions[0.50][ref_idx]
        q_matrix = np.sort(
            np.column_stack([q_predictions[0.10], q_predictions[0.50], q_predictions[0.90]]),
            axis=1,
        )
        for i, age in enumerate(ages):
            rows.append(
                {
                    "CircuitName": item.CircuitName,
                    "Compound": item.Compound,
                    "TrackTempReferenceC": float(item.median_track_temp),
                    "TyreLife": int(age),
                    "ReferenceTyreLife": int(ages[ref_idx]),
                    "PaceLossQ10Sec": float(q_matrix[i, 0] - ref_median),
                    "PaceLossQ50Sec": float(q_matrix[i, 1] - ref_median),
                    "PaceLossQ90Sec": float(q_matrix[i, 2] - ref_median),
                    "NStints": int(item.n_stints),
                    "NLaps": int(item.n_laps),
                    "ObservedAgeMax": float(item.max_age),
                    "SupportFlag": "supported_range",
                }
            )
    return pd.DataFrame(rows)


def _weighted_bic(
    subset: pd.DataFrame,
    residuals: np.ndarray,
    breakpoint: int | None,
) -> float:
    """Fit a weighted pooled line/hinge curve with circuit intercepts."""
    if len(subset) != len(residuals):
        raise ValueError("Breakpoint subset and residual arrays are misaligned.")
    age = subset["TyreLife"].to_numpy(dtype=float)
    circuits = pd.get_dummies(subset["CircuitName"].astype(str), drop_first=True, dtype=float).to_numpy()
    columns = [np.ones(len(subset)), age - REFERENCE_TIRE_AGE]
    if breakpoint is not None:
        columns.append(np.maximum(age - breakpoint, 0.0))
    if circuits.size:
        columns.append(circuits)
    design = np.column_stack(columns)

    group_column = "_BootstrapStint" if "_BootstrapStint" in subset.columns else "StintKey"
    stint_sizes = subset.groupby(group_column, observed=True)[group_column].transform("size").to_numpy()
    weights = 1.0 / np.maximum(stint_sizes, 1)
    root_w = np.sqrt(weights)
    coef = np.linalg.lstsq(design * root_w[:, None], residuals * root_w, rcond=None)[0]
    error = residuals - design @ coef
    weighted_sse = float(np.sum(weights * error**2))
    n_eff = max(2, subset[group_column].nunique())
    parameter_count = design.shape[1]
    return float(n_eff * np.log(max(weighted_sse / n_eff, 1e-12)) + parameter_count * np.log(n_eff))


def estimate_compound_cliffs(
    frame: pd.DataFrame,
    residuals: np.ndarray,
    n_bootstrap: int = 60,
    min_stints_each_side: int = 8,
    min_laps_each_side: int = 80,
) -> pd.DataFrame:
    """Estimate exploratory pooled compound change points with stint bootstrap.

    The candidate is a change in the residual pace-loss slope, pooled across
    circuits with circuit-specific intercepts. It is not a physical wear limit
    and has no external ground-truth label in the current project data.
    """
    work = frame[["Compound", "CircuitName", "TyreLife", "StintKey"]].copy().reset_index(drop=True)
    work["_Residual"] = np.asarray(residuals, dtype=float)
    rng = np.random.default_rng(RANDOM_STATE)
    results: list[dict[str, Any]] = []

    for compound in DRY_COMPOUNDS:
        subset = work.loc[work["Compound"].eq(compound)].copy().reset_index(drop=True)
        unique_stints = subset["StintKey"].unique()
        empty = {
            "Compound": compound,
            "NStints": int(len(unique_stints)),
            "NLaps": int(len(subset)),
            "CandidateCliffAge": np.nan,
            "CliffAgeQ10": np.nan,
            "CliffAgeQ90": np.nan,
            "BICImprovement": np.nan,
            "BootstrapSupportPct": 0.0,
            "CliffDetected": False,
            "Status": "insufficient support",
        }
        if len(unique_stints) < 2 * min_stints_each_side:
            results.append(empty)
            continue

        y = subset["_Residual"].to_numpy(dtype=float)
        linear_bic = _weighted_bic(subset, y, breakpoint=None)
        age_min = int(np.ceil(subset["TyreLife"].quantile(0.05)))
        age_max = int(np.floor(subset["TyreLife"].quantile(0.95)))
        candidates = []
        for candidate in range(max(8, age_min + 2), min(41, age_max - 2), 2):
            before = subset["TyreLife"].le(candidate)
            after = ~before
            n_stints_before = subset.loc[before, "StintKey"].nunique()
            n_stints_after = subset.loc[after, "StintKey"].nunique()
            if (
                before.sum() < min_laps_each_side
                or after.sum() < min_laps_each_side
                or n_stints_before < min_stints_each_side
                or n_stints_after < min_stints_each_side
            ):
                continue
            bic = _weighted_bic(subset, y, breakpoint=candidate)
            candidates.append((candidate, bic))

        if not candidates:
            results.append(empty)
            continue

        candidate_age, candidate_bic = min(candidates, key=lambda item: item[1])
        improvement = linear_bic - candidate_bic
        boot_candidates: list[int] = []
        boot_supported = 0
        for _ in range(n_bootstrap):
            sampled = rng.choice(unique_stints, size=len(unique_stints), replace=True)
            boot_parts = []
            for replicate, stint in enumerate(sampled):
                part = subset.loc[subset["StintKey"].eq(stint)].copy()
                part["_BootstrapStint"] = f"{replicate}:{stint}"
                boot_parts.append(part)
            boot = pd.concat(boot_parts, ignore_index=True)
            boot_y = boot["_Residual"].to_numpy(dtype=float)
            boot_linear_bic = _weighted_bic(boot, boot_y, breakpoint=None)
            boot_grid = []
            for candidate, _ in candidates:
                before = boot["TyreLife"].le(candidate)
                after = ~before
                if (
                    before.sum() < min_laps_each_side
                    or after.sum() < min_laps_each_side
                    or boot.loc[before, "_BootstrapStint"].nunique() < min_stints_each_side
                    or boot.loc[after, "_BootstrapStint"].nunique() < min_stints_each_side
                ):
                    continue
                boot_grid.append((candidate, _weighted_bic(boot, boot_y, breakpoint=candidate)))
            if not boot_grid:
                continue
            boot_age, boot_bic = min(boot_grid, key=lambda item: item[1])
            if boot_linear_bic - boot_bic >= 6.0:
                boot_supported += 1
                boot_candidates.append(boot_age)

        support_pct = (boot_supported / n_bootstrap) * 100.0
        reliable = improvement >= 6.0 and support_pct >= 70.0 and len(boot_candidates) >= 10
        results.append(
            {
                "Compound": compound,
                "NStints": int(len(unique_stints)),
                "NLaps": int(len(subset)),
                "CandidateCliffAge": int(candidate_age),
                "CliffAgeQ10": float(np.quantile(boot_candidates, 0.10)) if boot_candidates else np.nan,
                "CliffAgeQ90": float(np.quantile(boot_candidates, 0.90)) if boot_candidates else np.nan,
                "BICImprovement": float(improvement),
                "BootstrapSupportPct": float(support_pct),
                "CliffDetected": bool(reliable),
                "Status": "candidate supported" if reliable else "no reliable cliff detected",
            }
        )
    return pd.DataFrame(results)


def predict_tire_pace_loss(
    bundle: dict[str, Any],
    compound: str,
    circuit_name: str,
    tire_age: float,
    track_temp_c: float,
) -> dict[str, Any]:
    """Query the v0.1 tire-loss curve, refusing unsupported extrapolations."""
    compound = str(compound).upper().strip()
    circuit_name = str(circuit_name)
    key = f"{circuit_name}|{compound}"
    support = bundle.get("curve_support", {}).get(key)
    if support is None:
        return {"supported": False, "reason": "No curve meets the minimum lap/stint support for this circuit and compound."}

    if tire_age < support["min_age"] or tire_age > support["max_age"]:
        return {
            "supported": False,
            "reason": f"Tire age is outside observed support [{support['min_age']}, {support['max_age']}].",
        }
    if track_temp_c < support["track_temp_q05"] or track_temp_c > support["track_temp_q95"]:
        return {
            "supported": False,
            "reason": f"Track temperature is outside the central observed range [{support['track_temp_q05']:.1f}, {support['track_temp_q95']:.1f}] C.",
        }

    reference_age = float(bundle["reference_tire_age"])
    query = pd.DataFrame(
        {
            "TyreLife": [float(tire_age), reference_age, float(tire_age)],
            "TrackTemp": [float(track_temp_c)] * 3,
            "Compound": [compound] * 3,
            "CircuitName": [circuit_name] * 3,
        }
    )[TIRE_NUMERIC + TIRE_CATEGORICAL]
    predictions = {}
    for alpha, model in bundle["quantile_models"].items():
        predictions[alpha] = model.predict(query)
    target_age_values = np.sort(
        np.array([predictions[0.10][0], predictions[0.50][0], predictions[0.90][0]])
    )
    reference_median = predictions[0.50][1]
    return {
        "supported": True,
        "compound": compound,
        "circuit_name": circuit_name,
        "tire_age": float(tire_age),
        "track_temp_c": float(track_temp_c),
        "reference_tire_age": reference_age,
        "pace_loss_q10_sec": float(target_age_values[0] - reference_median),
        "pace_loss_q50_sec": float(target_age_values[1] - reference_median),
        "pace_loss_q90_sec": float(target_age_values[2] - reference_median),
        "n_stints": int(support["n_stints"]),
        "observed_age_max": float(support["max_age"]),
        "interval_note": "Preliminary conditional quantiles; the interval for an age difference is approximate and needs calibration.",
    }


def train_final_model2(
    frame: pd.DataFrame,
    output_dir: str | Path,
    summary: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit final cross-fitted residual models and save artifacts for review."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    baseline_oof, full_baseline = _crossfit_baseline(frame)
    residuals = frame["LapTimeSeconds"].to_numpy(dtype=float) - baseline_oof
    tire_models = _fit_tire_quantiles(frame, residuals)
    linear_tire_model = _linear_tire_pipeline()
    linear_tire_model.fit(_linear_tire_design(frame), residuals)
    curves = _curve_table(frame, tire_models)
    cliff_estimates = estimate_compound_cliffs(frame, residuals)
    curve_support: dict[str, dict[str, Any]] = {}
    if not curves.empty:
        support_keys = curves[["CircuitName", "Compound"]].drop_duplicates()
        for item in support_keys.itertuples(index=False):
            part = frame.loc[
                frame["CircuitName"].eq(item.CircuitName) & frame["Compound"].eq(item.Compound)
            ]
            curve_support[f"{item.CircuitName}|{item.Compound}"] = {
                "min_age": float(part["TyreLife"].min()),
                "max_age": float(part["TyreLife"].max()),
                "n_stints": int(part["StintKey"].nunique()),
                "track_temp_q05": float(part["TrackTemp"].quantile(0.05)),
                "track_temp_q95": float(part["TrackTemp"].quantile(0.95)),
            }

    bundle: dict[str, Any] = {
        "model_name": "StintIQ Model 2 Tire Pace-Loss Quantiles",
        "model_version": "0.1.0",
        "baseline_type": "cross-fitted non-tire HistGradientBoostingRegressor",
        "baseline_features": BASELINE_NUMERIC + BASELINE_CATEGORICAL,
        "tire_features": TIRE_NUMERIC + TIRE_CATEGORICAL,
        "reference_tire_age": REFERENCE_TIRE_AGE,
        "quantile_models": tire_models,
        "linear_quadratic_reference_model": linear_tire_model,
        "pooled_compound_cliff_estimates": cliff_estimates.to_dict(orient="records"),
        "curve_support": curve_support,
        "non_tire_baseline": full_baseline,
        "fuel_assumption_kg_per_lap": 1.75,
        "cleaning_rule": "dry SOFT/MEDIUM/HARD; accurate green laps; no pit-in/out; TyreLife>=1; event-best+8% extreme-lap guardrail",
        "limitations": [
            "First research version; curve quantiles require calibration before strategy use.",
            "Cliff estimates are pooled by compound across circuits and are not validated against external cliff labels.",
            "Do not extrapolate beyond ObservedAgeMax or to unsupported circuits/conditions.",
            "Model 3 must combine a non-tire pace baseline with this tire effect; do not add this curve to the current tire-aware Model 1 prediction.",
        ],
    }
    joblib.dump(bundle, output / "model2_tire_pace_loss.joblib")
    curves.to_csv(output / "model2_degradation_curves.csv", index=False)
    cliff_estimates.to_csv(output / "model2_cliff_estimates.csv", index=False)
    if summary is not None:
        (output / "model2_data_summary.json").write_text(json.dumps(summary, indent=2))
    return bundle, curves

