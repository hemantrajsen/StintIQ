# StintIQ Model 2: Tire Degradation Guide

## Recommendation

**Yes, Model 2 is worth building as a small, interpretable pace-loss model.** It fills a real gap between lap-pace prediction and the planned race-strategy simulator: Model 3 needs to estimate how the pace of a stint will change as a tire ages, including under a different compound or pit plan.

Build it in stages. Start with a fuel- and context-adjusted curve and a simple uncertainty range. Add change-point or nonlinear models only if they improve predictions on complete held-out stints. The data supports a useful prototype, but it does not justify promising an exact tire “cliff” or a physical estimate of remaining tire life.

The current `lap_data_raw.csv` snapshot contains 19,644 lap rows across 17 events and 1,059 raw event-driver-stint keys. Soft and medium compounds appear at all 17 events, while hard appears at 15. These are pre-cleaning counts; they show enough coverage to start a pooled pilot, not enough evidence for reliable circuit-by-circuit cliff estimates after filtering.

## Current v0.1 implementation

The first implementation now lives in [Model2.ipynb](Model2.ipynb) and [model2_tire_degradation.py](model2_tire_degradation.py). It leaves [Model1_LapTime.ipynb](Model1_LapTime.ipynb) untouched and builds a separate non-tire baseline in the Model 2 pipeline.

The notebook now checks for `lap_data_raw.csv` before its legacy race-acquisition cells. If that saved dataset exists, it loads it once, skips the schedule/FastF1/OpenF1 exploration cells, and passes the in-memory dataframe to the Model 2 pipeline.

- The shared dry, accurate, green-flag data with pit-in/out laps removed leaves **15,193 laps across 843 stints and 17 events**. An event-best +8% guardrail removes extreme lap-time records, matching the current Model 1 cleaning choice; this is an initial retrospective rule that still needs sensitivity review.
- The non-tire baseline excludes `TyreLife`, `Compound`, `Stint`, and lap-pace lags. It uses estimated fuel load, race progress, weather, traffic gap, event, driver, and team. Cross-fitted predictions by whole stint produce the residual target for Model 2.
- Model 2 compares a zero-tire-effect baseline and a linear/quadratic curve with a gradient-boosted quantile curve. On one held-out-stint split, MAE is **0.621 s** for the quantile curve, **0.652 s** for the linear/quadratic curve, and **0.659 s** when tire effect is set to zero. The q10–q90 interval covers **81.0%** of held-out residuals.
- The first pooled change-point analysis finds **no reliable cliff** for the three compounds. Circuit-specific cliff estimates, live rolling-origin evaluation, interval calibration, and Model 3 integration remain open work.

These are prototype metrics, not deployment results: each event appears in both train and test through different stints, so this evaluation does not establish transfer to an unseen event or circuit. Artifacts are written under `models/` when the final training cell runs.

## 1. What Model 2 is meant to do

Model 2 estimates the **lap-time cost associated with tire use**. For a chosen compound, tire age, circuit context, and temperature, it should estimate how many seconds of pace are lost relative to a defined post-warm-up reference. It should also show how uncertain that estimate is and whether the queried age is supported by the training data.

The model is intended to:

- Describe and forecast the tire-related pace change across a stint.
- Compare supported compound and tire-age scenarios on the same circuit/context.
- Provide a degradation curve, pace loss per lap, uncertainty, and a cautious estimate of a change point when the data supports one.
- Supply Model 3 with the tire contribution to expected lap time, so a simulator can compare pit laps and compound sequences.

Model 2 does **not** estimate physical tread depth, tire temperature inside the carcass, puncture risk, or exact remaining tire life. Public lap timing measures the pace outcome indirectly. The model should call its output “tire-related pace loss,” not physical wear.

## 2. Where it sits in the project

The intended relationship is:

```text
Observed lap time
  = non-tire pace (car/driver/event baseline)
  + fuel effect
  + traffic and track-state effects
  + tire-related pace loss (Model 2)
  + unexplained variation
```

Model 1 estimates lap pace, Model 2 isolates the tire-age contribution, and Model 3 combines those components with pit-stop and race-state assumptions. Each effect must be added once.

### Important Model 1 handoff decision

The current Model 1 notebook uses `TyreLife` and `Compound` as prediction features. Its residuals may therefore have already had much of the tire signal explained away. Passing those residuals directly into Model 2 can make degradation appear too small; adding a full Model 2 curve back to Model 1 predictions can count tire effects twice.

Use one of these explicit designs:

1. **Preferred for a clean first implementation: fit a joint additive model.** Include non-tire controls and an explicit tire-age/compound curve in one model. Model 2's output is the tire component of that fit.
2. **Residual handoff: create a non-tire Model 1 baseline.** Exclude direct tire features and close proxies such as stint-age lags from that baseline, generate out-of-fold or past-only predictions, and train Model 2 on the remaining tire signal.

Do not treat the present Model 1 predictions as a non-tire baseline without changing their feature design. The first v0.1 implements a separate baseline inside Model 2 instead. For live backtests, also avoid Model 1's current full-event fastest-lap reference and its first-lap lag fill from the current target; both can expose future information.

## 3. Define the target before fitting

The technical document specifies **fuel-corrected pace delta** as the Model 2 target. A practical formulation is:

\[
T_i = B_i + F_i + C_i + G(c_i, a_i, x_i) + \epsilon_i
\]

where:

- `T_i` is observed lap time or event-relative lap-time delta.
- `B_i` is the non-tire car/driver/event baseline.
- `F_i` is the fuel-mass effect.
- `C_i` collects measured context such as traffic, weather, track evolution, and race status.
- `G(c_i, a_i, x_i)` is tire-related pace loss for compound `c`, tire age `a`, and context `x` such as circuit and temperature.
- `epsilon_i` is unexplained lap-to-lap variation.

The desired Model 2 value is `G`, in seconds per lap. If fuel is corrected directly, use a documented reference fuel load:

\[
T_{i,\mathrm{fuel\ ref}} = T_i - \hat\gamma (m_i - m_{\mathrm{ref}})
\]

Here `m_i` is estimated fuel mass and `gamma` is the estimated seconds-per-kilogram sensitivity. Do not quietly assume that an approximate fuel burn or penalty is exact. Estimate it where possible, and report a sensitivity range when fuel mass is only a proxy.

Define the curve relative to a **post-warm-up reference**, not automatically the first lap on a new set. Pit-out laps should be excluded from the degradation fit, but normal tire warm-up can extend beyond the pit-out lap. Either model warm-up separately or set the reference age to a supported post-warm-up window and document it.

## 4. Data to use and cleaning rules

### Source

Use the project-wide `lap_data_raw.csv` as the source for the first model. It contains lap timing, driver/team, circuit/event, stint, compound, tire age, weather, and timing/status fields. `lap_data_model_ready.csv` was prepared for Model 1 and does not retain all Model 2 context needed here, including circuit and track temperature; extend the shared data pipeline if a reusable model-ready table is desired.

The original exploration cells in `Model2.ipynb` test one Australian race and use a separate OpenF1 call path. The new Section 11 instead uses the shared raw CSV, so its event names and `TyreLife` field align with Model 1. Keep the older API cells as exploration only unless their age formula and cleaning are reconciled.

### First scope: dry slick race stints

Start with SOFT, MEDIUM, and HARD race laps. Keep intermediate, wet, mixed-weather, and transition laps out of the first fit; they follow different regimes and should not be pooled into dry-compound curves.

For each lap retain or derive:

- Event, circuit, year, driver, team, lap number, stint, compound, and tire age.
- Lap time and sector times, plus valid/accurate-lap indicators.
- Pit-in, pit-out, and track-status flags, including Safety Car/VSC and yellow/neutralized laps.
- Track and air temperature, rainfall/track regime, and session progress or track-evolution proxy.
- Fuel-load proxy and a consistent traffic/gap-to-car-ahead feature where available.
- A source timestamp or explicit information boundary for backtesting.

### Cleaning requirements

1. Keep one row per event-driver-lap-stint and diagnose duplicate joins before removing them.
2. Remove missing/invalid lap times, pit-in and pit-out laps, and non-green/neutralized laps from the initial dry-pace fit.
3. Replace the draft's fixed `lap_duration < 110` cutoff with a robust, event-aware outlier rule. Inspect removed laps; incidents and mechanical failures should not silently become tire degradation.
4. The v0.1 training pipeline uses the shared raw `TyreLife` field and does not recompute age. If the original OpenF1 cells are reused, reconcile their formula (which omits the `+ 1` used by Model 1), verify the source meaning of `tyre_age_at_start`, and use one convention throughout.
5. Record counts before and after every filter, by event, compound, and stint. Suppress or pool curves with too few independent stints.
6. In live-style backtests, weather and other joins must use information available by the prediction time. A nearest-time join can select a future weather record; use a backward/as-of join for those evaluations.

## 5. Model path: start simple and add complexity only when earned

### Baseline A: descriptive adjusted curves

After cleaning and context adjustment, plot per-stint adjusted pace against tire age. Summarize with medians and uncertainty bands by compound. This checks whether the data has a plausible signal; it is not yet a predictive model.

### Baseline B: linear or quadratic degradation

Fit a simple curve by compound, with a stint-specific intercept and an age slope. Add a quadratic or spline term only if held-out stints show curvature that the linear model misses. Pool evidence across circuits and drivers; use circuit/temperature interactions only where there is enough independent data.

This gives an interpretable baseline for:

- Pace loss at a given tire age.
- Average loss per additional tire lap over the observed range.
- Differences across compounds, with uncertainty.

Do not force every compound or circuit to have a different slope when the data cannot distinguish them.

### Change point / “cliff”

Only after the baseline works, fit a segmented curve such as:

\[
G(a) = b_0 + b_1a + b_2\max(0, a-k)
\]

where `k` is a candidate change point. Compare candidate `k` values with a complexity penalty or use a Bayesian change-point model. Require enough observations on both sides of `k`. Bootstrap or otherwise retain a distribution/interval over the change point. If the interval is wide, report “no reliable cliff detected” instead of a precise lap.

### Optional nonlinear model

Try Gradient Boosting/CatBoost only after the interpretable baselines and only if it improves whole-stint holdout results. Tree models can represent interactions, but they may extrapolate poorly beyond ages seen in training. Keep the interpretable curve as a sanity check and mark unsupported tire ages rather than projecting them confidently.

### Optional sequential model

A Bayesian state-space model can represent hidden tire pace, fuel effects, lap-to-lap noise, and pit-stop resets. It is most useful if the goal expands to updating a driver's degradation estimate during a live race. It is a later step because it adds modeling and inference complexity.

## 6. Validation plan

Random row splits are not acceptable: adjacent laps from the same stint share tire state and time-series context.

Use three evaluation views:

1. **Held-out stints:** Keep every lap in a stint together. This tests performance on a new stint for known events/drivers.
2. **Held-out events:** Hold out whole races to test transfer to new circuit/weather context. Use grouped folds rather than one lucky split where practical.
3. **Rolling-origin predictions:** For live-use claims, predict the next lap using only earlier observations. Rebuild fuel, reference, weather, and lag features at each cutoff.

Report:

- MAE as the primary pace-loss error and RMSE as a check on large misses.
- Performance against a no-age baseline and the linear/quadratic baseline.
- Results by compound, circuit, temperature range, and tire-age range, with sample/stint counts.
- Interval coverage or calibration for uncertainty outputs.
- Cliff detection within ±2 laps only when an independent reference label is defensible; otherwise report change-point interval width and stability.
- Extrapolation stress tests beyond the age range observed for each compound/circuit.
- Ablations removing fuel correction, traffic, weather, and track-evolution controls to reveal how much they affect the estimated tire curve.

The technical document's `<0.30 s` engineering target belongs to Model 1. It is not a Model 2 acceptance threshold. Set Model 2's acceptance from held-out-stint baselines and downstream needs; require any added complexity to beat the simpler alternative.

## 7. Model 3 output contract

Produce a versioned table or callable predictor with at least:

| Field | Meaning |
|---|---|
| `Compound`, `CircuitName` | Scenario keys; include broader pooled fallback where circuit support is sparse. |
| `TrackTemp` or temperature band | Environmental context represented in training. |
| `TyreLife` | Tire age at which pace loss is requested. |
| `PaceLossMedianSec` | Estimated tire-related seconds per lap relative to the documented reference. |
| `PaceLossQ10Sec`, `PaceLossQ90Sec` | Uncertainty bounds for simulation. |
| `DegradationSlopeSecPerLap` | Local or average pace-loss rate over a stated age interval. |
| `CliffAgeMedian`, `CliffAgeLow`, `CliffAgeHigh` | Optional change-point estimate and interval; nullable when unsupported. |
| `ObservedAgeMax`, `NStints`, `SupportFlag` | Evidence and extrapolation guardrails. |

Model 3 should add the Model 2 tire contribution exactly once, sample from the uncertainty distribution when simulating, and receive a clear fallback/unsupported response when a scenario exceeds observed support.

## 8. Suggested notebook implementation order

Refactor `Model2.ipynb` into these sections before considering a separate package:

1. Scope, target equation, and Model 1 interface.
2. Load shared raw data and print event/compound/stint coverage.
3. Validate schema, joins, tire-age convention, and duplicate keys.
4. Clean laps and build fuel/context features with documented assumptions.
5. Create the target and show pre/post correction diagnostic plots.
6. Plot descriptive adjusted curves with per-stint summaries and uncertainty.
7. Create grouped held-out-stint and held-out-event splits.
8. Fit no-age, linear, and quadratic baselines.
9. Evaluate change-point estimates; add nonlinear comparison only if useful.
10. Produce residual, calibration, stratified-error, and extrapolation checks.
11. Export the model/curve table, feature schema, data version, and evaluation summary.
12. Demonstrate one Model 3-style query and explicitly flag unsupported scenarios.

Keep reusable functions small and explicit, for example: `build_model2_frame`, `make_group_splits`, `fit_degradation_baseline`, `fit_change_point`, `evaluate_model2`, and `predict_tire_pace_loss`. Avoid embedding data downloads, cleaning, model fitting, plots, and exports in one long cell.

## 9. Definition of done and worth-it decision

Model 2 is ready for downstream research use when:

- It runs across the shared multi-event source and documents its cleaning/target decisions.
- Its tire-age convention and Model 1 decomposition are consistent and leakage-safe.
- The adjusted curve is supported by multiple stints and compared with no-age and simple curve baselines.
- Held-out-stint results, compound/circuit breakdowns, uncertainty, and unsupported age ranges are visible.
- Change points are reported with uncertainty and may be omitted when the signal is weak.
- A versioned output can answer Model 3's compound/age/context query without double-counting pace effects.

**Worth building:** yes, because the strategy simulator needs a tire-age pace component and the repository already has enough raw stints to test a pooled, interpretable first version. **Worth overbuilding now:** no. Begin with the adjusted curve and a change-point check; pursue a nonlinear or state-space model only if it improves honest held-out predictions or enables needed live updating. If the adjusted tire curve does not beat the no-age baseline consistently, keep Model 2 descriptive and do not feed it into strategy recommendations as a validated predictor.

## References

- Project specification: [StintIQ_Technical_Document.pdf](StintIQ_Technical_Document.pdf)
- Model 1 implementation: [Model1_LapTime.ipynb](Model1_LapTime.ipynb), [Model1_LapTime_Code_Guide.md](Model1_LapTime_Code_Guide.md)
- Shared data inventory: [lap_data_raw.csv](lap_data_raw.csv)
- Cappello and Hoegh (2026), “A state-space approach to modeling tire degradation in formula 1 racing,” *Journal of Sports Analytics*. The paper models fuel and pit-stop resets and evaluates rolling-origin predictions across 19 race sessions; it reports that compound-specific degradation differences were not always statistically distinct, a useful caution against overclaiming compound/cliff effects from sparse public timing data. [Article](https://doi.org/10.1177/22150218261446170)
