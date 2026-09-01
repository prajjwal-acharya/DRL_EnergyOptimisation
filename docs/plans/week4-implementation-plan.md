# Week 4 Implementation Plan: Probabilistic Demand/Solar Forecasting Module

This document is the binding, deterministic implementation spec for the Week 4 phase.
It is written to be executed top-to-bottom by an autonomous worker with no other context.
Phases must be completed in order A → B → C → D. Each phase has an acceptance gate that
must pass before the next phase starts.

Working directory for every command is `code/`. Python is `./.venv/bin/python`
(Python 3.9.6 — all new code must be 3.9-compatible: no PEP 604 `X | Y` type unions,
no `match` statements; use `typing.Optional`/`Tuple` or `from __future__ import annotations`).

Weeks 1–3 are complete and frozen. This phase adds the **probabilistic forecasting
module**: point forecasts **plus calibrated prediction intervals** for building demand and
solar generation. It is offline model work only — **no RL training, no controller changes,
no safety shield** in this phase. The controller integration of these forecasts is Week 5.

This phase closes the last September milestone item of the approved research plan
("initial load/PV forecasting model"; mid-semester evidence: "forecasting metrics —
MAE/RMSE and interval calibration").

---

## 0. Ground truth (verified facts — do not re-derive, do not contradict)

### 0.1 Environment and data contract

- CityLearn `2.5.0` pinned to source tag `v2.5.0`, commit `29062af6d077409e1c37a3e53a6cac30fd4d02bc`.
- Pinned dataset on disk (read-only, never modify):
  `data/raw/citylearn_challenge_2023_phase_1/`
  - `Building_1.csv` — 720 hourly rows (30 days), header included at line 1. Columns
    (verified): `month, hour, day_type, daylight_savings_status,
    indoor_dry_bulb_temperature, average_unmet_cooling_setpoint_difference,
    indoor_relative_humidity, non_shiftable_load, dhw_demand, cooling_demand,
    heating_demand, solar_generation, occupant_count,
    indoor_dry_bulb_temperature_cooling_set_point,
    indoor_dry_bulb_temperature_heating_set_point, hvac_mode`.
  - `weather.csv` — 720 rows. Actuals: `outdoor_dry_bulb_temperature,
    outdoor_relative_humidity, diffuse_solar_irradiance, direct_solar_irradiance`.
    Dataset-issued forecasts at row `t` for `t+1/t+2/t+3`:
    `outdoor_dry_bulb_temperature_predicted_1/2/3`, `outdoor_relative_humidity_predicted_1/2/3`,
    `diffuse_solar_irradiance_predicted_1/2/3`, `direct_solar_irradiance_predicted_1/2/3`.
  - `pricing.csv` — 720 rows: `electricity_pricing, electricity_pricing_predicted_1/2/3`.
  - `carbon_intensity.csv` — 720 rows.
- Step indexing convention (binding, matches weeks 2–3): CSV row `i` (0-based, after the
  header) is simulation step `t = i`. Dev window = steps 0–167; final window = 0–719.
- Causality of the dataset's own predicted columns: the `*_predicted_1/2/3` columns in
  row `t` are forecasts issued at `t` for `t+1/t+2/t+3`. Reading them at forecast origin
  `t` is causal. Reading **any actual (non-predicted) column at a row > t** is lookahead
  and is forbidden everywhere in this phase.
- No new third-party dependencies. `torch` (already installed via
  `stable-baselines3==2.3.2`) and `numpy 1.26.4` / `pandas 2.3.3` are sufficient.
  There is no scikit-learn in the environment — do not use it.

### 0.2 Observation-dimension correction (documentation fix, carry into review doc)

The single-building central-agent observation has **29 slots** (verified:
`build_observation_index('configs/schema-building1.json')`
returns 29 names; `configs/week3-ppo.yaml` freezes 29 normalisation entries). The
"49-dim" wording in `docs/plans/week3-implementation-plan.md` §preamble and
`docs/status/phase-reviews/week3-review.md` Phase A is a stale parent-scenario reference. Fix those two doc
strings to 29 in this phase's commit (text-only change; no code or config change).

### 0.3 Frozen problem definition (locked before any model is trained)

| Decision | Value | Reason |
| --- | --- | --- |
| Forecast targets | `solar_generation` (`E_pv`), `non_shiftable_load` (`L_nsl`), `cooling_demand` (`Q_cool`) | the demand/solar signals the plan names; `dhw_demand` is excluded (DHW tank decouples it from control; smallest signal) — may be added in a later named config, not this one |
| Forecast horizon | `H = 3` hourly steps (h ∈ {1, 2, 3}) | matches the dataset's own 3-step predicted-weather slots, and the hour-ahead decisions available to the 1-hour control loop |
| Quantile levels | τ ∈ {0.05, 0.25, 0.50, 0.75, 0.95} | gives the 90% and 50% central intervals; τ = 0.50 is the point forecast |
| Point-forecast definition | the τ = 0.50 quantile | one consistent definition across all models |
| Quantile monotonicity | enforce `q05 ≤ q25 ≤ q50 ≤ q75 ≤ q95` by cumulative maximum over sorted τ at prediction time | guarantees valid intervals from every model including neural ones |
| Evaluation scheme | rolling-origin (expanding-window) backtest, folds frozen in §0.4 | the only honest accuracy evidence with 720 rows; also the plan's own mitigation for "forecast data/model is insufficient" |
| RL-integration refit | after selection, the winning variant per target is refit on all 720 steps; that artefact is what Week 5 consumes | unavoidable (the policy needs forecasts from step 0 onward); residual leakage is disclosed and **symmetric across Week-5 comparison arms** — see `docs/plans/week5-implementation-plan.md` |
| Compute | CPU only, seed 42 for every model fit | reproducibility contract |
| Metrics library | implemented in numpy inside this repo | no new dependencies; the metric code itself is a testable deliverable |

### 0.4 Backtest fold scheme (frozen — do not tune after seeing results)

- Initial training segment: steps `0–239` (first 10 days).
- 12 test folds, each 40 steps: fold `k` (k = 1…12) covers steps `240 + 40·(k−1)` …
  `279 + 40·(k−1)` (last fold ends at 719).
- For each fold, every model is **refit from scratch** on all steps `< fold start`
  (expanding window), then produces quantile forecasts at every origin `t` inside the
  fold for every horizon h ∈ {1, 2, 3}.
- Out-of-sample totals: 480 origins × 3 horizons = 1440 evaluated `(t, h)` pairs per
  target per model. Pooled metrics are computed over these; per-fold metrics are saved
  but not decision-driving.
- The dev window (0–167) lies inside every training segment by construction. Therefore
  **no dev-window forecast-accuracy claim is made** — the dev window's role is control
  evaluation. This sentence must appear verbatim in `docs/status/phase-reviews/week4-review.md`.
- Per-fold internal validation (early stopping / conformal calibration): only the last
  20% of that fold's training segment may be used; never fold data.

### 0.5 Frozen input features (identical feature contract for every learned model)

At forecast origin `t` (all values from rows ≤ t; predicted columns only from row t):

| Group | Columns | Count |
| --- | --- | --- |
| Calendar | `sin(hour)`, `cos(hour)`, `day_type` | 3 |
| Weather actuals at t | `outdoor_dry_bulb_temperature`, `diffuse_solar_irradiance`, `direct_solar_irradiance` | 3 |
| Dataset weather forecasts | the 9 `*_predicted_1/2/3` irradiance + temperature columns of row t | 9 |
| Tariff | `electricity_pricing`, `electricity_pricing_predicted_1/2/3` | 4 |
| Carbon | `carbon_intensity` at t | 1 |
| Target history | `y_t`, `y_{t−24}` (same-hour previous day) | 2 |
| **Total** | | **22** |

`linear_quantile` consumes exactly these 22 (one model per target × horizon × τ is
allowed but prefer a single shared pinball multi-head — implementation freedom, frozen
once written). `gru_quantile` consumes the past 24 steps of
`[target, outdoor_dry_bulb_temperature, diffuse_solar_irradiance, direct_solar_irradiance,
electricity_pricing]` (5 × 24 sequence) plus the 22 static features above.

### 0.6 Model ladder (each rung must beat the one below or it does not ship)

| Model | Definition | Role |
| --- | --- | --- |
| `persistence_last` | ŷ_{t+h} = y_t for every h | sanity floor |
| `persistence_24h` | ŷ_{t+h} = y_{t+h−24} (same hour yesterday) | the realistic naive floor for hourly building data |
| `climatology_hourly` | ŷ_{t+h} = mean of y over training rows with the same `hour` | unconditional reference |
| `linear_quantile` | linear model, pinball loss per τ, torch full-batch Adam (lr 0.01, 2000 steps, seed 42) | interpretable learned floor |
| `gru_quantile` | 1-layer GRU (hidden 32) + per-τ linear heads, joint pinball loss, Adam lr 1e-3, batch 64, max 200 epochs, early stop patience 10 on the fold-internal validation split, seed 42 | the plan's named deep option |

The persistence/climatology rungs emit "quantiles" as point masses (all τ equal). They
are still scored with the full interval metric set — their intervals will be
degenerate; that is expected and is exactly why they cannot win selection unless every
learned model fails.

### 0.7 Declared pass bars and selection rule (frozen before evaluation)

- **Coverage bar** (pooled OOS, per target × horizon): a variant is *calibrated* if the
  90% central interval's empirical coverage ∈ [0.85, 0.95] and the 50% interval's
  empirical coverage ∈ [0.42, 0.58].
- **Conformal fallback** (declared now, applied mechanically, not judgementally): if a
  learned model fails the coverage bar, produce a `+conformal` variant by split-conformal
  adjustment (absolute residual quantile on the fold-internal calibration split,
  90% and 50% levels) and score it identically. Both raw and conformalised results are
  reported; only bar-passing variants are selectable.
- **Naive-floor bar**: a variant is *competitive* on a target only if its median-MAE
  (pooled OOS, averaged over horizons) ≤ min(MAE of both persistence rungs) for that
  target. If no learned variant is competitive on a target, **persistence ships for
  that target** and the failure is recorded as a finding.
- **Selection rule** (per target): among calibrated + competitive variants, select the
  lowest **mean pinball loss** averaged over the five τ levels and three horizons;
  tie-break: narrower mean 90% interval width. The selection is executed by script,
  written to `results/runs/forecasting/selected_models.json`, and never overridden by hand.

### 0.8 Solar-specific caveat (must appear in the review doc)

`solar_generation` is exactly 0 at night. Pooled MAE/coverage are dominated by trivial
night zeros. All solar metrics are therefore reported **twice**: pooled over all OOS
steps, and restricted to daylight origins (row `t` where
`direct_solar_irradiance + diffuse_solar_irradiance > 0`). Selection for
`solar_generation` uses the **daylight-restricted** pinball mean; the choice is part of
this frozen rule.

### 0.9 Existing code to build on (do not duplicate; do not modify weeks 1–3 behaviour)

- `src/energy_optimisation/forecasting/` — reserved empty package; all new module code
  goes here.
- `configs/` — new subdirectory `configs/` for this phase's frozen config.
- `results/` — new root `results/runs/forecasting/`; nothing under `results/runs/baselines/`,
  `results/runs/ppo/`, `results/runs/smoke/`, `results/inspection/`, or
  `results/tables/baseline_comparison.csv`, `ppo_multiseed_summary.csv`,
  `ppo_vs_baselines.csv` may be modified.
- All 60 existing tests must keep passing, unchanged.

---

## Phase A — Data pipeline and metrics module

### A1. `src/energy_optimisation/forecasting/data.py`

- `load_dataset(dataset_dir) -> Dataset` — loads `Building_1.csv`, `weather.csv`,
  `pricing.csv`, `carbon_intensity.csv` into aligned float64 arrays; validates row counts
  (720) and column presence against §0.1; raises on mismatch.
- `build_forecast_frame(dataset, origin_t, target, horizon) -> (x, y_true)` — assembles
  the §0.5 feature vector from rows ≤ t plus row-t predicted columns, and the true value
  `y[t + horizon]`. This function is the single enforcement point of the causality
  contract: it must accept an explicit `t` and never read a row index > t for actuals.
- `make_folds(dataset, scheme) -> List[Fold]` — materialises the §0.4 scheme
  (fold `k`: train `< 240 + 40·(k−1)`, test the 40 fold steps).

### A2. `src/energy_optimisation/forecasting/metrics.py`

Pure numpy functions, each taking prediction/quantile arrays and truth: `mae`, `rmse`,
`pinball_loss(y_true, y_q, tau)`, `empirical_coverage(y_true, q_lo, q_hi)`,
`mean_interval_width`, `winkler_score(y_true, q_lo, q_hi, alpha)` (the standard
interval score: width + 2/α · penalty per miss). No metric may drop NaNs silently.

### A3. Tests (`tests/test_forecasting.py`)

- `test_dataset_row_counts_and_columns`
- `test_forecast_frame_causality` — calling `build_forecast_frame` at t, then corrupting
  every actual-value row > t in the dataset, must return an identical feature vector.
  Corrupting rows ≤ t (or row-t predicted columns) must change it.
- `test_folds_match_frozen_scheme` — fold boundaries exactly §0.4; every train segment
  ends before its test segment starts.
- `test_metrics_on_hand_computed_cases` — MAE, pinball (τ = 0.5 reduces to MAE/2),
  coverage, and Winkler on small hand-computed arrays.
- `test_quantile_monotonicity_enforcement` — a deliberately crossing quantile vector is
  repaired to non-decreasing order by the shared helper.

**Acceptance gate A:** all new tests pass; all 60 existing tests still pass.

---

## Phase B — Model ladder

### B1. `src/energy_optimisation/forecasting/models.py`

A common interface `fit(frame_train) -> self` / `predict_quantiles(origin_t) -> dict[τ -> float]`,
implemented by `PersistenceLast`, `Persistence24h`, `ClimatologyHourly`,
`LinearQuantile`, `GruQuantile` exactly per §0.5–§0.6. Every learned model: CPU, seed 42,
dtype float64→float32 only at the torch boundary. Training hyperparameters live in the
config (Phase C), never in code. The split-conformal adjustment is a wrapper
`Conformalized(model, level)` implementing §0.7's fallback.

### B2. `src/energy_optimisation/forecasting/api.py` (the Week-5 interface, delivered now)

`ForecastProvider` — loads the frozen selection (`selected_models.json` + refit-on-full-
series weights), exposes `predict_quantiles(t) -> {target: {horizon: {τ: value}}}` and
`feature_vector(t, variant)` where variant ∈ {`point`, `interval`} produces exactly the
Week-5 appended-feature blocks (defined in `docs/plans/week5-implementation-plan.md` §0.6).
Structural causality: the provider internally truncates the dataset to rows ≤ t before
any model sees it.

### B3. Tests (append to `tests/test_forecasting.py`)

- `test_persistence_definitions` — constructed 48-step series, exact expected outputs.
- `test_climatology_uses_only_training_hours`
- `test_linear_and_gru_deterministic_per_seed` — two fits from the same seed produce
  identical predictions (tolerance 0; gru compared after fixed-epoch early stop).
- `test_gru_quantiles_monotone` on a real fold.
- `test_conformal_widens_intervals` — conformalised interval ⊇ raw interval on the
  calibration split.
- `test_provider_causality` — mutating rows > t does not change
  `predict_quantiles(t)`; mutating row t does.

**Acceptance gate B:** all tests pass; every model in the ladder runs end-to-end on
fold 1 without error.

---

## Phase C — Backtest execution, selection, tables and figures

### C1. Frozen config `configs/week4-forecasting.yaml`

One file containing: dataset path; targets; quantiles; horizon; the §0.4 fold scheme;
the §0.5 feature list; every §0.6 hyperparameter; seeds; the §0.7 bars and selection
rule; the §0.8 daylight rule; output paths. Written **before** the first full backtest
run and never edited afterwards (SHA-256 recorded by the runner into every artefact).

### C2. `scripts/15_train_forecasters.py --config configs/week4-forecasting.yaml`

For every model × target × fold: refit on the fold's training segment, predict all
in-fold origins × horizons, and write:

- `results/runs/forecasting/<model>/<target>/predictions.csv` —
  `fold, t, horizon, y_true, q05, q25, q50, q75, q95`
- `results/runs/forecasting/<model>/<target>/metrics.json` — pooled and per-fold metrics
  (MAE, RMSE, pinball per τ, coverage 90%/50%, mean widths, Winkler 90%/50%);
  for `solar_generation`, both pooled and daylight-restricted blocks.
- `results/runs/forecasting/backtest_run_metadata.json` — git commit, config SHA-256,
  torch/numpy versions, wall clock, seeds.

### C3. `scripts/16_compare_forecasters.py`

Executes the frozen §0.7 selection mechanically and writes:

- `results/tables/forecast_model_comparison.csv` — one row per model × target:
  median-MAE (avg over horizons), mean pinball, coverage 90/50, mean width 90,
  Winkler 90, `calibrated` flag, `competitive` flag.
- `results/tables/forecast_calibration_by_hour.csv` — empirical 90% coverage per
  target × hour-of-day for the selected variants (the calibration-behaviour evidence).
- `results/runs/forecasting/selected_models.json` — per target: selected variant, its
  metrics, the rule trace, config hash.
- Figures under `results/figures/`:
  `forecast_<target>_fanchart.png` (a representative 48 h OOS window: truth, median,
  50% and 90% bands, selected model), `forecast_coverage_by_target.png`
  (nominal vs empirical, 90% and 50%), `forecast_pinball_by_horizon.png`
  (selected model, per target × horizon).

**Acceptance gate C:** all tables/figures exist and are NaN-free; the selection rule's
output is reproducible (deleting `selected_models.json` and re-running the script
reproduces it byte-identically); any target where persistence ships is recorded in the
run note, not hidden.

---

## Phase D — Verification, documentation, commit

### D1. `scripts/17_gate_week4.py` (mirror `14_gate_week3.py` structure; hard pass/fail)

- `./.venv/bin/python -m pytest -q` passes (60 existing + all new tests).
- `configs/week4-forecasting.yaml` exists and its SHA-256 matches the hash recorded in
  `results/runs/forecasting/backtest_run_metadata.json`.
- All three targets have a `selected_models.json` entry with a complete rule trace.
- `predictions.csv` exists for every model × target; row counts match §0.4 arithmetic
  (480 origins × 3 horizons).
- `forecast_model_comparison.csv` and `forecast_calibration_by_hour.csv` exist with the
  expected schema; no NaNs.
- The three required figures exist.
- Quantile monotonicity holds in every `predictions.csv` row.
- Weeks 1–3 evidence untouched: `results/tables/{baseline_comparison,ppo_multiseed_summary,
  ppo_vs_baselines}.csv` byte-identical to git HEAD; nothing under
  `results/runs/baselines/`, `results/runs/ppo/` modified (git status clean for those paths).
- `docs/status/phase-reviews/week4-review.md` exists.

### D2. `docs/status/phase-reviews/week4-review.md` (written last, honestly)

Structure: infrastructure evidence first (pipeline, causality enforcement, interface),
then forecasting results (ladder table, calibration, selection), then negative findings
— e.g. if GRU loses to linear, or persistence ships for a target, that is a headline
finding, not a footnote. Must contain verbatim: (a) the §0.4 dev-window disclaimer
sentence, (b) the §0.8 solar caveat, (c) the leakage disclosure from §0.3
(full-series refit for RL integration; backtest evidence is the honest accuracy claim),
and (d) the supervisor update:

> The probabilistic forecasting module is complete: a five-rung model ladder is
> evaluated under a frozen rolling-origin backtest, with point accuracy (MAE/RMSE) and
> interval calibration (coverage, width, pinball, Winkler) reported per target and
> horizon, and a selected, calibrated forecaster per target frozen for controller
> integration. No controller, RL, or safety-shield work was touched.

### D3. Commit

Single commit closing the phase, e.g.
`feat: week-4 probabilistic forecasting module` (includes the §0.2 doc corrections).

**Acceptance gate D:** `./.venv/bin/python scripts/17_gate_week4.py` exits 0; repo green;
all changes committed.

---

## Guardrails (binding for the worker)

- **Out of scope:** no RL training or evaluation, no observation/adapter changes, no
  safety shield, no scenario/robustness work, no dashboard, no manuscript work.
- Never modify anything under `data/raw/`, or weeks 1–3 results/configs/tests/docs
  except the two §0.2 text corrections.
- Never install new dependencies; torch/numpy/pandas only.
- The config is frozen before the first full backtest; if a bar fails, apply the
  declared conformal fallback or ship persistence — never retune hyperparameters after
  seeing metrics to make a model win.
- Never hand-edit anything under `results/`; regenerate from scripts only.
- If a command fails: retry once; if it fails again, record the error and treat it as a
  blocker. Do not weaken or skip a failing test.

## Definition of done

1. `data.py` / `metrics.py` / `models.py` / `api.py` implemented with all Phase A/B tests
   passing; 60 pre-existing tests unchanged and green.
2. Five-rung ladder (+ conformal fallback) backtested under the frozen §0.4 scheme for
   all three targets × three horizons × five quantiles; predictions and metrics on disk.
3. Frozen selection executed: one calibrated, competitive forecaster per target recorded
   in `selected_models.json` (persistence shipping for a target counts, if that is the
   honest outcome).
4. `ForecastProvider` delivered with a passing structural-causality test — the exact
   interface Week 5 consumes.
5. `results/tables/forecast_model_comparison.csv`, `forecast_calibration_by_hour.csv`,
   and the three figures exist and are NaN-free.
6. `./.venv/bin/python -m pytest -q` passes; `./.venv/bin/python scripts/17_gate_week4.py`
   exits 0.
7. `docs/status/phase-reviews/week4-review.md` written with the required verbatim statements; phase committed.
