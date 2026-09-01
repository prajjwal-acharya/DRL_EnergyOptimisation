# Week 2 Review — CMDP Formulation and Deterministic Baselines

Phase review for `docs/plans/week2-implementation-plan.md` (Phases A–D). All numbers below are read
from recorded phase evidence (`results/runs/baselines/**`, `results/tables/baseline_comparison.csv`)
produced by the frozen config `configs/week2-baselines.yaml` (seed 42; windows dev 0–167 /
final 0–719; CityLearn 2.5.0). Nothing under `results/` was hand-edited; every artifact is
regenerable via `scripts/10_run_baselines.py` and `scripts/11_compare_baselines.py`.

## 1. Infrastructure evidence

- **CMDP specification** (`docs/reference/cmdp-spec.md`, Phase A): all six sections complete — state table
  (29 Building_1 observations, name-addressed via `src/energy_optimisation/observation_names.py`),
  action table, transition statement, reward declaration, constraints, KPI mapping.
  Recorded constants: `Ē_B0 = 0.477229108554339` kWh, `P_ref = 7.694016456604004` kWh,
  `P_grid,max = 1.8084184527397151` kWh (95th percentile of the B0 dev-window net-consumption
  series), τ = 0.0439 USD/kWh, ΔT = 2.0 °C, reserve band [0.2, 0.9], w_E/w_P/w_C = 1.0/1.0/10.0.
- **Harness validated before controllers existed** (Phase B): `tests/test_runner.py::`
  `test_b0_matches_smoke_kpis` reproduces every non-empty §0 smoke anchor on window 0–167,
  seed 42, tolerance 1e-9 — worst observed |delta| = 0. Seed-reproducibility, artifact
  completeness, and NaN-free traces are pinned by dedicated tests.
- **Test suite**: `./.venv/bin/python -m pytest -q` → **39 passed** (week-1 environment tests
  included; Phase A/B/C test files added).
- **Measurement convention documented** (cmdp_spec §1): under CityLearn 2.5.0 the post-step
  observation vector carries uncomputed zeros for computed slots (`net_electricity_consumption`,
  SoCs, indoor temperature); the runner reads executed values from the building time series that
  `env.evaluate()` consumes, and repairs controller inputs causally (no lookahead).
- **Episode-length convention**: CityLearn terminates an episode after exactly
  `simulation_end_time_step − simulation_start_time_step` hourly steps, so the frozen windows
  produce 167 (dev) and 719 (final) trace rows — the same convention that produced the §0 smoke
  anchors (`completed_steps == 167`).

## 2. Controller-performance results

Primary KPIs (normalised by CityLearn against its no-action baseline; lower is better except as
noted). Source: `results/tables/baseline_comparison.csv`.

### Dev window (steps 0–167)

| Controller | cost_total ↓ | all_time_peak_average ↓ | electricity_consumption_total ↓ | discomfort_hot_proportion ↓ | discomfort_proportion ↓ | ramping_average ↓ | zero_net_energy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| b0_neutral | **0.4420** | **0.8618** | **0.4641** | 0.9152 | 0.9152 | **0.8572** | **0.3500** |
| b1_fixed_schedule | 1.8631 | 1.0920 | 1.7639 | **0.0000** | 0.8788 | 1.2024 | 1.7952 |
| b2_tariff_aware | 1.6709 | 1.0920 | 1.6773 | **0.0000** | **0.8667** | 0.9771 | 1.7183 |

### Final window (steps 0–719)

| Controller | cost_total ↓ | all_time_peak_average ↓ | electricity_consumption_total ↓ | discomfort_hot_proportion ↓ | discomfort_proportion ↓ | ramping_average ↓ | zero_net_energy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| b0_neutral | **0.3636** | **0.8618** | **0.3883** | 0.9804 | 0.9804 | **0.8299** | **0.2666** |
| b1_fixed_schedule | 1.7259 | 1.1381 | 1.6706 | **0.0000** | 0.9719 | 1.2579 | 1.6812 |
| b2_tariff_aware | 1.5568 | 1.0920 | 1.5971 | **0.0000** | **0.9691** | 1.0170 | 1.6195 |

The two outage KPIs (`one_minus_thermal_resilience_proportion`,
`power_outage_normalized_unserved_energy_total`) are empty in this dataset and excluded per plan §0.

## 3. Negative outcome recorded (no tuning applied)

**Finding:** both active baselines *increase* normalised cost and consumption relative to B0
(do-nothing) on both windows; tariff-aware B2 beats fixed-schedule B1 on cost in both windows
(dev −10.3%, final −9.8%) but remains far above B0. Per plan §D2 step 3 this is a recorded
finding, not a failure to fix — constants were not tuned.

Analysis (mechanistic, from traces and derived metrics):

1. **`cooling_device` is an electrical-input ratio, not a setpoint.** B1/B2 drive it at 0.5–0.8
   across most daytime hours while B0 leaves it at 0; total electricity consumption rises to
   ~1.6–1.8× of the CityLearn normalisation baseline (vs 0.39–0.46 for B0). The comfort signal
   flips accordingly: hot-discomfort proportion drops 0.9152 → 0 (dev) / 0.9804 → 0 (final),
   while cold-discomfort proportion appears at 0.879–0.972 — the building is systematically
   overcooled, which is exactly what a fixed open-loop cooling schedule without feedback on
   indoor temperature produces.
2. **Storage cycling adds losses without cost benefit.** Battery round-trip inefficiency
   (η = 0.95) plus charge/discharge spread over only 3 price levels (peak/mid ratio < 2.03×) is
   insufficient to recover losses at these frozen action levels. B2's battery barely leaves the
   reserve edge (dev SoC range 0.1967–0.2000): daily peak-window discharge (~5 kWh requested)
   exceeds overnight recharge at level −0.5 within the small 4 kWh / 3.32 kW device, so B2
   effectively runs the battery down to the reserve band and holds it there. The DHW tank SoC
   stays at 0 throughout for B1/B2 (tank capacity 2.2827 kWh is consumed by demand faster than
   level −0.5 charging restores it), so the DHW term contributes cycling losses only.
   B1's battery cycles the full usable band (SoC up to 0.99993) yet still loses money —
   consistent with point 1 dominating.
3. **Peaks worsen slightly.** Midday cooling coincides with high net demand:
   `all_time_peak_average` rises 0.8618 → 1.0920 (dev) for both controllers, and derived
   `peak_net_demand_kw` rises from 7.694 (B0) to 9.749–10.160 (B1/B2).
4. **What did transfer:** tariff awareness helps where it can act. B2 improves on B1 for
   cost_total, electricity_consumption_total, ramping_average, and discomfort_proportion on both
   windows while eliminating all hot discomfort. This supports RQ3's contrast (price-adaptive >
   calendar-only) even though neither beats the do-nothing reference at these action levels.

Constraint monitoring (all six runs): clipping events = 0, reserve events = 0 — every requested
action lay inside bounds and inside the SoC reserve logic. Comfort-band violations (> setpoint +
2 °C) mirror the hot-discomfort flip above (B0: 154/706 violation-hours dev/final; B1/B2: 0).
Grid-limit exceedances vs `P_grid,max` = 1.8084: B0 9 (dev) / 38 (final); B1 101 / 425;
B2 100 / 417 — the extra cooling load pushes net demand above the data-derived reference limit
for most daytime hours under both active controllers.

Implication carried forward (no action taken this phase): September's learned policy should treat
the cooling-device dimension cautiously (it dominates energy cost) and use storage margins more
conservatively than the ±0.5 fixed levels; the CMDP reward/constraint definitions in
`docs/reference/cmdp-spec.md` already encode these signals.

## 4. Artifact index

- Per-run evidence: `results/runs/baselines/<controller>/<window>/` with `run_metadata.json`,
  `trace.csv`, `district_kpis.csv`, `derived_metrics.json`, `README.md` for
  {b0_neutral, b1_fixed_schedule, b2_tariff_aware} × {dev, final}.
- Comparison: `results/tables/baseline_comparison.csv` (6 rows = 3 controllers × 2 windows);
  figures `results/figures/dev_*.png` and `results/figures/final_*.png` (cost-by-controller bar,
  48-hour net-demand overlay, electrical SoC trace, indoor temperature vs cooling setpoint —
  4 kinds × 2 windows).
- Regeneration: `./.venv/bin/python scripts/10_run_baselines.py --window dev final` then
  `./.venv/bin/python scripts/11_compare_baselines.py --window dev --window final`; phase gate:
  `./.venv/bin/python scripts/12_gate_week2.py`.
- Note: `results/runs/baselines/b0_zero_actions/0-167/` is the retained Phase-B gate-B harness-
  regression artifact (pre dev/final naming lock), and the earlier `0-167_*` figure prefix
  predates that lock; both are kept untouched (nothing under `results/` is hand-edited) and are
  superseded by the `b0_neutral/dev` run and `dev_*` figures.

## 5. Supervisor update

> The CityLearn environment is reproducible, the building-control problem is specified as a
> CMDP, and three deterministic baselines have comparative KPI evidence. PPO has not started;
> it will be evaluated only after these baselines and measurements are locked.
