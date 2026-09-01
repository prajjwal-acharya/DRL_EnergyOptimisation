# Week 2 Implementation Plan: CMDP Formulation and Deterministic Baselines

This document is the binding, deterministic implementation spec for the Week 2 phase.
It is written to be executed top-to-bottom by an autonomous worker with no other context.
Phases must be completed in order A → B → C → D. Each phase has an acceptance gate that
must pass before the next phase starts.

Working directory for every command is `code/`. Python is `./.venv/bin/python`
(Python 3.9.6 — all new code must be 3.9-compatible: no PEP 604 `X | Y` type unions,
no `match` statements; use `typing.Optional`/`Tuple` or `from __future__ import annotations`).

---

## 0. Ground truth (verified facts — do not re-derive, do not contradict)

Environment contract:

- CityLearn `2.5.0` pinned to source tag `v2.5.0`, commit `29062af6d077409e1c37a3e53a6cac30fd4d02bc`.
- Dataset `citylearn_challenge_2023_phase_1`, derived single-building schema:
  `configs/schema-building1.json` (Building_1 only, central agent).
- Episode: 720 hourly steps (30 days). Development window: steps 0–167 (7 days).
  Final baseline-evaluation window: steps 0–719. Random seed for all runs: `42`.
- Building state evolves via CityLearn's trained dynamics model (`Building_1.pth`).

Action space (3 dims, order is fixed):

| Action | Bounds | Semantics |
| --- | --- | --- |
| `dhw_storage` | `[-1, 1]` | negative = charge DHW tank, positive = discharge |
| `electrical_storage` | `[-1, 1]` | negative = charge battery, positive = discharge |
| `cooling_device` | `[0, 1]` | ratio of cooling-device nominal electrical input (NOT a setpoint) |

Assets:

- PV: 2.4 kW nominal.
- Battery: 4.0 kWh capacity, 3.32 kW nominal power, 0.95 efficiency, 0.8 depth of discharge.
- DHW tank: 2.2827 kWh capacity, loss coefficient 0.0032.

Observations: Building_1 exposes 29 active observations; the central agent receives a
49-dim combined vector. The exact ordered name list is in
`results/inspection/citylearn_2023_phase_1.json` → `first_building.observations`.

Tariff structure (extracted from `data/raw/.../pricing.csv` — deterministic, 3 levels):

| Level | Price (USD/kWh) | Hours | Timesteps |
| --- | --- | --- | --- |
| off-peak | 0.02893 | 0–7, 23 | 405 |
| mid | 0.02915 | 8–15, 19–22 | 252 |
| peak | 0.05867 | 16, 17, 18 | 63 |

Max-margin threshold between mid and peak bands: **τ = 0.0439 USD/kWh** (frozen constant).

B0 anchor (zero actions, window 0–167, seed 42 — from `results/runs/smoke/district_kpis.csv`):

- `cost_total = 0.44198876839332574`
- `all_time_peak_average = 0.8618364154405324`
- `electricity_consumption_total = 0.464085898307736`
- `discomfort_proportion = 0.9151515151515152`
- `ramping_average = 0.8571830450575444`
- `zero_net_energy = 0.35004620158879785`
- Note: `one_minus_thermal_resilience_proportion` and
  `power_outage_normalized_unserved_energy_total` are empty (no outages in this dataset) —
  exclude them from comparisons.

Existing code to build on (do not duplicate):

- `src/energy_optimisation/environment.py`: `load_environment`, `describe_space`,
  `create_single_building_schema`, `neutral_actions`, `inspect_environment`.
- `scripts/05_gate_week1.py`, `tests/test_environment.py` (2 tests, must keep passing).

Frozen constants (single source of truth — put in `configs/week2-baselines.yaml`, never hard-code in code):

| Constant | Value | Status |
| --- | --- | --- |
| τ (B2 price threshold) | 0.0439 USD/kWh | data-derived, frozen |
| ΔT comfort band | 2.0 °C | research assumption |
| SoC reserve band | [0.2, 0.9] | research assumption |
| Reward weights w_E, w_P, w_C | 1.0, 1.0, 10.0 | declared now, consumed by PPO in September only |
| B1/B2 hour bands and action levels | per §C below | frozen at first full run |
| Seed | 42 | frozen |
| Dev / final windows | 0–167 / 0–719 | frozen |

---

## Phase A — CMDP formulation (documentation + one small module)

### A1. Observation index map

Create `src/energy_optimisation/observation_names.py`:

- A function `build_observation_index(schema_path) -> "OrderedDict[str, int]"` that loads the
  schema, replicates CityLearn's central-agent observation ordering (shared observations first,
  then building-specific, following `shared_in_central_agent` in the schema's `observations`
  section and CityLearn 2.5.0 semantics), and maps each of the 49 observation names to its index.
- Export a frozen module-level dict for the Building_1 schema so controllers never use magic indices.
- Test (`tests/test_observation_names.py`): the derived index must reproduce the positions implied
  by `results/inspection/citylearn_2023_phase_1.json` and must be consistent with the
  loaded environment's observation space size (49).

### A2. Formal CMDP specification

Create `docs/reference/cmdp-spec.md` with exactly these sections:

1. **State** — table of all 29 Building_1 observations: symbol, name, unit, group
   (calendar / weather-solar / tariff / building / storage), source. State the state vector is
   the 49-dim central-agent observation.
2. **Actions** — the action table from §0 with sign conventions and physical meaning.
3. **Transition** — P(s_{t+1} | s_t, a_t) is given by the CityLearn dynamics model at 1-hour
   steps; no real-building transition claim is made.
4. **Reward** (declared now, consumed only by PPO in September; frozen before training):

   ```
   r_t = − w_E · (E_t / Ē_B0) − w_P · max(0, E_t − P_ref) / P_ref − w_C · D_t
   ```

   - `E_t` = `net_electricity_consumption` at step t.
   - `Ē_B0` = mean E_t of the B0 run on the dev window (compute once, record the value).
   - `P_ref` = max E_t of the B0 run on the dev window (compute once, record the value).
   - `D_t` = max(0, T_in − (T_set + ΔT)) in °C, ΔT = 2.0.
   - Weights w_E = 1.0, w_P = 1.0, w_C = 10.0.
   - Rule: any later change requires a new named config, never a silent edit.
5. **Constraints** (separate from reward penalties — each with measurement method):
   - Comfort: `T_in ≤ T_set + 2.0 °C`; measured by discomfort KPIs + per-step trace.
   - SoC reserve: requests that would push SoC outside [0.2, 0.9] are counted as reserve events;
     CityLearn enforces physical bounds internally, we count *requested* violations.
   - Action clipping: any requested action outside the Box bounds before clipping is counted.
   - Grid import limit: `P_grid,max` = 95th percentile of B0 net consumption on the dev window
     (compute once, record the value); document as a data-derived assumption.
6. **KPI mapping** — mark primary KPIs (`cost_total`, `all_time_peak_average`,
   `electricity_consumption_total`, `discomfort_hot_*`, `discomfort_proportion`, `ramping_average`,
   `zero_net_energy`) vs contextual; exclude the two empty KPIs listed in §0.

**Acceptance gate A:** every state variable, action, reward term, and constraint has a table row
with unit + measurement method; every non-CityLearn threshold is marked "research assumption"
with a one-line justification; `Ē_B0`, `P_ref`, `P_grid,max` have concrete recorded values
(fill them in during Phase D when B0 first runs, or mark "pending first B0 run" and update then).

---

## Phase B — Common evaluation harness (before any controller is written)

### B1. Controller interface

`src/energy_optimisation/baselines/controllers.py`:

```python
class Controller(ABC):
    name: str
    def reset(self, seed: int) -> None: ...
    def act(self, observation: np.ndarray) -> np.ndarray:  # shape (3,), dtype float
        ...
```

All controllers (and September's PPO wrapper) implement exactly this interface.

### B2. Runner

`src/energy_optimisation/evaluation/runner.py` — one loop for every controller:

- Load env via `environment.load_environment(schema_path)`; support
  `simulation_start_time_step`/`simulation_end_time_step` overrides for windows.
- `reset(seed=42)`; step to terminal; at each step record: timestep, hour, electricity price,
  action requested (pre-clip), action applied (post-clip), net electricity consumption, both SoCs,
  indoor temperature, cooling setpoint.
- Call `env.evaluate()`; return `(kpis: dict, trace: pd.DataFrame)`.
- Record resolved config + seed + git commit (`git rev-parse HEAD`) — paths relative to project
  root, never absolute machine paths.

### B3. Artifacts

`src/energy_optimisation/evaluation/artifacts.py` — write to
`results/runs/baselines/<controller>/<window>/`: `run_metadata.json`, `trace.csv`,
`district_kpis.csv`, `README.md` run note (purpose: research result, controller, seed, window).

### B4. Derived metrics

`src/energy_optimisation/evaluation/metrics.py` — from traces: comfort-violation hours,
SoC min/max, clipping-event count, reserve-event count, peak kW, solar self-consumption,
grid-limit exceedance count vs `P_grid,max`.

### B5. Comparison report

`scripts/11_compare_baselines.py` — reads all run dirs for a window and writes
`results/tables/baseline_comparison.csv` (rows = controllers, columns = fixed KPI set) and
`results/figures/`: (1) cost-by-controller bar, (2) 48-hour net-demand overlay of all controllers,
(3) electrical SoC trace, (4) indoor temperature vs cooling setpoint trace.

### Phase B tests (`tests/test_runner.py`)

- `test_run_produces_all_artifacts`
- `test_identical_seed_reproducible_kpis` (two runs → identical KPI dict)
- `test_trace_has_no_nan`
- `test_b0_matches_smoke_kpis` — B0 on window 0–167, seed 42 must reproduce the §0 anchor values
  exactly (tolerance 1e-9). This validates the harness before B1/B2 exist.

**Acceptance gate B:** all Phase B tests pass with B0 only; existing week-1 tests still pass.

---

## Phase C — Controllers, in order

### C0. `baselines/neutral.py` — B0

`act` returns `[0.0, 0.0, 0.0]` every step (reuse `environment.neutral_actions`). Run through the
new harness on the dev window; confirm the B0 regression test passes.

### C1. `baselines/fixed_baseline.py` — B1 (calendar-only)

Reads ONLY `hour` (via the observation index map). Must not read price or forecast observations.
Constants in config:

- `electrical_storage`: hours 0–5 → −0.5 (charge); hours 17–20 → +0.5 (discharge); else 0.
- `dhw_storage`: hours 0–5 → −0.5; else 0.
- `cooling_device`: hours 12–20 → 0.8; hours 8–11 and 21–22 → 0.5; else 0.2.

The schedule is deliberately fixed: B1 cannot adapt if tariffs change (this is the contrast
that later supports RQ3).

### C2. `baselines/tariff_aware_baseline.py` — B2 (current-price heuristic)

Reads ONLY `electricity_pricing`, `hour`, `electrical_storage_soc`, `dhw_storage_soc`.
No forecasts, no lookahead, no optimisation — an interpretable heuristic only.

- If `p_t ≥ τ = 0.0439` (peak band):
  - `electrical_storage` = +0.5 if `electrical_storage_soc ≥ 0.2`, else 0.
  - `dhw_storage` = +0.5 if `dhw_storage_soc ≥ 0.2`, else 0.
  - `cooling_device` = 0.6 (comfort-protective).
- If `p_t < τ`:
  - `electrical_storage` = −0.5 if `electrical_storage_soc ≤ 0.9`, else 0.
  - `dhw_storage` = −0.5 if `dhw_storage_soc ≤ 0.9`, else 0.
  - `cooling_device` = same hour bands as B1.

### Phase C tests (`tests/test_baselines.py`)

- `test_all_controllers_action_shape_and_bounds` (random observations, many seeds)
- `test_b0_is_always_zero`
- `test_b1_ignores_price_signal` — varying the price observation must not change B1's action
- `test_b2_discharges_above_threshold`, `test_b2_charges_below_threshold`,
  `test_b2_respects_soc_reserve` (SoC < 0.2 in peak → action 0)
- `test_controllers_deterministic_under_seed`
- `test_baselines_do_not_import_forecasting_or_rl` — scan `baselines/` sources for
  `forecasting`, `stable_baselines3`, `torch` imports; must be absent

**Acceptance gate C:** all tests pass; three controllers run to terminal on the dev window
with zero unclipped invalid actions (every requested action either in bounds or clipped + logged).

---

## Phase D — Lock, run, verify, document, commit

### D1. Config lock

`configs/week2-baselines.yaml`: schema path, dev/final windows, seed, every C0–C2 constant,
τ, reserve band, reward weights, `P_grid,max` placeholder, output dirs. After the first full
final-window run this file is frozen; any constant change creates `week2b.yaml` instead.

### D2. Execution order

1. B0 dev (0–167) → B0 regression vs smoke anchors.
2. B1 dev → B2 dev. Inspect traces and the comparison table.
3. Fix **code bugs only**. Never tune constants to improve results — a worse B2 result is a
   recorded finding, not a failure to fix. Rerun all three together after any fix.
4. Final runs: B0, B1, B2 on window 0–719 with the identical config.
5. `scripts/11_compare_baselines.py` for both windows; record `Ē_B0`, `P_ref`, `P_grid,max`
   values into `docs/reference/cmdp-spec.md` (fill the placeholders).

### D3. Verification

`scripts/12_gate_week2.py` (mirror the structure of `scripts/05_gate_week1.py`). It must check,
each as a hard pass/fail with a clear message:

- `./.venv/bin/python -m pytest -q` passes.
- All three controllers have complete artifact sets for BOTH windows.
- `results/tables/baseline_comparison.csv` exists with exactly 3 controller rows.
- At least 4 comparison figures exist.
- B0 dev-window KPIs match the §0 anchors (tolerance 1e-9).
- No NaN in any trace; clipping/reserve event counts present in metrics.
- `docs/reference/cmdp-spec.md` exists with all 6 sections and no unfilled placeholders.
- `configs/week2-baselines.yaml` exists; baselines/ contains no forbidden imports.
- `docs/status/phase-reviews/week2-review.md` exists.

### D4. Documentation and commit

- `docs/status/phase-reviews/week2-review.md`: separate infrastructure evidence from controller-performance
  results; record negative outcomes (e.g., if B2 costs more than B1, analyse, do not tune);
  include the supervisor update verbatim:
  > The CityLearn environment is reproducible, the building-control problem is specified as a
  > CMDP, and three deterministic baselines have comparative KPI evidence. PPO has not started;
  > it will be evaluated only after these baselines and measurements are locked.
- Single commit closing the phase, e.g. `feat: week-2 cmdp specification and deterministic baselines`.

**Acceptance gate D:** `scripts/12_gate_week2.py` exits 0; repo green; all changes committed.

---

## Guardrails (binding for the worker)

- **Out of scope:** no PPO/RL training, no forecasting models, no safety shield, no EV,
  multi-building, or deployment work. Do not modify `src/energy_optimisation/forecasting/`
  or `safety/` beyond their `.gitkeep`.
- Never edit anything under `data/raw/` or hand-edit anything under `results/`.
- Never install new dependencies; everything needed is in `.venv` (CityLearn 2.5.0, pandas,
  numpy, matplotlib, pytest, stable-baselines3 — the last must remain unused this phase).
- Constants live in config, not code. No magic numbers, no hard-coded paths or seeds in code.
- If a command or run fails: retry once; if it fails again with the same error, record the
  error in the run note / progress doc and treat it as a blocker rather than working around it.
- Do not weaken or skip a failing test to make progress — fix the code or document the blocker.
- The two empty KPIs (`one_minus_thermal_resilience_proportion`,
  `power_outage_normalized_unserved_energy_total`) are expected to be empty; exclude them.

## Definition of done (mirrors the mission's done criteria)

1. `docs/reference/cmdp-spec.md` complete per gate A with recorded `Ē_B0`, `P_ref`, `P_grid,max`.
2. B0, B1, B2 implemented, evaluated on windows 0–167 and 0–719, artifacts under `results/runs/baselines/`.
3. `results/tables/baseline_comparison.csv` + ≥4 comparison figures exist and are regenerable
   from `scripts/11_compare_baselines.py`.
4. `./.venv/bin/python -m pytest -q` passes (week-1 tests included).
5. `./.venv/bin/python scripts/12_gate_week2.py` exits 0.
6. `docs/status/phase-reviews/week2-review.md` written; phase committed.
