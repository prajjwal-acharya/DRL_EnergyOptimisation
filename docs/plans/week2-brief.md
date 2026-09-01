# Week 2: Formal CMDP and Deterministic Baselines

## Objective

- Turn the CityLearn setup into a precise, measurable control problem.
- Establish reproducible deterministic baselines before any forecasting, PPO, or safety-shield work.
- Produce the first fair controller comparison: neutral control vs fixed rule-based control vs tariff-aware rule-based control.

## Scope boundary

- Use the pinned CityLearn `2.5.0` Building 1-only schema only.
- Keep the work simulation-only and single-building.
- Control only the exposed actions: `dhw_storage`, `electrical_storage`, and `cooling_device`.
- Treat `cooling_device` as cooling-device control; it is **not** a direct thermostat/set-point action.
- Do not start forecasting, PPO, safety-shield, multi-building, EV, or real-world-deployment work this week.

## Work plan

### 1. Freeze the experimental contract

- Define one named Week 2 baseline configuration.
- Record the schema path, CityLearn version, episode window, seed list, controller name, and output directory.
- Use exactly the same configuration for every baseline comparison.
- Maintain a fast development window and a separately documented final baseline-evaluation window.
- Do not change a configuration after seeing a result; create a new named configuration instead.

### 2. Write the formal CMDP specification

- Create `code/docs/reference/cmdp-spec.md` as the mathematical source of truth.
- Define the state `s_t` using only verified CityLearn observations:
  - calendar: hour and day type;
  - weather and solar signals: outdoor temperature, irradiance, and available short-horizon forecasts;
  - tariff signals: current electricity price and available price forecasts;
  - building state: indoor temperature, cooling demand, DHW demand, occupancy, non-shiftable load, PV generation, and net electricity consumption;
  - storage state: DHW-storage and electrical-storage state of charge.
- Define the action `a_t = [a_dhw, a_battery, a_cooling]` using actual action bounds:
  - DHW storage: `[-1, 1]`;
  - electrical storage: `[-1, 1]`;
  - cooling device: `[0, 1]`.
- Define the transition and episode horizon from CityLearn, without claiming a real-building model.
- Define a reward that reports, at minimum:
  - electricity cost;
  - peak-grid-import penalty;
  - comfort violation;
  - optional storage-cycling penalty.
- Define explicit CMDP constraints separately from reward penalties:
  - thermal-comfort violation duration/magnitude;
  - battery/DHW state-of-charge bounds;
  - action bounds and clipping events;
  - grid-import limit, if a defensible threshold is selected and documented.
- Add one table for each state variable, action, reward term, constraint, unit, range/source, and measurement method.
- Mark any threshold not directly provided by CityLearn as a research assumption and justify it before using it.

### 3. Define the common evaluation harness

- Create one reusable baseline runner and evaluator; do not duplicate episode/KPI logic across controllers.
- Start with the existing deterministic zero-action controller as the neutral reference (`B0`).
- For every run, save:
  - resolved configuration and random seed;
  - controller name and parameters;
  - per-step action trace and key state trace;
  - CityLearn KPI CSV;
  - compact run note;
  - reproducible output path and Git commit ID.
- Report the same metrics for every controller:
  - electricity cost;
  - peak grid import;
  - total electricity consumption;
  - hot/cold/total comfort violation;
  - battery and DHW state-of-charge range;
  - requested-action clipping or invalid-action count;
  - solar self-consumption when available from the KPI path.
- Create, at minimum:
  - one controller-comparison table;
  - one cost-by-controller plot;
  - one 24- or 48-hour grid-demand comparison plot;
  - one storage-state/comfort trace for interpretation.

### 4. Implement the fixed rule-based controller (`B1`)

- Create `code/src/energy_optimisation/baselines/fixed_baseline.py`.
- Use fixed calendar-based rules only; do not read electricity-price or forecast inputs.
- Use predetermined charge/discharge and cooling/DHW-control periods.
- Clip every proposed action to CityLearn's action space before stepping the environment.
- Keep the policy intentionally simple, deterministic, documented, and reproducible.
- Compare `B1` directly against neutral control (`B0`) using the common evaluation harness.

### 5. Implement the tariff-aware rule-based controller (`B2`)

- Create `code/src/energy_optimisation/baselines/tariff_aware_baseline.py`.
- Use current electricity price only for the initial policy; do not use forecasts yet.
- Define and document a simple low-price/high-price threshold rule.
- Charge storage during low-price periods and discharge/reduce flexible consumption during high-price periods, subject to action bounds and recorded comfort/storage checks.
- Do not turn this baseline into an optimiser; it must remain an interpretable deterministic heuristic.
- Compare `B2` against both `B0` and `B1` with the same configuration and metrics.

### 6. Validate and document the result

- Add focused tests for:
  - action-vector order, shape, and bounds;
  - deterministic repeatability for a fixed seed and configuration;
  - common output files and comparison-table schema;
  - absence of forecast/PPO dependencies in `B1` and `B2`.
- Run the complete Week 2 verification sequence from `code/`:

  ```bash
  source .venv/bin/activate
  python -m pytest -q
  python scripts/05_verify_week1.py
  # Add and run the Week 2 baseline verifier once implemented.
  ```

- Record result interpretation, including negative outcomes.
- Treat higher cost, shifted peak demand, or increased discomfort as valid findings to analyse rather than silently tune away.

## Required deliverables

- `code/docs/reference/cmdp-spec.md` with the formal state, action, reward, constraints, horizon, and metrics.
- One locked baseline configuration and documented seed/evaluation-window policy.
- `B0`: neutral zero-action reference, evaluated through the common harness.
- `B1`: fixed rule-based controller, evaluated against `B0`.
- `B2`: tariff-aware rule-based controller, evaluated against `B0` and `B1`.
- One comparison table and at least two interpretable comparison plots.
- Focused automated tests and a Week 2 verification command/script.
- A short progress note that distinguishes infrastructure evidence from controller-performance results.

## Exit criteria

- CMDP specification is complete and uses the actual CityLearn interface.
- All three baseline controllers run to the configured terminal state without invalid actions.
- Each controller has results from the same schema, time window, seeds, metrics, and output format.
- Results and plots can be regenerated from configuration without manual edits.
- The comparison makes cost, peak, comfort, and constraint trade-offs visible.
- The repository remains passing before the work is considered complete.

## Expected supervisor update

> The CityLearn environment is reproducible, the building-control problem is specified as a CMDP, and three deterministic baselines have comparative KPI evidence. PPO has not started; it will be evaluated only after these baselines and measurements are locked.

## Next phase after Week 2

- Build and evaluate demand/PV forecasting baselines, including point accuracy and interval calibration.
- Train standard PPO against the locked deterministic baselines.
- Only then integrate uncertainty inputs, add the safety shield, and run ablations and robustness tests.
