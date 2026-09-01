# Week 1 Progress Against the CP-I Research Plan

**Reporting date:** 18 August 2026

**Project:** *Risk-Aware Deep Reinforcement Learning for Building Energy Optimisation under Uncertain Demand, Solar Generation, and Time-of-Day Tariffs*

**Plan reference:** `Prajjwal_Acharya_CP-I_Research_Plan_Energy_Optimisation.pdf`

**Reporting boundary:** This report covers the completed Week 1 foundation work only. It distinguishes verified infrastructure work from research results. No PPO policy, forecast model, safety shield, tariff-aware baseline, or performance comparison has been trained or evaluated yet.

## 1. Week 1 outcome

Week 1 established a reproducible, simulation-first starting point for the CP-I project. The repository, research questions, CityLearn environment, one-building schema, end-to-end smoke simulation, literature evidence base, and verification checks are complete. The environment can now be reset, supplied with valid actions, stepped to its configured terminal state, and evaluated for CityLearn KPIs.

This work directly supports the research plan's August objectives to review literature, define the research questions, formulate the environmental decision boundary, and set up a reproducible CityLearn-based environment. The August rule-based baseline and the formal CMDP mathematics are the next work items; they are not claimed as complete in this report.

## 2. Research-question alignment

The three research questions in the approved plan have been copied into the repository without changing their meaning:

| Research question | Week 1 progress | Evidence created |
| --- | --- | --- |
| **RQ1:** Can forecast uncertainty improve the cost-comfort trade-off compared with point forecasts? | Question fixed; evaluation approach and literature gaps defined. No forecast model or PPO comparison yet. | `README.md`; `docs/reference/literature-matrix.csv`; `docs/reference/literature.md` |
| **RQ2:** Can a safety shield reduce constraint violations during forecast errors and high-demand events while preserving cost savings? | Question fixed; safe-RL literature and candidate constraints identified. No shield, constraints implementation, or violation experiment yet. | `README.md`; `docs/reference/literature.md`; `docs/reference/environment-selection.md` |
| **RQ3:** How robust is the controller under changed tariffs, solar, occupancy, and weather conditions? | Question fixed; a future scenario matrix and worst-case reporting requirement defined. No robustness scenarios run yet. | `README.md`; `docs/reference/literature.md`; `docs/reference/experiment-protocol.md` |

## 3. Progress against the August work plan

The research plan assigns the following work to August: literature review; research-question definition; state, action, reward, and constraint formulation; CityLearn exploration; and a fixed-schedule/tariff-aware baseline.

| Planned August activity | Current status | What was completed in Week 1 | Remaining work |
| --- | --- | --- | --- |
| Literature review | **Foundation complete** | Created a 12-source matrix across building-energy RL, probabilistic forecasting, safe RL, and tariff-aware demand response. Six sources received full extraction and all analysed sources map to one or more RQs. Four testable gaps were recorded. | Fully extract the remaining six screened sources; expand the review as methods are selected. |
| Define research questions | **Complete** | RQ1-RQ3, scope, non-goals, expected evidence, and metrics are documented in the README. | Keep the questions fixed unless a supervisor approves a change. |
| Define state and action boundary | **Partially complete** | Inspected the CityLearn interface and selected a one-building `Building_1` scenario. The central agent has one observation vector and three valid actions: DHW storage, electrical storage, and cooling device. | Convert the observed variables and action bounds into the formal CMDP state/action notation. |
| Define reward and constraints | **Not yet complete** | Identified the required measurement and safety categories: cost, peak demand, comfort, battery state of charge, action bounds, and grid import. | Specify reward components, exact limits, penalties if any, and hard shield/fallback logic. |
| Install and explore CityLearn | **Complete** | Installed and pinned CityLearn 2.5.0, inspected a suitable scenario, documented the interface, and created reusable utilities. | Continue using the same pinned environment for baseline development. |
| Reproducible simulator configuration | **Complete** | Pinned CityLearn source tag `v2.5.0` and commit `29062af6d077409e1c37a3e53a6cac30fd4d02bc`; created a versioned one-building schema and fixed smoke-run configuration. | Add baseline and later experiment configurations without changing this provenance record. |
| Fixed-schedule baseline | **Not started** | None; deliberately deferred until the environment and measurement path were proved. | Implement during Week 2. |
| Tariff-aware rule-based baseline | **Not started** | None; tariff signals are confirmed in the selected scenario. | Implement during Week 2, then compare fairly with fixed schedule. |
| Baseline result plots | **Not started** | A smoke-run KPI plot is generated only as output-path evidence. It is not a baseline result plot. | Produce baseline cost, peak, comfort, and safety plots after baseline implementation. |

## 4. Completed technical deliverables

### 4.1 Reproducible repository and dependency environment

The project repository is organised so reusable code, configuration, evidence, and generated data are separated:

- `src/energy_optimisation/` contains reusable environment utilities and reserved modules for baselines, forecasting, safety, and evaluation.
- `configs/` contains the fixed smoke-run configuration and the derived one-building schema.
- `scripts/` contains repeatable bootstrap, inspection, schema-generation, smoke-run, and verification commands.
- `docs/` contains the initial formulation, protocol, literature evidence, and progress reports.
- `data/raw/` and `results/` are excluded from Git because they hold downloaded source data and generated evidence.

The Python environment has been verified with CityLearn `2.5.0`. The first project snapshot was committed as:

```text
d00ba1b chore: establish CP-I research foundation
```

### 4.2 CityLearn environment selection and inspection

The original CityLearn 2020 climate-zone example was inspected and rejected for CP-I because it has nine buildings, inactive electricity-price observations, and no cooling-device action. The selected parent scenario is `citylearn_challenge_2023_phase_1`, which has three buildings and 720 hourly timesteps.

For CP-I scope control, a derived schema retains `Building_1` only. It exposes the signals required to begin the planned work:

- Weather and solar-irradiance observations with three-step forecasts.
- Indoor temperature, cooling demand, DHW demand, occupancy, non-shiftable load, solar generation, battery state of charge, and net electricity consumption.
- Electricity-price observation with three-step price forecasts.
- A 2.4 kW nominal PV asset and 4.0 kWh electrical-storage capacity.
- Three controllable actions: DHW storage, electrical storage, and cooling device.

This aligns with the plan's HVAC-related and battery flexibility scope, but it is important to be precise: the selected schema exposes **cooling-device control**, not a direct thermostat/setpoint action. The formal CMDP and baseline design must respect this actual interface rather than assume a direct setpoint action exists.

### 4.3 End-to-end smoke simulation

The initial simulation proves the complete technical path:

```text
load one-building schema -> reset environment -> create valid zero actions
-> step to termination -> evaluate district KPIs -> write evidence
```

The checked smoke configuration uses seed `42`, a `0-167` evaluation window, and a deterministic zero-action controller. It completed 167 simulator actions, reached the terminal state without truncation, and wrote:

- `results/runs/smoke/run_metadata.json`
- `results/runs/smoke/district_kpis.csv`
- `results/runs/smoke/district_kpis.png`
- `results/runs/smoke/README.md`

The output is evidence that the environment, actions, terminal handling, KPI evaluation, and output directory work. It is explicitly **not** an energy-saving, cost, comfort, or controller-performance result.

### 4.4 Research documentation and experimental discipline

The README records the project definition, in-scope assets, explicit non-goals, RQ1-RQ3, evaluation measures, limitations, and future controller sequence. The initial experimental protocol requires every later experiment to record the schema/version, controller, configuration, seed, time ranges, output directory, and Git commit identifier.

This implements the plan's simulation-first feasibility and risk-management approach: comparisons will be made only after their configuration and measurement path are reproducible.

### 4.5 Literature evidence base

The literature review has a deliberately practical purpose: it informs the implementation and evaluation rather than claiming a completed systematic review.

| Topic | Sources in matrix | Fully analysed | Week 1 use |
| --- | ---: | ---: | --- |
| Building-energy RL | 4 | 2 | Simulator choice, control boundary, and fair baselines |
| Probabilistic forecasting | 4 | 2 | Point/quantile forecasts, interval coverage, and width metrics |
| Safe RL | 3 | 2 | CMDP terminology, action correction, and explicit violation reporting |
| Tariff-aware demand response | 2 | 1 | Tariff-scenario motivation and cost/peak evaluation boundary |

The review identifies four testable gaps that become the planned experiment design:

1. Compare point-only and interval-aware controller inputs under the same PPO architecture and scenarios.
2. Add auditable comfort, state-of-charge, action-power, and grid-import checks rather than relying on reward penalties alone.
3. Report scenario spread and worst-case outcomes, not only average performance.
4. Establish fixed-schedule and tariff-aware rule baselines before attributing gains to PPO or uncertainty modelling.

## 5. Progress against the November deliverables

The approved plan defines these as end-of-semester deliverables. Week 1 contributes to them but does not complete them.

| End-of-semester deliverable | Week 1 contribution | Current status |
| --- | --- | --- |
| Documented CMDP with reward, bounds, and explicit constraints | Environment choice, observed signals, controllable actions, and target safety categories are documented. | **In progress** |
| Reproducible CityLearn prototype with rule-based, standard DRL, and risk-aware DRL controllers | Reproducible CityLearn configuration, utilities, and smoke path exist. | **Foundation complete; controllers not started** |
| Demand/solar forecaster with point and interval evaluation | Evaluation intent and literature-backed metrics are specified. | **Not started** |
| Cost, peak, comfort, constraint, and robustness comparisons | Metric contract and output discipline are specified. | **Not started** |
| Technical report or paper draft | README, CMDP note, protocol, and literature review form initial material. | **Early documentation only** |
| Lightweight dashboard or demonstration | Output CSV and KPI plot establish a future data path. | **Not started** |

## 6. Verification completed

The following checks have been executed successfully from the project environment:

```bash
python scripts/foundation/01_fetch_pinned_dataset.py --skip-clone
python scripts/foundation/02_derive_building_schema.py
python scripts/foundation/03_inspect_environment.py
python scripts/foundation/04_run_smoke_test.py
python -m pytest -q
python scripts/foundation/05_gate_week1.py
```

Verification results:

- CityLearn version: `2.5.0`.
- Project tests: `2 passed`.
- Week 1 verifier: passed repository, documentation, literature, environment-interface, and smoke-evidence checks.
- Smoke run: terminal state reached after 167 actions; expected outputs written.

## 7. Implementation note and limitation

CityLearn's named-dataset download route depends on GitHub's anonymous API. It was rate-limited in this environment, so the project does not claim that an anonymous API request succeeded. Instead, it uses the exact CityLearn source tag and commit, derives the CP-I schema locally, and primes the required CityLearn cache files. The locally pinned schema reset and smoke simulation are the reproducibility evidence.

The selected simulator scenario is also not an India-specific building or state tariff dataset. India-oriented Time-of-Day tariffs remain the motivation for controlled tariff scenarios; no CityLearn value will be presented as a real Indian tariff without separate data and validation.

## 8. Next planned work

The next implementation sequence follows the approved plan and the evidence review:

1. Complete the formal CMDP: state definition, actions and bounds, transition description, reward, and explicit comfort/storage/grid constraints.
2. Implement a fixed-schedule controller and a tariff-aware rule-based baseline.
3. Lock baseline configurations, seeds, metrics, and plots before starting PPO.
4. Then implement forecast baselines, followed by standard PPO, interval-aware PPO, and the safety-shield ablations.

This order keeps the project within CP-I scope and ensures that later claims answer RQ1, RQ2, or RQ3 through reproducible comparisons.
