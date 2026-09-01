# Risk-Aware Deep Reinforcement Learning for Building Energy Optimisation

> A simulation-first Capstone Project-I prototype for safe, forecast-informed building-energy
> control under uncertain demand, solar generation, and time-of-day electricity prices.

**Research period:** August – November 2026 · **CityLearn 2.5.0** · Python 3.9 · CPU only

**Current status:** Weeks 1–3 complete and verified — environment foundation, CMDP
specification, locked evaluation harness, B0/B1/B2 deterministic baselines, and a standard
PPO controller trained across seeds 42/43/44. Weeks 4 (probabilistic forecasting) and
5 (uncertainty-aware PPO) are fully specified in binding plans and awaiting execution.

**No performance or real-world savings claim has been made.** All results so far are
simulation evidence under one tariff profile.

| Want to… | Read |
| --- | --- |
| understand the project in full | [`docs/status/research-log.md`](docs/status/research-log.md) — the complete narrative record |
| see where we are now | [`docs/status/current-stage.md`](docs/status/current-stage.md) |
| see the numbers and findings | [`docs/status/results.md`](docs/status/results.md) |
| see blockers and known issues | [`docs/status/issues.md`](docs/status/issues.md) |
| understand the data | [`docs/reference/dataset.md`](docs/reference/dataset.md) |
| run everything | [Quickstart](#quickstart) below |

## Research questions

| ID | Question | Evidence required to answer it |
| --- | --- | --- |
| **RQ1** | Can forecast uncertainty improve the cost-comfort trade-off compared with a controller that uses only point forecasts? | Matched point-forecast and interval-aware PPO runs across identical seeds and scenarios. |
| **RQ2** | Can a safety shield reduce comfort, battery, and grid-limit violations during forecast error and high-demand events while preserving most cost savings? | Violation counts and duration, cost difference, and adverse-scenario traces with and without the shield. |
| **RQ3** | How robust is the proposed controller under changed tariffs, solar availability, occupancy patterns, and weather conditions? | Scenario-by-method results across fixed seeds, including worst-case outcomes rather than averages alone. |

## Research boundary

### In scope

- A reproducible CityLearn simulation prototype: one building, cooling-device and battery control.
- Rule-based (B0/B1/B2), standard PPO, and proposed uncertainty-aware safe-PPO comparisons.
- Probabilistic load/PV forecasts as point forecasts plus prediction intervals.
- Comfort, battery state-of-charge, action-bound, and grid-import safety monitoring.
- Cost, peak demand, comfort, constraint, solar-self-consumption, and robustness evaluation.

### Explicitly out of scope for CP-I

- Physical building deployment or autonomous actuation.
- EV charging, multi-building coordination, a new control platform/runtime.
- Any claim that a simulator saving transfers to a real building.
- Any claim of safety based solely on a reward penalty.

India's Time-of-Day policy context motivates the problem, but the tariff here is a controlled
simulation input, not a model of a specific state tariff.

## What is implemented

- [x] Pinned, reproducible environment (CityLearn 2.5.0, exact source commit, local data bootstrap).
- [x] Derived single-building schema (`configs/schema-building1.json`) — Building_1 only, central agent.
- [x] Formal CMDP specification with frozen constants ([`docs/reference/cmdp-spec.md`](docs/reference/cmdp-spec.md)).
- [x] Locked evaluation harness — controller interface, runner, artifacts, metrics — validated by a B0 regression against the smoke anchors at 1e-9.
- [x] Deterministic baselines B0 (neutral), B1 (fixed schedule), B2 (tariff-aware), evaluated on the locked dev (0–167) and final (0–719) windows.
- [x] Standard PPO (seeds 42/43/44, frozen `configs/week3-ppo.yaml`), checkpoint evaluation through the locked harness, frozen selection rule, final-window evaluations.
- [x] Phase gates `05/12/25_gate_week*.py` — artifacts, regressions, config hashes, byte-identity of prior evidence.
- [ ] Week 4 — probabilistic forecasting module ([`docs/plans/week4-implementation-plan.md`](docs/plans/week4-implementation-plan.md)).
- [ ] Week 5 — uncertainty-aware PPO matched comparison, RQ1 verdict ([`docs/plans/week5-implementation-plan.md`](docs/plans/week5-implementation-plan.md)).
- [ ] Safety shield, ablations, robustness matrix (October). Dashboard and manuscript (November).

## Quickstart

**Use the system Python 3.9** (`/usr/bin/python3` on macOS). Homebrew's `python3` is newer
(3.14) and cannot install these pinned wheels (matplotlib 3.9.4 has no 3.14 build).
Run everything from this repository root.

```bash
/usr/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Foundation (Week 1)
python scripts/01_fetch_pinned_dataset.py          # verify pinned dataset (--fetch to re-extract)
python scripts/02_derive_building_schema.py             # derive the single-building schema
python scripts/03_inspect_environment.py     # export interface evidence
python scripts/04_run_smoke_test.py               # deterministic zero-action smoke run
python scripts/05_gate_week1.py            # Week 1 phase gate

# Deterministic baselines (Week 2)
python scripts/10_run_baselines.py           # B0/B1/B2 on both locked windows
python scripts/11_compare_baselines.py --window dev --window final
python scripts/12_gate_week2.py            # Week 2 phase gate

# Standard PPO (Week 3) — ~5 min per seed on CPU
python scripts/21_train_ppo.py --config configs/week3-ppo.yaml --seed 42   # then 43, 44
python scripts/22_evaluate_checkpoints.py --seed 42                        # then 43, 44
python scripts/23_evaluate_final_window.py
python scripts/24_compare_ppo.py
python scripts/25_gate_week3.py            # Week 3 phase gate

python -m pytest -q                          # 60 tests
```

Generated evidence lands under `results/` (git-ignored; always regenerable from
`src/ + configs/` — that pair is the reproducibility contract).

## Repository layout

```text
docs/        plans/ (what we intend), reference/ (how things work), status/ (where we are)
src/         the library — environment, baselines, evaluation harness, RL adapter
configs/     frozen experiment definitions (schema, week2 baselines, week3 PPO)
scripts/     runnable commands, numbered by phase (01–05 setup, 10–12 week 2, 20–25 week 3)
tests/       60 contract tests, incl. the B0-anchor 1e-9 regression
results/     generated evidence — runs/, tables/, figures/ — git-ignored
data/raw/    the pinned citylearn_challenge_2023_phase_1 dataset — read-only, git-ignored
automation/  conductor wrappers and the week-4→5 chain script
```

Folder-by-folder detail: [`docs/reference/folder-map.md`](docs/reference/folder-map.md).

## Reproducibility rules

Every experiment records, before its results are usable: dataset/schema and CityLearn
version; controller variant and hyperparameters; forecast variant; scenario, window, and
seed; reward terms and every shield/fallback event; git commit and output directory.
Constants are frozen in configs before runs and never silently edited — any change
requires a new named config. Nothing under `results/` is hand-edited; regenerate from
scripts only. See [`docs/reference/experiment-protocol.md`](docs/reference/experiment-protocol.md).

## Limitations and responsible use

- CityLearn results are simulation evidence, not a deployment recommendation.
- The dataset is not an India-specific building or tariff dataset; India is motivation, not a calibration target.
- The B0/B1/B2 comparison is single-seed heuristic evidence; the PPO comparison is three-seed evidence under one tariff profile. Neither is a savings claim — and PPO did **not** beat the do-nothing B0 on cost. Full findings: [`docs/status/results.md`](docs/status/results.md).
- PPO is the initial algorithm, not an assumption of superiority; it must earn its inclusion through the evaluation contract.
- BOPTEST is a credible future physics-based cross-check, but is not part of CP-I.
