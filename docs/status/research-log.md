# Research Log — Complete State of the Project

> **Provenance note:** this log was maintained in the project's previous workspace (`/Volumes/code/Research Project`) and migrated here unchanged apart from path updates. References to `../.agent-conductor/` and `../conductor/` point to that previous location, and the repository layout sketched in §2.4 is that workspace's; the current layout is [`docs/reference/folder-map.md`](../reference/folder-map.md).

**Last updated:** 1 September 2026
**Maintainer note:** this is the single narrative document that ties together *everything* done so far — setup, implementations, experimental observations, inferences, anomalies, and current status. Per-phase details live in the weekly review/plan docs (§10 index); this log is the layer above them and must be updated whenever a phase completes or blocks.

**One-paragraph summary.** This is a CP-I capstone project building a risk-aware deep-RL controller for building energy optimisation in CityLearn. Weeks 1–3 are complete, verified, and committed: environment foundation, CMDP specification, a locked evaluation harness, three deterministic baselines (B0/B1/B2), and a standard PPO controller trained across three seeds. The headline empirical result so far is honest and negative-leaning: PPO halves discomfort versus every baseline but does **not** beat the do-nothing controller on cost, and no controller respects the data-derived grid-import limit. Weeks 4 (probabilistic forecasting) and 5 (uncertainty-aware PPO for RQ1) are fully specified in binding plans but **never executed** — the autonomous worker mission for week 4 blocked on an external model-API failure before writing a single file, and week 5 was chained behind it and never started. Resume instructions are in [`docs/status/phase-reviews/week4-5-status.md`](week4-5-status.md).

---

## 1. Project identity and governance

| Field | Value |
| --- | --- |
| Title | Risk-Aware Deep Reinforcement Learning for Building Energy Optimisation under Uncertain Demand, Solar Generation, and Time-of-Day Tariffs |
| Course | Capstone Project-I (CS4095), NIT Rourkela CSE, Autumn 2026-27 |
| Candidate | Prajjwal Acharya (123CS0143) |
| Research period | August – November 2026 |
| Approved plan | `../123CS0143_PrajjwalAcharya_ResearchPlan.pdf` / `../Prajjwal_Acharya_CP-I_Research_Plan_Energy_Optimisation.docx` |
| Code home | this repository (`code/`), git `master`, HEAD `0d6de3a`, working tree clean |

### 1.1 Research questions (fixed; supervisor approval required to change)

- **RQ1** — Can forecast uncertainty improve the cost-comfort trade-off compared with a controller that uses only point forecasts?
- **RQ2** — Can a safety shield reduce comfort, battery, and grid-limit violations during forecast error and high-demand events while preserving most cost savings?
- **RQ3** — How robust is the proposed controller under changed tariffs, solar availability, occupancy patterns, and weather conditions?

### 1.2 Approved month plan vs actual progress

| Month | Planned | Actual status |
| --- | --- | --- |
| August | Literature review, CMDP formulation, CityLearn setup, rule-based + tariff-aware baselines | ✅ Done (Weeks 1–2, plus Week 3 PPO pulled forward — completed 25–26 Aug) |
| September | Time-series features, load/PV forecasting model, standard PPO, measurement protocol | ⚠️ PPO done; **forecasting not started** (Week 4 mission blocked — infra failure, see §8) |
| October | Uncertainty in controller state, safety shield, forecast-noise/tariff/peak experiments | Planned (Week 5 spec written for the uncertainty part; shield = Week 6+) |
| November | Robustness testing, aggregation, report/manuscript, dashboard | Planned |

Mid-semester milestone (30 Sep): formulation + simulator + baselines + **initial forecast model**. The forecast model is the only missing piece; it is fully specified in `docs/plans/week4-implementation-plan.md` and waiting on execution.

### 1.3 Scope boundary (unchanged since Week 1)

In scope: one building, cooling-device + battery + DHW-storage control, CityLearn simulation only, probabilistic forecasts as point + intervals, comfort/SoC/action/grid safety checks, robustness evaluation.
Out of scope for CP-I: physical deployment, EV charging, multi-building coordination, real-world savings claims, safety claims based solely on reward penalties.

---

## 2. The setup

### 2.1 Machine and software environment

| Component | Value |
| --- | --- |
| Host | macOS 26.5 (darwin 25.5.0), arm64 |
| Python | 3.9.6 in `code/.venv` (all code must stay 3.9-compatible: no `X \| Y` unions, no `match`) |
| CityLearn | **2.5.0**, pinned to source tag `v2.5.0`, commit `29062af6d077409e1c37a3e53a6cac30fd4d02bc` |
| RL stack | stable-baselines3 2.3.2, torch 2.8.0 (transitive), gymnasium 0.28.1, **CPU only** |
| Other pins | pandas 2.3.3, numpy 1.26.4, matplotlib 3.9.4, jupyter 1.1.1, pytest 8.4.2 (`requirements.txt`) |
| Not installed | scikit-learn — deliberately absent; metrics are implemented in-repo in numpy |

### 2.2 Dataset and derived scenario

- Parent scenario: `citylearn_challenge_2023_phase_1` (3 buildings, 720 hourly steps). The 2020 climate-zone example was **explicitly rejected** in Week 1 (9 buildings, inactive price observations, no cooling-device action) — rationale in `docs/reference/environment-selection.md`.
- CP-I scenario: derived single-building schema [`configs/schema-building1.json`] — Building_1 only, `central_agent: true`, bounds 0–719, generated read-only from the parent (never editing raw data).
- Building_1 assets: PV 2.4 kW nominal; battery 4.0 kWh / 3.32 kW / η = 0.95 / DoD 0.8 (usable SoC band ⇒ starts at ≈ 0.2); DHW tank 2.2827 kWh (loss coeff 0.0032); heat pump ≈ 4.11 kW; dynamics = pretrained LSTMDynamicsBuilding (`Building_1.pth`, lookback 12).
- Data files (720 rows each, read-only): `Building_1.csv`, `weather.csv` (actuals + dataset-issued `*_predicted_1/2/3` forecasts at row t for t+1/2/3), `pricing.csv`, `carbon_intensity.csv`.
- Tariff (data-derived, `pricing.csv`): off-peak 0.02893 USD/kWh (hours 0–7, 23; 405 steps), mid 0.02915 (hours 8–15, 19–22; 252 steps), peak 0.05867 (hours 16–18; 63 steps). Frozen mid→peak threshold **τ = 0.0439 USD/kWh**.

### 2.3 Frozen experimental constants

| Constant | Value | Provenance |
| --- | --- | --- |
| Dev window | steps 0–167 (167 trace rows) | locked Week 2 |
| Final window | steps 0–719 (719 trace rows, evaluation-only) | locked Week 2 |
| Seeds | 42 (baselines, smoke), {42, 43, 44} (PPO) | locked |
| `Ē_B0` | 0.477229108554339 kWh | mean net consumption of B0 dev run |
| `P_ref` | 7.694016456604004 kWh | max net consumption of B0 dev run (hour 140) |
| `P_grid,max` | 1.8084184527397151 kWh | 95th percentile of B0 dev net consumption |
| Reward weights | w_E = 1.0, w_P = 1.0, w_C = 10.0 | research assumption, frozen before training |
| Comfort band ΔT | 2.0 °C above cooling setpoint | research assumption |
| SoC reserve band | [0.2, 0.9] | research assumption |

Single source of truth for all of these: `docs/reference/cmdp-spec.md` + `configs/week2-baselines.yaml`. Rule: any change requires a new named config, never a silent edit.

### 2.4 Repository layout

```text
code/
├── configs/                baseline.yaml (smoke), baselines/week2.yaml, ppo/week3.yaml,
│                           generated/…building_1.json   ← versioned experiment inputs
├── docs/                   all plans, reviews, specs, literature (§10 index)
├── notebooks/              exploration only, not source of truth
├── scripts/                every repeatable command + gate_week{1,2,3}.py phase gates
│                           + check_week{2..5}.sh conductor wrappers
├── src/energy_optimisation/
│   ├── environment.py      CityLearn loading / inspection / schema derivation
│   ├── observation_names.py  name→index observation mapping (no magic indices)
│   ├── baselines/          Controller ABC + B0/B1/B2
│   ├── evaluation/         runner (locked harness), metrics, artifacts
│   ├── rl/                 env_adapter (Gymnasium), PPOController, checkpoint_selection
│   ├── forecasting/        EMPTY (.gitkeep) — Week 4 target
│   └── safety/             EMPTY (.gitkeep) — Week 6+ target
├── tests/                  7 files, 60 passing tests
├── results/                git-ignored generated evidence (see §5)
└── data/raw/citylearn-2.5.0/  git-ignored pinned CityLearn v2.5.0 source
```

Outside `code/`: `Plans/week1.md`, `Plans/week2.md` (the original weekly briefs), the research-plan PDF/DOCX, and `.agent-conductor/missions/` + `conductor/` (the automation layer, §8).

### 2.5 Automation setup (agent-conductor)

Each implementation week is executed by an autonomous "conductor" mission driven by the binding plan doc. Missions live in `../.agent-conductor/missions/`: week 2 (`…-5cfebe`, completed), week 3 (`…-28a0fd`, completed), week 4 (`…-4050fc`, **blocked**), week 5 (`…-e45c06`, **created, never started**). A chain script `../conductor/chain_week4_then_week5.sh` was meant to run week 4, gate on `check_week4.sh`, then run week 5 — it aborted when week 4 blocked (details in `docs/status/phase-reviews/week4-5-status.md`). Verification wrappers `automation/check_week{2..5}.sh` exist so the conductor can run the phase gates; `check_week4.sh` / `check_week5.sh` currently fail because `17_gate_week4.py` / `20_gate_week5.py` do not exist yet (they are Week 4/5 deliverables).

---

## 3. Problem formulation (CMDP) — summary

Full spec: `docs/reference/cmdp-spec.md` (binding). Essentials:

**State** — the 29-slot single-building central-agent observation vector, addressed **by name** through `BUILDING_1_OBSERVATION_INDEX` (`src/energy_optimisation/observation_names.py`): calendar (day_type, hour), weather actuals + 3-step-ahead dataset forecasts (temperature, diffuse/direct irradiance), carbon intensity, indoor temperature, non-shiftable load, solar generation, DHW + electrical SoC, net electricity consumption, electricity pricing + 3-step forecasts, cooling demand, DHW demand, occupant count, cooling setpoint. (The parent 3-building scenario is 49-dim; our derived scenario is 29 — an early "49-dim" wording in week-3 docs is a known stale reference, corrected by the pending Week-4 commit.)

**Actions** — 3 continuous dimensions per hourly step: `dhw_storage` ∈ [−1,1] (−charge/+discharge), `electrical_storage` ∈ [−1,1] (−charge/+discharge), `cooling_device` ∈ [0,1] (ratio of nominal cooling electrical input — **not** a temperature setpoint).

**Reward** (PPO only; baselines never see it), frozen before training:

```
r_t = − w_E·(E_t/Ē_B0) − w_P·max(0, E_t − P_ref)/P_ref − w_C·D_t
D_t = max(0, T_in − (T_set + 2.0 °C))
```

**Constraints** (monitored, separate from reward): comfort band (T_in ≤ T_set + 2 °C), SoC reserve band [0.2, 0.9] on requested actions, action-bound clipping counts, grid import E_t ≤ P_grid,max = 1.8084 kWh.

**KPI mapping** — decision-driving: `cost_total`, `all_time_peak_average`, `electricity_consumption_total`, `discomfort_proportion` (+hot variants), `ramping_average`, `zero_net_energy`. All CityLearn KPIs are normalised against its internal no-action baseline (lower = better). The two outage KPIs are empty in this dataset and excluded.

### 3.1 Critical measurement conventions (CityLearn 2.5.0 quirks, verified empirically)

1. **Post-step observation zeros.** In the observation vector returned by `step()`, computed slots (`net_electricity_consumption`, SoCs, indoor temperature) read 0.0 — CityLearn writes the underlying arrays after composing the observation. The harness reads executed values from the same building time series `env.evaluate()` consumes (`runner.executed_step_values`) and repairs the controller's next input causally with step t−1 values. Any new code touching observations must respect this.
2. **Episode length convention.** CityLearn terminates after exactly `end − start` hourly steps ⇒ 167/719 trace rows.
3. **Neutral RL action.** `[-1,-1,-1]` in RL space maps to CityLearn `[0, 0, −1→0]`, behaviourally identical to B0's zero actions (battery sits at its DoD floor ≈ 0.2; DHW tank at 0). RL `[0,0,0]` is *not* neutral (maps to half-rate cooling).
4. **Causality rule.** Reading any actual (non-`*predicted*`) CSV column at a row > t is lookahead and forbidden everywhere.

---

## 4. Implementations (what exists, where)

### 4.1 Week 1 — foundation (complete, commit `d00ba1b…`, reviewed in `docs/status/phase-reviews/week1-review.md`)

Repo scaffold, pinned venv, local CityLearn bootstrap avoiding GitHub API rate limits (`scripts/01_fetch_pinned_dataset.py`), derived single-building schema, parent-scenario interface inspection (`results/inspection/citylearn_2023_phase_1.json`), deterministic zero-action smoke run through step 167 (`results/runs/smoke/`), README with RQ1–3, literature matrix (12 sources screened, 6 fully analysed — `docs/reference/literature.md`, `docs/reference/literature-matrix.csv`), phase gate `scripts/05_gate_week1.py`.

### 4.2 Week 2 — CMDP + harness + deterministic baselines (complete, commit `db35976`, reviewed in `docs/status/phase-reviews/week2-review.md`)

- `src/energy_optimisation/observation_names.py` — schema-derived, frozen (immutable `MappingProxyType`) name→index map.
- `src/energy_optimisation/evaluation/` — the **locked harness**: `runner.run_episode()` (repair → act → clip → step → trace), `metrics.compute_derived_metrics()` (comfort hours, SoC min/max, clipping/reserve events, peak, solar self-consumption, grid-limit exceedances), `artifacts.write_run_artifacts()` (the standard 5-file run set: `run_metadata.json`, `trace.csv`, `district_kpis.csv`, `derived_metrics.json`, `README.md`).
- `src/energy_optimisation/baselines/controllers.py` — `Controller` ABC (`act(observation) → (3,)` requested actions) + **B0** neutral (zeros), **B1** fixed-schedule (hour-banded, price-blind), **B2** tariff-aware (discharges at price ≥ τ within reserve band, charges off-peak, peak cooling 0.6).
- Frozen lock `configs/week2-baselines.yaml`; runs via `scripts/06_run_baselines.py`; comparison via `scripts/07_compare_baselines.py`; gate `scripts/08_gate_week2.py` (9 checks incl. B0 regression vs smoke anchors at 1e-9).
- Harness validated *before* controllers existed: B0 through the harness reproduces all six smoke KPI anchors exactly (max |Δ| = 0).

### 4.3 Week 3 — standard PPO (complete, commit `52e7f94`, reviewed in `docs/status/phase-reviews/week3-review.md`)

- `src/energy_optimisation/rl/env_adapter.py` — `CityLearnRLEnv(gymnasium.Env)`: 29-dim min-max-normalised observation (per-feature `(offset, scale)` frozen in `configs/week3-ppo.yaml`, computed once by `scripts/09_compute_normalization_stats.py` from the B0 dev trace + schema static ranges), action space Box(−1,1)³ mapped to CityLearn as `[a0, a1, (a2+1)/2]`, reward = frozen CMDP formula from executed values, `terminated` always False / `truncated` only at window end, pre-clip violation counting.
- `src/energy_optimisation/rl/controller.py` — `PPOController` (SB3 PPO on the week-2 Controller interface, deterministic predict, frozen normalisation) + `episode_return_from_trace` (exact replay of training return from a harness trace).
- `src/energy_optimisation/rl/checkpoint_selection.py` — frozen rule: **lowest dev `cost_total`, tie-break lower `discomfort_proportion`**.
- Scripts: `10_train_ppo.py` (SB3 2.3.2, MlpPolicy [64,64], n_steps 2048, batch 256, 10 epochs, lr 3e-4, γ 0.99, λ 0.95, clip 0.2, ent 0.01, vf 0.5, max-grad-norm 0.5, 200k steps, checkpoint every 10k, CPU, refuses non-CPU), `11_evaluate_checkpoints.py` (all 21 checkpoints/seed through the locked harness), `12_evaluate_final_window.py`, `13_compare_ppo.py`, gate `14_gate_week3.py` (9 checks incl. independent re-execution of the selection rule and byte-identity of week-2 evidence).
- Hyperparameters frozen in config *before* the first run; `14_gate_week3.py` checks each run's recorded config SHA-256 against the current file.
- Adapter regression: constant RL action `[-1,-1,-1]` reproduces all six B0 smoke anchors at ≤ 1e-9.

### 4.4 Test suite

60 tests passing (`./.venv/bin/python -m pytest -q`): environment (2), observation names (5), controllers (20 items incl. real-env dev-window runs), metrics (6), runner/artifacts (6), RL env adapter (9, incl. the B0-anchor regression), PPO controller + selection rule (12). `.pytest_cache` holds a stale 62-test snapshot from an earlier revision — harmless.

### 4.5 What does NOT exist yet (as of 1 Sep 2026)

- **Week 4 forecasting** — `src/energy_optimisation/forecasting/` is an empty package (`.gitkeep` only); no `configs/`, no `results/runs/forecasting/`, no forecast tables/figures, no `17_gate_week4.py`. Fully specified in `docs/plans/week4-implementation-plan.md`.
- **Week 5 uncertainty-aware PPO** — no `results/runs/ppo_week5/`, no `configs/ppo/week5_{point,interval}.yaml`, no `19_compare_rq1.py`/`rq1_verdict.json`. Fully specified in `docs/plans/week5-implementation-plan.md`.
- **Safety shield** (`src/energy_optimisation/safety/` empty — Week 6+), scenario/robustness work, dashboard, manuscript.
- A repo-wide grep for forecast/shield/uncertainty/quantile/conformal in `src/`+`scripts/` matches only incidental docstrings and the guard that *bans* forecast tokens in baseline code.

---

## 5. Experiments and observations (the numbers)

All KPIs are CityLearn-normalised (lower = better). Windows: dev = 0–167 (selection), final = 0–719 (evaluation-only). Sources: `results/tables/*.csv`, `results/runs/baselines/**`, `results/runs/ppo/**`.

### 5.1 Smoke run (Week 1 — infrastructure check only, not a research result)

Zero actions, 167 steps, seed 42: `cost_total` 0.4420, peak 0.8618, consumption 0.4641, hot-discomfort 91.5%, total reward −14502.62. Purpose: prove env/action/output/terminal handling.

### 5.2 Deterministic baselines (Week 2, seed 42)

| Controller | dev cost | final cost | dev discomfort | final discomfort | dev peak | final peak |
| --- | --- | --- | --- | --- | --- | --- |
| B0 neutral (do nothing) | **0.4420** | **0.3636** | 0.9152 | 0.9804 | **0.8618** | **0.8618** |
| B1 fixed schedule | 1.8631 | 1.7259 | 0.8788 | 0.9719 | 1.0920 | 1.1381 |
| B2 tariff-aware | 1.6709 | 1.5568 | **0.8667** | **0.9691** | 1.0920 | 1.0920 |

Derived metrics (dev / final): comfort-violation hours (hot band) B0 154/706, B1 0/0, B2 0/0 — but B1/B2 over-cool instead (cold-side CityLearn discomfort 87.9%/97.2% and 86.7%/96.9%). Grid-limit exceedances: B0 9/38, B1 101/425, B2 100/417. Peak net demand: B0 7.694 kW, B1 9.749/10.160 kW, B2 9.749 kW. Clipping 0 and reserve events 0 everywhere. B2 beats B1 on cost in both windows (−10.3% dev / −9.8% final) yet loses to doing nothing.

### 5.3 PPO training (Week 3, seeds 42/43/44)

Each seed: 200,704 steps completed (2048-rollout rounding of the 200k request), 1201 episodes, ≈ 5 min wall clock on CPU, 21 checkpoints. Sanity gates held everywhere: zero monitor NaNs, **zero pre-clip action violations**, all artifacts complete. Episode returns improve ≈ −845 → ≈ −380 over training (comfort-driven), while `cost_total` bottoms early then drifts up, and `ramping_average` steadily rises ≈ 0.87 → ≈ 1.5 — training systematically trades cost for comfort.

### 5.4 Checkpoint selection (frozen rule, dev window)

| Seed | Selected checkpoint | dev cost | dev discomfort | episode return |
| --- | --- | --- | --- | --- |
| 42 | 40k of 200k | 0.7524 | 0.4485 | −1444.97 |
| 43 | 20k of 200k | 0.7263 | 0.4848 | −1348.82 |
| 44 | 100k of 200k | 0.7526 | 0.3576 | −639.92 |

All three record `beats_b0_cost: false`. The cost-optimal checkpoints are early and have poor returns; later checkpoints with better returns and lower discomfort cost only ~0.01–0.03 more.

### 5.5 PPO vs baselines (selected checkpoints)

| Controller | dev cost | final cost | dev discomfort | final discomfort |
| --- | --- | --- | --- | --- |
| B0 neutral | **0.4420** | **0.3636** | 0.9152 | 0.9804 |
| B1 fixed schedule | 1.8631 | 1.7259 | 0.8788 | 0.9719 |
| B2 tariff-aware | 1.6709 | 1.5568 | 0.8667 | 0.9691 |
| PPO seed 42 | 0.7524 | 0.6583 | 0.4485 | 0.5288 |
| PPO seed 43 | **0.7263** | **0.6388** | 0.4848 | 0.5400 |
| PPO seed 44 | 0.7526 | 0.6741 | **0.3576** | **0.2735** |

Multi-seed dev means (min–max): cost 0.7438 (0.7263–0.7526), consumption 0.7544, discomfort 0.4303 (0.3576–0.4848), peak 0.9252, ramping 1.0224. Final-window derived: grid-limit exceedances 123/82/102 (seeds 42/43/44) vs B0's 38; reserve events 485/88/416; electrical SoC stays within 0.186–0.200 in *all* PPO runs (the policy barely uses the battery); DHW SoC pinned at 0; solar self-consumption 0.84–0.87.

---

## 6. Findings and inferences

Recorded honestly per the project's no-post-hoc-tuning discipline — hyperparameters and rules were frozen before results and never retuned.

1. **No controller beats doing nothing on cost.** B0 (zero actions) is cheapest on both windows because it never runs the cooling device; it "pays" with 91.5–98.0% hot discomfort. Every active controller buys comfort with 60–180% more consumption. This is the frontier any proposed method must improve.
2. **PPO is decisively better than the naive active baselines** — ≈ 2.2× cheaper than B1/B2 on both windows while roughly halving their discomfort. The learned policy is genuinely controlling; it just cannot beat a controller that does nothing.
3. **The frozen reward explains the trade.** With w_C = 10, comfort dominates the return by design; the policy cuts hot discomfort from 91.5% → ~29–48% at the cost of consumption. An uncertainty-aware controller (RQ1) should reduce the comfort term *without* the ~70% energy premium; the safety shield (RQ2) has a concrete target in the grid-limit counts.
4. **`cooling_device` is the dominant cost lever, and it is an input ratio, not a setpoint.** B1/B2 driving it at 0.5–0.8 daytime pushes consumption to 1.6–1.8× and systematically over-cools (cold discomfort ≈ 97%). Fixed open-loop cooling without indoor-temperature feedback is the wrong shape of controller — which is the motivation for learned, state-aware control.
5. **Storage arbitrage is a trap at this tariff spread.** Battery round-trip losses (η = 0.95) plus a peak/mid price ratio < 2.03 cannot be recovered at the frozen ±0.5 action levels; B2's battery drains to the reserve edge and stays there; the DHW tank never charges above 0. Week-3 PPO independently learned to *ignore* the battery (SoC within 0.186–0.200 throughout) — two very different controllers converging on "don't cycle storage" is strong evidence the arbitrage opportunity is absent under this tariff.
6. **Grid-limit discipline is the open safety problem.** Exceedances vs P_grid,max = 1.8084 kWh: B0 9/38, B1 101/425, B2 100/417, PPO 30/17/24 (dev) and 123/82/102 (final). An unconstrained PPO trained on a penalty-only reward does not respect the import limit — the direct empirical motivation for the Week-6+ shield (RQ2).
7. **Tariff-awareness helps where it can act** (B2 > B1 on cost, consumption, ramping, and discomfort on both windows) — supports the RQ3 contrast price-adaptive vs calendar-only even though neither beats B0.
8. **Checkpoint selection on cost alone is fragile** — it picks early, low-return policies (20k–100k of 200k) and behaves differently per seed. Seed 44 shows the largest behavioural spread (much lower discomfort at similar cost). Multi-seed variance is now quantified and small on cost (±0.013) but material on behaviour.
9. **Methodological inference:** byte-frozen configs, anchor regressions at 1e-9, independent re-execution of selection rules in the phase gates, and "never touch prior outputs" byte-identity checks have kept three weeks of autonomous execution fully reproducible — this discipline is the reason the negative results above are trustworthy.

---

## 7. Known anomalies, caveats, and technical debt

Verified against primary artifacts; none invalidate results, all worth knowing:

1. **Seed-recording inconsistency (week 3):** `run_metadata.json` for seeds 43/44 has top-level `seed: 43/44` correct, but the embedded `config_ppo_block.seed` reads 42 for all runs (the YAML hardcodes 42; the run seed was a CLI override). Cosmetic provenance wart.
2. **Two comfort definitions coexist.** Derived `comfort_violation_hours` counts the *hot* band only (matches the CMDP constraint); CityLearn's `discomfort_proportion` counts hot + cold. B1/B2 therefore show 0 derived violation hours while CityLearn reports ~97% (cold) discomfort. Both are correct under their own definitions; always state which one a number comes from.
3. **Erratum in `docs/status/phase-reviews/week3-review.md`:** final-window grid exceedances are written "123/102/82 (seeds 42/43/44)"; the artifacts say **123/82/102** (seeds 43/44 transposed). To be corrected alongside the Week-4 doc fixes (the week-4 plan already schedules two other text corrections in week-3 docs).
4. **Legacy artifacts:** `results/runs/baselines/b0_zero_actions/0-167/` (pre-naming-lock harness-regression run, no derived metrics) and the `results/figures/0-167_*` set (pre-`dev_*` prefix) are superseded but retained untouched. `results/logs/` is empty.
5. **`train_stdout.log` exists only for seed 42** (seeds 43/44 have no stdout log).
6. **Stale references:** "49-dim" wording in two week-3 docs (correct = 29); `.pytest_cache` lists 62 tests vs the current 60.
7. **`check_week4.sh` / `check_week5.sh` fail by design** until `17_gate_week4.py` / `20_gate_week5.py` exist (they are phase deliverables).
8. **Stale `.pause.request` files** exist in the week-2/4/5 mission dirs (leftovers from the documented daemon-stop procedure; harmless but should be cleared before restarting those missions — see `docs/status/phase-reviews/week4-5-status.md`).

---

## 8. Current status and what happens next

**Status as of 1 Sep 2026:** Weeks 1–3 complete, verified, committed (HEAD `0d6de3a`, clean tree). Week 4 (forecasting) and Week 5 (RQ1 comparison) are fully specified but **not started** — the week-4 autonomous mission blocked on 26 Aug 2026 at its very first worker dispatch because its model route (`stealth/ox-alpha`) was retired by the provider (HTTP 404), exhausting the per-task retry cap before any file was written. The week-4→5 chain script correctly refused to start week 5. Zero research work was lost; the repo is exactly at its week-3 state. Full timeline, evidence, and the exact resume procedure: [`docs/status/phase-reviews/week4-5-status.md`](week4-5-status.md).

**Execution order from here:**

1. Resume/re-run Week 4 per `docs/plans/week4-implementation-plan.md` (forecasting package, 12-fold rolling-origin backtest, frozen selection) → `docs/status/phase-reviews/week4-review.md`.
2. Week 5 per `docs/plans/week5-implementation-plan.md` (matched-pair point-vs-interval PPO, pre-registered RQ1 verdict rule) → `docs/status/phase-reviews/week5-review.md`.
3. October: safety shield + forecast-noise/tariff/solar scenarios (RQ2/RQ3).
4. November: robustness aggregation, dashboard, manuscript.

**Claim discipline reminder:** nothing so far is a savings claim. B0/B1/B2 is single-seed heuristic evidence; week-3 PPO is three-seed evidence under one tariff profile; all of it is superseded by the planned multi-seed, multi-scenario evaluation.

---

## 9. Reproduction quick reference

```bash
cd code
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt

# Foundation (week 1)
python scripts/01_fetch_pinned_dataset.py
python scripts/02_derive_building_schema.py
python scripts/03_inspect_environment.py
python scripts/04_run_smoke_test.py
python scripts/05_gate_week1.py

# Baselines (week 2)
python scripts/06_run_baselines.py
python scripts/07_compare_baselines.py --window dev --window final
python scripts/08_gate_week2.py

# PPO (week 3) — ~5 min per seed on CPU
python scripts/10_train_ppo.py --config configs/week3-ppo.yaml --seed 42   # then 43, 44
python scripts/11_evaluate_checkpoints.py --seed 42                        # then 43, 44
python scripts/12_evaluate_final_window.py
python scripts/13_compare_ppo.py
python scripts/14_gate_week3.py

python -m pytest -q    # 60 tests
```

---

## 10. Documentation index

| Document | Content |
| --- | --- |
| `README.md` | Project overview, RQs, scope, quick start, evaluation contract (entry point) |
| **`docs/status/research-log.md`** | **This document — the complete state of the project** |
| `docs/status/phase-reviews/week4-5-status.md` | Week 4/5 blocked-mission record and resume procedure |
| `docs/reference/cmdp-spec.md` | Formal CMDP: state/actions/transition/reward/constraints/KPIs + frozen constants |
| `docs/reference/environment-selection.md` | Environment selection rationale (why 2023 phase 1, why not 2020) |
| `docs/reference/experiment-protocol.md` | Evolving experiment recording protocol |
| `docs/reference/literature.md` + `literature-matrix.csv` | 12 screened sources, 6 analysed, gaps mapped to RQs |
| `docs/status/phase-reviews/week1-review.md` / `week2-review.md` / `week3-review.md` | Phase completion records with all numbers |
| `docs/week2/3/4/5-implementation-plan.md` | Binding per-phase specs (4 and 5 await execution) |
| `docs/status/phase-reviews/week1-progress-against-research-plan.md` | Week-1 mapping onto the approved plan |
| `../Plans/week1.md`, `../Plans/week2.md` | Original weekly briefs |
| `../123CS0143_PrajjwalAcharya_ResearchPlan.pdf` | Approved semester research plan |
