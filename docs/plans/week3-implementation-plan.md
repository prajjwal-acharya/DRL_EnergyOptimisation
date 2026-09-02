# Week 3 Implementation Plan: Standard PPO Controller and Learning Curves

This document is the binding, deterministic implementation spec for the Week 3 phase.
It is written to be executed top-to-bottom by an autonomous worker with no other context.
Phases must be completed in order A → B → C → D. Each phase has an acceptance gate that
must pass before the next phase starts.

Working directory for every command is `code/`. Python is `./.venv/bin/python`
(Python 3.9.6 — all new code must be 3.9-compatible: no PEP 604 `X | Y` type unions,
no `match` statements; use `typing.Optional`/`Tuple` or `from __future__ import annotations`).

Week 2 (CMDP spec, evaluation harness, B0/B1/B2 baselines) is complete and frozen. This
phase adds the standard PPO controller: **no forecast-derived features, no uncertainty
representation, no safety shield** — those are October work. Standard PPO consumes the
plain 29-dim CityLearn observation and learns from the frozen CMDP reward only.

---

## 0. Ground truth (verified facts — do not re-derive, do not contradict)

Environment contract:

- CityLearn `2.5.0` pinned to source tag `v2.5.0`, commit `29062af6d077409e1c37a3e53a6cac30fd4d02bc`.
- Dataset `citylearn_challenge_2023_phase_1`, derived single-building schema:
  `configs/schema-building1.json` (Building_1 only, central agent).
- Episode: 720 hourly steps (30 days). Development window: steps 0–167 (7 days, 167 steps).
  Final evaluation window: steps 0–719 (719 steps). Week-2 baseline seed: `42`.
- RL library: `stable-baselines3==2.3.2` and `torch==2.8.0` are installed in `.venv`
  (verified 2026-08-25) and pinned in `requirements.txt`. Install nothing new.

Action space (3 dims, order is fixed; identical to week 2):

| Action | Bounds | Semantics |
| --- | --- | --- |
| `dhw_storage` | `[-1, 1]` | negative = charge DHW tank, positive = discharge |
| `electrical_storage` | `[-1, 1]` | negative = charge battery, positive = discharge |
| `cooling_device` | `[0, 1]` | ratio of cooling-device nominal electrical input (NOT a setpoint) |

Frozen CMDP constants (from `docs/reference/cmdp-spec.md` — single source of truth; never re-derive,
never recompute, never edit):

| Constant | Value |
| --- | --- |
| `Ē_B0` (mean net consumption normaliser) | `0.477229108554339` kWh |
| `P_ref` (peak-excess reference) | `7.694016456604004` kWh |
| `P_grid,max` (grid-import limit, 95th pct B0 dev) | `1.8084184527397151` kWh |
| Reward weights `w_E, w_P, w_C` | `1.0, 1.0, 10.0` |
| Comfort band ΔT | `2.0 °C` |
| SoC reserve band | `[0.2, 0.9]` |
| Dev / final windows | `0–167` / `0–719` |

Frozen reward (consumed exactly as written; do not rescale, do not add terms):

```
r_t = − w_E · (E_t / Ē_B0) − w_P · max(0, E_t − P_ref) / P_ref − w_C · D_t
D_t = max(0, T_in − (T_set + ΔT))   in °C
```

`E_t` = net electricity consumption at step t. Reward magnitudes up to ≈ −15 per step are
expected and acceptable (comfort dominates by design); PPO normalises advantages internally.

B0 anchor (zero actions, window 0–167, seed 42 — from `results/runs/smoke/district_kpis.csv`):

- `cost_total = 0.44198876839332574`
- `all_time_peak_average = 0.8618364154405324`
- `electricity_consumption_total = 0.464085898307736`
- `discomfort_proportion = 0.9151515151515152`
- `ramping_average = 0.8571830450575444`
- `zero_net_energy = 0.35004620158879785`
- The two outage KPIs are empty in this dataset; exclude them everywhere (week-2 convention).

Observation repair convention (binding, from `docs/reference/cmdp-spec.md` §1): under CityLearn 2.5.0
the post-step observation vector carries uncomputed zeros for computed slots
(`net_electricity_consumption`, SoCs, indoor temperature). The week-2 runner reads executed
values from the building time series that `env.evaluate()` consumes and repairs controller
inputs **causally (no lookahead)**. The RL adapter MUST reuse that same repair logic —
extract it into a shared helper if needed; do not reimplement it differently.

Week-2 recorded findings (context for expectation-setting; do not try to "fix" these):

1. Both active baselines B1/B2 *increase* normalised cost and consumption relative to B0
   (do-nothing) because driving `cooling_device` at 0.5–0.8 raises total consumption
   ~1.6–1.8×; hot discomfort drops to 0 but cold discomfort appears (systematic overcooling).
2. Battery cycling at fixed ±0.5 levels loses money (round-trip losses exceed the <2.03×
   peak/mid price spread). B2's battery collapses to the reserve edge.
3. Grid-limit exceedances: B0 9 (dev) / 38 (final); B1 101 / 425; B2 100 / 417.
4. A PPO result worse than B0 on cost is a **recorded finding, not a failure** — never tune
   constants or hyperparameters after seeing results to make them look better.

Existing code to build on (do not duplicate; do not modify week-2 behaviour):

- `src/energy_optimisation/environment.py` — `load_environment` (supports window overrides).
- `src/energy_optimisation/observation_names.py` — the 49-name → index map.
- `src/energy_optimisation/baselines/controllers.py` — the `Controller` ABC
  (`name`, `reset(seed)`, `act(observation)`). September's PPO wrapper implements exactly
  this interface (comment already in the file).
- `src/energy_optimisation/evaluation/` — `runner.py`, `artifacts.py`, `metrics.py`.
- `configs/week2-baselines.yaml` — frozen; read-only reference.
- Tests: all 39 existing tests must keep passing, unchanged.

Trap to avoid: `scripts/cmdp_baselines/08_gate_week2.py` requires `results/tables/baseline_comparison.csv`
to contain **exactly 3 controller rows**. Do NOT add PPO rows to that file. The Week 3
comparison goes to a **new** table `results/tables/ppo_vs_baselines.csv`. Nothing under
`results/runs/baselines/` or `results/runs/smoke/` may be modified.

Decisions locked before training (do not revisit inside this phase):

| Decision | Value | Reason |
| --- | --- | --- |
| Training window | dev `0–167` episodes | fast iteration; final window stays held-out |
| Final-window use | frozen-policy evaluation only, never training | preserves dev/final discipline |
| Compute device | `cpu` for every reported run | MPS kernels are non-deterministic; reproducibility contract |
| Seeds | `{42, 43, 44}` (42 first for bring-up) | ≥3 seeds for mean ± spread reporting |
| Checkpoint selection | lowest dev-window `cost_total`; tie-break lower `discomfort_proportion`; rule frozen before final runs | no post-hoc cherry-picking |

---

## Phase A — RL environment adapter (Gymnasium wrapper)

### A1. Adapter module

Create `src/energy_optimisation/rl/__init__.py` and `src/energy_optimisation/rl/env_adapter.py`
exposing `CityLearnRLEnv(gymnasium.Env)`:

- Constructor takes the schema path, window overrides, and a config dict (from
  `configs/week3-ppo.yaml`). Internally uses `environment.load_environment`.
- **Observation space**: `Box` matching the 29-dim central-agent observation, normalised.
  Normalisation constants are per-feature `(offset, scale)` pairs frozen in
  `configs/week3-ppo.yaml` and computed once by `scripts/standard_ppo/09_compute_normalization_stats.py`
  from the existing B0 dev trace `results/runs/baselines/b0_neutral/dev/trace.csv` and the
  schema's static observation ranges: features already bounded in [0, 1] (SoCs, occupancy,
  ratio-type signals) use identity; unbounded physical signals (temperatures, energies,
  prices) use min-max stats from the B0 dev trace with a small epsilon guard. The script
  writes the stats into the config; hand-editing the stats afterwards is forbidden.
- **Action space**: `Box(low=[-1,-1,-1], high=[1,1,1])`. Mapping to CityLearn actions:
  `dhw_storage = a0`, `electrical_storage = a1`, `cooling_device = (a2 + 1) / 2`.
  Note: RL action `[0, 0, 0]` is NOT neutral B0 — the neutral action is `[-1, -1, -1]`
  (maps to `[0, 0, 0]` CityLearn). Clip before submission; count pre-clip violations.
- **Reward**: computed from the repaired executed values (E_t, T_in, T_set) exactly per
  the frozen formula in §0. Do not normalise or clip the reward.
- **Step contract**: `terminated = False` always; `truncated = True` only at the window's
  terminal step (167 dev). `reset(seed=...)` must be deterministic per seed. Reuse the
  week-2 causal observation repair so the policy never sees lookahead information.
- No forecast-derived or uncertainty features may be added to the observation beyond the
  49 CityLearn dimensions (price/solar forecast signals already inside the observation
  vector are part of the plain state and stay).

### A2. Adapter tests (`tests/test_rl_env.py`)

- `test_observation_space_shape_and_finiteness`
- `test_reset_deterministic_per_seed` (same seed → identical first observation)
- `test_episode_length_is_window_length` (167 steps dev)
- `test_neutral_action_reproduces_b0_anchors` — constant RL action `[-1, -1, -1]` through
  the adapter, window 0–167, seed 42, must reproduce the §0 B0 anchor KPIs (tolerance
  1e-9). This is the adapter-validation regression, equivalent to week 2's harness check.
- `test_action_mapping_bounds` (a2 = -1 → cooling 0.0; a2 = +1 → cooling 1.0)
- `test_reward_matches_frozen_formula` — hand-computed reward for at least 3 constructed
  (E_t, T_in, T_set) triples.

**Acceptance gate A:** all `tests/test_rl_env.py` tests pass; all 39 existing tests still
pass; the B0-anchor regression is exact (≤1e-9).

---

## Phase B — Single-seed PPO bring-up (seed 42, dev window)

### B1. Training script and config

- `configs/week3-ppo.yaml` — every hyperparameter lives here, never in code. Initial
  values (adjustable **only before** the first full training run, then frozen):
  `MlpPolicy [64, 64]`, `n_steps = 2048`, `batch_size = 256`, `n_epochs = 10`,
  `learning_rate = 3e-4`, `gamma = 0.99`, `gae_lambda = 0.95`, `clip_range = 0.2`,
  `ent_coef = 0.01`, `vf_coef = 0.5`, `max_grad_norm = 0.5`, `total_timesteps = 200000`,
  `checkpoint_every = 10000`, `seed = 42`, `device = cpu`, plus the normalisation stats
  block from Phase A.
- `scripts/standard_ppo/10_train_ppo.py --config configs/week3-ppo.yaml --seed 42` — SB3 `PPO` on
  `CityLearnRLEnv`; saves numbered checkpoints under `results/runs/ppo/seed42/checkpoints/`
  every `checkpoint_every` steps plus `final`, writes the SB3 CSV monitor and a
  `run_metadata.json` (git commit, config hash, torch/SB3 versions, device, seed).
- Training must run on CPU (pinned via config). Expect roughly 1–2 hours for 200k steps;
  if wall-clock exceeds 4 hours, record the timing and continue — do not silently cut the
  budget.

### B2. Bring-up sanity gates

- Training completes with zero NaNs in the monitor log.
- All requested actions within the Box (pre-clip violation count = 0 expected; any
  clipping is logged and reported).
- Episode-return learning curve (raw return vs environment steps) saved to
  `results/figures/ppo_seed42_return_curve.png`.

**Acceptance gate B:** training completes; sanity gates hold; checkpoint directory
contains ≥15 numbered checkpoints plus `final`.

---

## Phase C — KPI learning curves through the locked harness

### C1. PPO controller wrapper

`src/energy_optimisation/rl/controller.py` — `PPOController(Controller)` implementing the
week-2 interface: loads an SB3 checkpoint, `act(observation)` returns the mapped 3-dim
CityLearn action, deterministic (no sampling). This is what plugs into the existing
`evaluation/runner.py` unchanged.

### C2. Checkpoint evaluation

`scripts/standard_ppo/11_evaluate_checkpoints.py --seed 42 --window dev`:

- For every checkpoint: run through `evaluation/runner.py` on the dev window (seed 42),
  producing the same KPI set and derived metrics as the week-2 baselines.
- Write `results/runs/ppo/seed42/evaluations.csv` — one row per checkpoint:
  `checkpoint, timestep, episode_return, cost_total, all_time_peak_average,
  electricity_consumption_total, discomfort_proportion, discomfort_hot_proportion,
  ramping_average, zero_net_energy, comfort_violation_hours, grid_limit_exceedances,
  clipping_events, reserve_events`.
- Figures (dev window): `results/figures/ppo_seed42_kpi_curves.png` (cost, discomfort,
  peak vs training progress — the plan's required "PPO learning curve" evidence) and
  `ppo_seed42_return_curve.png` from Phase B.

### C3. Checkpoint selection (rule executed, not judged)

Apply the frozen selection rule from §0 to `evaluations.csv`: lowest `cost_total`,
tie-break lower `discomfort_proportion`. Record the chosen checkpoint and its full KPI
row in `results/runs/ppo/seed42/selected_checkpoint.json`. This choice is final for Phase D.

**Acceptance gate C:** `evaluations.csv` covers all checkpoints, is NaN-free, and the KPI
curves figure exists. If no checkpoint beats B0 on cost, that is a recorded finding —
proceed to Phase D anyway.

---

## Phase D — Multi-seed runs, locked comparison, verification, documentation, commit

### D1. Multi-seed execution

- Repeat Phases B–C for seeds 43 and 44 with the **identical frozen config** (only the
  seed changes; write it via CLI flag, not by editing hyperparameters).
- Aggregate: `results/tables/ppo_multiseed_summary.csv` — per KPI: mean, min, max across
  the three selected checkpoints (dev window).
- Final-window evaluation: each seed's selected checkpoint, evaluated on window 0–719
  through the locked runner, artifacts under `results/runs/ppo/seed<>/final/` in the same
  shape as `results/runs/baselines/<controller>/<window>/` (run_metadata.json, trace.csv,
  district_kpis.csv, derived_metrics.json, README.md).

### D2. Comparison table (additive — never touch week-2 evidence)

`scripts/standard_ppo/13_compare_ppo.py` — reads the week-2 baseline table and the new PPO runs, writes
`results/tables/ppo_vs_baselines.csv` (rows: B0, B1, B2, PPO-seed42, PPO-seed43,
PPO-seed44; columns: the fixed week-2 KPI set, both windows) plus one figure
`results/figures/ppo_vs_baselines_cost.png`. `results/tables/baseline_comparison.csv`
must remain byte-identical to its week-2 state.

### D3. Verification

`scripts/standard_ppo/14_gate_week3.py` (mirror the structure of `scripts/cmdp_baselines/08_gate_week2.py`), checking
each as a hard pass/fail with a clear message:

- `./.venv/bin/python -m pytest -q` passes (week-1 + week-2 + week-3 tests).
- All three seeds have complete artifact sets: checkpoints, `evaluations.csv`,
  `selected_checkpoint.json`, dev + final run dirs.
- `results/tables/ppo_multiseed_summary.csv` and `results/tables/ppo_vs_baselines.csv`
  exist with the expected row structure.
- Learning-curve figures exist (return curve + KPI curves per seed).
- `results/tables/baseline_comparison.csv` unchanged vs git HEAD (week-2 evidence intact).
- No NaN in any PPO trace; clipping/reserve event counts present in derived metrics.
- `configs/week3-ppo.yaml` exists with the frozen hyperparameters and normalisation stats.
- `docs/status/phase-reviews/week3-review.md` exists.

### D4. Documentation and commit

- `docs/status/phase-reviews/week3-review.md`: infrastructure evidence first, then controller-performance
  results; report whether PPO beat B0/B1/B2 honestly (negative results are findings);
  multi-seed mean ± spread; carry-forward implications for October (uncertainty-aware
  state, safety shield). Include the supervisor update verbatim:
  > The standard PPO controller is trained and evaluated against the locked B0/B1/B2
  > baselines across three seeds on the dev and final windows. Forecasting and the
  > safety shield have not started; they will be built on this frozen PPO foundation.
- Single commit closing the phase, e.g. `feat: week-3 standard ppo controller and learning curves`.

**Acceptance gate D:** `./.venv/bin/python scripts/standard_ppo/14_gate_week3.py` exits 0; repo green;
all changes committed.

---

## Guardrails (binding for the worker)

- **Out of scope:** no forecasting models, no prediction intervals or uncertainty-aware
  inputs, no safety shield, no tariff/solar/occupancy scenario modifications, no
  robustness experiments, no dashboard, no manuscript work. October concerns do not
  leak into this phase.
- Never modify anything under `results/runs/baselines/`, `results/runs/smoke/`,
  `results/inspection/`, `results/tables/baseline_comparison.csv`,
  `configs/week2-baselines.yaml`, or any week-2 doc or test.
- Never edit anything under `data/raw/`; never hand-edit anything under `results/`.
- Never install new dependencies; `stable-baselines3==2.3.2` and `torch==2.8.0` are
  already pinned in `requirements.txt`.
- All reported runs on CPU, fixed seeds, constants in config not code.
- Hyperparameters are chosen **before** the first full training run and frozen; if
  training diverges (NaN), fix code bugs only — never retune hyperparameters after
  seeing results to improve them.
- If a command or run fails: retry once; if it fails again with the same error, record
  the error in the run note / progress doc and treat it as a blocker rather than working
  around it. Do not weaken or skip a failing test to make progress.

## Definition of done (mirrors the mission's done criteria)

1. `CityLearnRLEnv` adapter implemented with the B0-anchor regression passing at ≤1e-9.
2. PPO trained for seeds 42/43/44 on the dev window with checkpoints and monitor logs.
3. KPI learning curves per seed via the locked harness; frozen checkpoint selection rule
   applied and recorded.
4. Final-window (0–719) evaluations per seed with complete artifacts.
5. `results/tables/ppo_multiseed_summary.csv` and `results/tables/ppo_vs_baselines.csv`
   exist; `baseline_comparison.csv` unchanged.
6. `./.venv/bin/python -m pytest -q` passes; `./.venv/bin/python scripts/standard_ppo/14_gate_week3.py`
   exits 0.
7. `docs/status/phase-reviews/week3-review.md` written; phase committed.
