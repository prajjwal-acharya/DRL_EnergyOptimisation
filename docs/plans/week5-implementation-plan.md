# Week 5 Implementation Plan: Uncertainty-Aware PPO — Point vs Interval Matched Comparison (RQ1)

This document is the binding, deterministic implementation spec for the Week 5 phase.
It is written to be executed top-to-bottom by an autonomous worker with no other context.
Phases must be completed in order A → B → C → D. Each phase has an acceptance gate that
must pass before the next phase starts.

Working directory for every command is `code/`. Python is `./.venv/bin/python`
(Python 3.9-compatible code only — see the Week 4 plan preamble).

Weeks 1–4 are complete and frozen. This phase answers **RQ1** — *can forecast
uncertainty improve the cost-comfort trade-off compared with a controller that uses only
point forecasts?* — via a **matched-pair experiment**: two PPO arms identical in every
respect except the forecast-feature block appended to the observation. The week-3 plain
PPO (no forecast features) is re-reported as a reference. **No safety shield, no
scenario/robustness work** — those are Week 6+ (October).

Prerequisite: `docs/plans/week4-implementation-plan.md` completed, in particular
`results/runs/forecasting/selected_models.json` and the `ForecastProvider` interface
(`src/energy_optimisation/forecasting/api.py`). If Week 4 shipped persistence for a
target, this phase proceeds with it — the matched pair only needs both arms to consume
the identical forecaster.

---

## 0. Ground truth (verified facts — do not re-derive, do not contradict)

### 0.1 Frozen environment and constants (unchanged from weeks 2–4)

- CityLearn `2.5.0`, dataset `citylearn_challenge_2023_phase_1`, derived single-building
  schema `configs/schema-building1.json`; 720 hourly steps;
  dev window 0–167, final window 0–719; reward/window/seed discipline unchanged.
- Observation: the single-building central-agent vector has **29 slots** (the "49-dim"
  wording in week-3 docs is corrected by the Week 4 commit; do not reintroduce it).
- Frozen CMDP constants (`docs/reference/cmdp-spec.md` — single source of truth):

| Constant | Value |
| --- | --- |
| `Ē_B0` | `0.477229108554339` kWh |
| `P_ref` | `7.694016456604004` kWh |
| `P_grid,max` | `1.8084184527397151` kWh |
| `w_E, w_P, w_C` | `1.0, 1.0, 10.0` |
| ΔT | `2.0 °C` |
| SoC reserve band | `[0.2, 0.9]` |

- Frozen reward (identical for both arms — never rescale, never add terms):

```
r_t = − w_E · (E_t / Ē_B0) − w_P · max(0, E_t − P_ref) / P_ref − w_C · D_t
D_t = max(0, T_in − (T_set + ΔT))   in °C
```

- Action space and mapping identical to week 3: RL `[-1,1]³` → CityLearn
  `[dhw_storage, electrical_storage, (a2+1)/2]`; neutral RL action `[-1,-1,-1]`.
- RL stack: `stable-baselines3==2.3.2`, torch 2.8.0, CPU only.

### 0.2 Week-3 reference results (context for expectation-setting; do not try to "fix")

- Plain PPO (seeds 42/43/44) dev `cost_total` mean 0.7438 vs B0 0.4420 — PPO **lost to
  do-nothing on cost** because `w_C = 10` makes the policy buy comfort (~36–48%
  discomfort vs B0's 91.5%) with ~60–70% more consumption. This trade-off is the
  frontier the interval arm must improve.
- Grid-limit exceedances (dev): PPO 30/17/24 vs B0's 9 — the Week-6 shield's target;
  recorded here only as carried context.
- Checkpoint selection landed early (20k–100k of 200k steps) for all seeds.

### 0.3 Experiment arms (the matched pair — everything identical except the feature block)

| Arm | Name | Observation | Purpose |
| --- | --- | --- | --- |
| Reference | `plain` | the 29 week-3 slots, week-3 normalisation | re-reported from week-3 runs; **not retrained** |
| A | `point` | 29 slots + **9 point features** (§0.6) | RQ1 control |
| B | `interval` | 29 slots + **36 uncertainty features** (§0.6) | RQ1 treatment |

Shared by arms A and B — identical, frozen, and verified byte-identical outside the
variant block: schema, windows, reward constants, normalisation of the base 29 slots
(copied verbatim from `configs/week3-ppo.yaml`), all PPO hyperparameters (§0.5),
seeds {42, 43, 44}, checkpoint cadence, checkpoint-selection rule, and evaluation
harness. Any performance difference between A and B is then attributable to the
uncertainty information alone.

### 0.4 Forecast consumption and leakage discipline (binding)

- Both arms consume the **same** frozen Week-4 forecaster artefacts via
  `ForecastProvider` (full-series refit per Week-4 §0.3). Identical inputs ⇒ any
  residual in-sample leakage is **symmetric across arms** and cannot explain an A-vs-B
  difference. This argument and its limits must appear verbatim in
  `docs/status/phase-reviews/week5-review.md` as a declared limitation.
- Causality at decision time: features at step t come from
  `ForecastProvider.feature_vector(t, variant)`, whose structural causality is enforced
  and unit-tested in Week 4 (`test_provider_causality`). The environment adapter must
  call the provider with the **current** t only — never t+1.
- Determinism: features are a pure function of (frozen artefacts, t). The adapter
  computes them once at reset into a precomputed array indexed by t; training and
  evaluation consume that array. The SHA-256 of the flattened feature matrix is written
  into every run's `run_metadata.json`; `20_gate_week5.py` recomputes and compares.

### 0.5 Locked decisions (do not revisit inside this phase)

| Decision | Value | Reason |
| --- | --- | --- |
| Hyperparameters | week-3 values verbatim: `MlpPolicy [64,64]`, `n_steps=2048`, `batch_size=256`, `n_epochs=10`, `lr=3e-4`, `γ=0.99`, `λ_GAE=0.95`, `clip=0.2`, `ent=0.01`, `vf=0.5`, `max_grad_norm=0.5`, `total_timesteps=200000`, `checkpoint_every=10000`, device `cpu` | isolates the observation change; changing two things at once destroys RQ1 attribution |
| Seeds | `{42, 43, 44}` (42 first for bring-up) | matches week 3; paired per-seed analysis |
| Training window | dev 0–167 | unchanged; final window stays evaluation-only |
| Checkpoint selection | lowest dev `cost_total`; tie-break lower `discomfort_proportion` | the week-3 frozen rule, unchanged |
| Output root | `results/runs/ppo_week5/<variant>/seed<seed>/…` | never write into `results/runs/ppo/` (week-3 evidence) |
| Pre-registered RQ1 verdict | §0.7 | frozen before any training in this phase |

### 0.6 Frozen appended-feature blocks (exact layout, exact order)

Targets in fixed order `T = [solar_generation, non_shiftable_load, cooling_demand]`,
horizons `H = [1, 2, 3]`, quantiles from the frozen Week-4 set. Feature `i` is
addressed by name `(<target>, <horizon>, <kind>)` — no magic indices anywhere.

- Arm A `point` (9 features): for each `T × H` in row-major target-then-horizon order:
  `q50` (the point forecast).
- Arm B `interval` (36 features): for each `T × H` in the same order:
  `q05`, `q50`, `q95`, `width90 = q95 − q05`.
- Normalisation of all appended features: min-max `(offset, scale)` pairs computed once
  by `scripts/uncertainty_aware_ppo/18_compute_forecast_feature_stats.py` from the Week-4 **out-of-sample
  backtest predictions** (pooled over folds; identity pairs where max−min < 1e-12),
  written into both arm configs **identically**, frozen before training. Hand-editing
  afterwards is forbidden.
- Diagnostic duty: interval widths must actually vary. Per appended feature the runner
  records mean/std over the dev window into
  `results/tables/rq1_feature_diagnostics.csv`; `width90` features with std < 1e-3 are
  flagged `degenerate` — the October forecast-noise scenarios exist precisely for that
  outcome, which is a finding to report, not a reason to stop.

### 0.7 Pre-registered RQ1 verdict rule (frozen now, executed mechanically in Phase D)

Compute per-seed paired deltas (arm B − arm A) on the **dev window**: `cost_total` and
`discomfort_proportion`.

- RQ1 answer is **"uncertainty helps"** iff: arm B has lower dev `cost_total` than arm A
  on **≥ 2 of 3 seeds**, AND arm B's mean dev `discomfort_proportion` is not worse than
  arm A's by more than **0.02 absolute**.
- Otherwise the answer is **"no improvement in this scenario"** — a recorded finding
  (report per-seed deltas, the worst seed, and the feature-degeneracy diagnostic; a
  negative answer fed by degenerate widths explicitly motivates October's noise
  scenarios).
- Final-window results, the frontier plot, and grid-limit counts are reported in full
  but do not enter the verdict rule (dev-window discipline).

---

## Phase A — Adapter extension and forecast-feature provider wiring

### A1. `src/energy_optimisation/rl/env_adapter.py` (extend; never change plain behaviour)

- `CityLearnRLEnv.__init__(…, forecast_provider=None, observation_variant="plain")`.
  - `plain` (default): byte-for-byte week-3 behaviour — same spaces, same normalisation,
    same repair logic. The existing week-3 regression test must pass **unchanged**.
  - `point` / `interval`: observation space becomes `Box` of size 38 / 65 (29 base +
    appended block per §0.6); appended features are min-max normalised with the config
    pairs and appended **after** the 29 base features; reward, actions, step contract,
    and observation repair are untouched.
- The provider array is built once at reset: `features[t] = provider.feature_vector(t,
  variant)` for every t in the window; `step()` reads `features[self._t]`.
- `configs/week5-point.yaml` and `configs/week5-interval.yaml` are structurally
  identical to `week3.yaml` plus: the `observation.variant` block, the appended-feature
  normalisation pairs, and a distinct `experiment_name`. No other differences — ever.

### A2. Tests (`tests/test_rl_forecast_obs.py`)

- `test_plain_variant_unchanged` — plain mode reproduces the B0 anchors (window 0–167,
  seed 42, constant RL action `[-1,-1,-1]`) to ≤1e-9, and the observation equals the
  week-3 adapter's for a fixed trajectory (spot-check ≥ 3 steps).
- `test_observation_shapes_per_variant` — 29 / 38 / 65; all finite; within [0, 1] after
  normalisation (modulo float epsilon).
- `test_point_block_identical_across_arms` — for ≥ 100 sampled t, arm B's `q50`
  features equal arm A's point features exactly.
- `test_features_causality_and_determinism` — features depend only on the frozen
  Week-4 artefacts and t: recomputing after corrupting dataset rows > t changes nothing
  (delegates to the Week-4 provider test but re-verified through the adapter); two env
  resets give identical arrays.
- `test_configs_share_all_blocks_except_variant` — YAML-diff of the two configs after
  removing the `observation.variant` and `experiment_name` keys is empty; base-29
  normalisation pairs equal `week3.yaml`'s verbatim.

**Acceptance gate A:** all new tests pass; all existing tests (60 + week-4 additions)
still pass, unchanged.

---

## Phase B — Feature statistics and configs frozen

### B1. `scripts/uncertainty_aware_ppo/18_compute_forecast_feature_stats.py`

Reads the Week-4 backtest `predictions.csv` for the selected variants, computes the
§0.6 min-max pairs for both feature blocks, and writes them into both arm configs'
`normalisation.forecast_features` blocks (identical bytes in both files). Refuses to run
if the configs already contain the block (no silent overwrites after freezing).

### B2. Freeze check

- Both configs exist with the full §0.5 hyperparameter set (byte-identical to week 3's
  values), the §0.6 blocks, and distinct experiment names.
- `run_metadata.json` for every future run records: git commit, config SHA-256,
  provider artefact hash (Week-4 `selected_models.json` + model weights), feature-matrix
  SHA-256, torch/SB3 versions, device, seed.

**Acceptance gate B:** `scripts/uncertainty_aware_ppo/18_compute_forecast_feature_stats.py` runs once and exits 0;
the A2 config-identity test passes; nothing under `results/` from weeks 1–4 changed.

---

## Phase C — Training and locked-harness evaluation (both arms × three seeds)

### C1. Training

- `./.venv/bin/python scripts/standard_ppo/10_train_ppo.py --config configs/week5-point.yaml --seed 42`
  (then 43, 44; then `configs/week5-interval.yaml` × 3 seeds) — 6 runs total,
  ≈5 min each on CPU per week-3 timing; budget ceiling 4 h per run (record timing,
  never cut silently).
- Outputs under `results/runs/ppo_week5/<variant>/seed<seed>/checkpoints/` + `final`,
  monitor CSV, `run_metadata.json` — mirroring the week-3 layout one level deeper.
- Bring-up sanity gates per run (same as week 3): zero NaNs, zero pre-clip action
  violations, ≥15 numbered checkpoints + final, return-curve figure
  `results/figures/ppo_week5_<variant>_seed<seed>_return_curve.png`.

### C2. Checkpoint evaluation and selection

- Reuse `scripts/standard_ppo/11_evaluate_checkpoints.py` / `scripts/standard_ppo/12_evaluate_final_window.py`,
  extended only with `--config` / `--output-root` flags (defaults unchanged; week-3
  behaviour regression-tested by gate A). Every checkpoint through the locked week-2
  harness on dev; `evaluations.csv` per run with the week-3 column set plus
  `variant`; the frozen selection rule applied by script → `selected_checkpoint.json`
  per run; final-window (0–719) evaluation per selected checkpoint with complete
  artefact sets under `results/runs/ppo_week5/<variant>/seed<seed>/final/`.

**Acceptance gate C:** all 6 runs complete with sanity gates held; selection executed
mechanically; final-window artefacts complete for all 6.

---

## Phase D — RQ1 comparison, verification, documentation, commit

### D1. Analysis artefacts (`scripts/uncertainty_aware_ppo/19_compare_rq1.py`, new)

- `results/tables/rq1_point_vs_interval.csv` — per seed × arm (`plain` reference rows
  copied from week-3 results, `point`, `interval`): dev and final `cost_total`,
  `electricity_consumption_total`, `discomfort_proportion`,
  `discomfort_hot_proportion`, `all_time_peak_average`, `ramping_average`,
  `grid_limit_exceedances`, `reserve_events`, `clipping_events`; plus per-seed paired
  delta rows (interval − point) for `cost_total` and `discomfort_proportion`.
- `results/tables/rq1_multiseed_summary.csv` — per arm: mean ± (min–max) of the same
  KPI set across seeds (dev and final).
- `results/tables/rq1_feature_diagnostics.csv` — per appended feature: mean, std,
  `degenerate` flag (§0.6).
- `results/tables/rq1_verdict.json` — the §0.7 rule executed mechanically: verdict
  string, the seed-wise comparisons that produced it, and the rule text hash.
- Figures: `results/figures/rq1_cost_discomfort_frontier.png` (dev window:
  `discomfort_proportion` vs `cost_total`, all of B0/B1/B2/plain-PPO/point/interval,
  one marker per seed for the PPO arms) and
  `results/figures/rq1_paired_deltas.png` (per-seed interval−point deltas, cost and
  discomfort).

### D2. `scripts/uncertainty_aware_ppo/20_gate_week5.py` (hard pass/fail, mirroring week 3/4)

- `./.venv/bin/python -m pytest -q` passes (all phases).
- Six complete run artefact sets under `results/runs/ppo_week5/`; `evaluations.csv` NaN-free;
  selection rule reproducible; config SHA-256s match every run's metadata.
- Both arm configs identical outside the variant block (re-check in code, not by trust).
- Feature-matrix SHA-256 recomputed and matched for all 6 runs.
- The four §D1 tables + two figures exist, NaN-free, expected schema.
- `rq1_verdict.json` present and its rule text hash matches §0.7 as written in the
  frozen config copied at freeze time.
- Weeks 1–4 evidence byte-identical to git HEAD (`results/tables/*` from weeks 2–4;
  `results/runs/baselines/`, `results/runs/ppo/`, `results/runs/forecasting/` unmodified).
- `docs/status/phase-reviews/week5-review.md` exists.

### D3. `docs/status/phase-reviews/week5-review.md` (written last, honestly)

Structure: infrastructure evidence (adapter extension, feature wiring, freeze checks)
first; then the matched-pair results; then the §0.7 verdict **as executed** — including
per-seed deltas, worst seed, feature-degeneracy diagnostic, and final-window tables; a
negative verdict is reported with the same prominence as a positive one. Must contain
verbatim: (a) the §0.4 symmetric-leakage limitation statement, (b) the statement that
hyperparameters were frozen at week-3 values before training and never retuned, and
(c) the supervisor update:

> The uncertainty-aware PPO comparison is complete: point-forecast and
> interval-forecast arms, identical in all other respects, were trained across three
> seeds and evaluated against the locked B0/B1/B2 and week-3 PPO references through the
> unchanged harness, with a pre-registered verdict rule deciding RQ1. The safety shield
> and robustness scenarios remain future work and have a concrete target in the recorded
> grid-limit exceedance counts.

### D4. Commit

Single commit closing the phase, e.g.
`feat: week-5 uncertainty-aware ppo matched comparison (rq1)`.

**Acceptance gate D:** `./.venv/bin/python scripts/uncertainty_aware_ppo/20_gate_week5.py` exits 0; repo green;
all changes committed.

---

## Guardrails (binding for the worker)

- **Out of scope:** no safety shield, no forecast-noise / tariff / solar / occupancy
  scenario modifications, no robustness experiments, no ablations beyond the two-arm
  design, no dashboard, no manuscript work, no changes to the forecaster.
- Never modify anything under `results/runs/baselines/`, `results/runs/ppo/`,
  `results/runs/forecasting/`, `results/runs/smoke/`, week-2/3/4 tables,
  `configs/week2-baselines.yaml`, `configs/week3-ppo.yaml`,
  `configs/week4-forecasting.yaml`, or any week 1–4 doc/test.
- Never write into `results/runs/ppo/`; this phase's outputs live only under
  `results/runs/ppo_week5/`, new `results/tables/rq1_*`, and new `results/figures/`.
- Never install new dependencies; CPU only; constants in config not code.
- The verdict rule (§0.7) and hyperparameters are frozen before training; if training
  diverges (NaN), fix code bugs only — never retune after seeing results.
- Never hand-edit anything under `results/`; regenerate from scripts only.
- If a command fails: retry once; if it fails again, record the error and treat it as a
  blocker. Do not weaken or skip a failing test.

## Definition of done

1. `CityLearnRLEnv` supports the three observation variants with the plain-mode B0-anchor
   regression still exact (≤1e-9) and unchanged week-3 behaviour.
2. Both arm configs frozen with identical shared blocks and config-level proof of that
   identity; forecast-feature normalisation computed once and frozen.
3. Six training runs (2 arms × seeds 42/43/44) complete with sanity gates held;
   checkpoints, learning curves, and mechanical checkpoint selection recorded.
4. Dev + final evaluations of every selected checkpoint through the locked harness,
   complete artefact sets under `results/runs/ppo_week5/`.
5. `rq1_point_vs_interval.csv`, `rq1_multiseed_summary.csv`,
   `rq1_feature_diagnostics.csv`, `rq1_verdict.json`, and the two RQ1 figures exist and
   are NaN-free; the verdict was produced by the pre-registered rule, unedited.
6. `./.venv/bin/python -m pytest -q` passes; `./.venv/bin/python scripts/uncertainty_aware_ppo/20_gate_week5.py`
   exits 0; weeks 1–4 evidence byte-identical.
7. `docs/status/phase-reviews/week5-review.md` written with the required verbatim statements; phase committed.
