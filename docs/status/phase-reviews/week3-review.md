# Week 3 Review: Standard PPO Controller and Learning Curves

Plan reference: `docs/plans/week3-implementation-plan.md` (binding spec). This review reports
the completed phase A→D evidence in the required order — infrastructure first, then
controller performance — including honest negative findings. Nothing under
`results/runs/baselines/`, `results/runs/smoke/`, or `results/tables/baseline_comparison.csv`
was modified (enforced byte-level by `scripts/14_gate_week3.py`).

---

## 1. Infrastructure evidence

### Phase A — Gymnasium adapter

- `src/energy_optimisation/rl/env_adapter.py` exposes `CityLearnRLEnv(gymnasium.Env)`:
  49-dim normalised central-agent observation (per-feature `(offset, scale)` pairs frozen
  in `configs/week3-ppo.yaml`, computed once by `scripts/09_compute_normalization_stats.py`
  from the B0 dev trace + schema static ranges), `Box(low=[-1,-1,-1], high=[1,1,1])`
  action space mapped to CityLearn as `dhw_storage = a0`, `electrical_storage = a1`,
  `cooling_device = (a2 + 1)/2`, terminated always `False`, truncated only at the window
  terminal step.
- The adapter reuses the week-2 causal observation repair (`runner.repair_observation`
  convention): computed slots repaired from the executed building time series, strictly
  causal, no lookahead.
- Reward consumed exactly as frozen in §0 of the plan (`w_E=1, w_P=1, w_C=10`,
  `Ē_B0=0.477229108554339`, `P_ref=7.694016456604004`, ΔT=2.0 °C); never rescaled or clipped.
- Adapter validation regression: constant RL action `[-1,-1,-1]` through the adapter on
  window 0–167, seed 42 reproduces all six B0 smoke anchors exactly (tolerance ≤1e-9;
  test `test_neutral_action_reproduces_b0_anchors`). All 60 tests pass (39 pre-existing +
  21 new week-1/2/3 additions), unchanged.

### Phase B — Single-seed bring-up (seed 42)

- `scripts/10_train_ppo.py --config configs/week3-ppo.yaml --seed 42`: SB3 2.3.2 PPO,
  torch 2.8.0, device pinned `cpu`, MlpPolicy [64,64], n_steps=2048, batch_size=256,
  n_epochs=10, lr=3e-4, γ=0.99, λ_GAE=0.95, clip=0.2, ent=0.01, vf=0.5, max_grad_norm=0.5,
  200k steps, checkpoint every 10k. Hyperparameters were frozen in
  `configs/week3-ppo.yaml` before the first full run and never edited afterwards
  (`14_gate_week3.py` checks every run's recorded config SHA-256 against the current file).
- Sanity gates held for all three seeds: zero NaNs in every monitor log, **zero pre-clip
  action violations** (all requested actions inside the Box), 20 numbered checkpoints +
  `final.zip` per seed, return-curve figure per seed under `results/figures/`.
- Training wall clock ≈ 5 min/seed (well inside the 4-hour budget).

### Phases C/D — Locked-harness evaluation

- Every checkpoint evaluated through `energy_optimisation.evaluation.runner.run_episode`
  (the locked week-2 harness, unchanged) on the dev window; `evaluations.csv` per seed
  covers all 21 checkpoints NaN-free with clipping/reserve event counts.
- Frozen selection rule (lowest dev `cost_total`; tie-break lower
  `discomfort_proportion`) executed by `scripts/11_evaluate_checkpoints.py`;
  `14_gate_week3.py` independently re-executes the rule and requires the same winner.
- Final-window (0–719) evaluations of each seed's selected checkpoint via
  `scripts/12_evaluate_final_window.py`, artifacts under `results/runs/ppo/seed<seed>/final/` in
  the locked harness shape (run_metadata.json, trace.csv, district_kpis.csv,
  derived_metrics.json, README.md).

## 2. Controller performance results

Selected checkpoints (dev window 0–167) and final-window (0–719) evaluation:

| Seed | Selected checkpoint | Dev cost_total | Dev discomfort | Final cost_total | Final discomfort |
| --- | --- | --- | --- | --- | --- |
| 42 | `ppo_00040000_steps.zip` | 0.752412 | 0.448485 | 0.658298 | 0.528752 |
| 43 | `ppo_00020000_steps.zip` | 0.726258 | 0.484848 | 0.638763 | 0.539972 |
| 44 | `ppo_00100000_steps.zip` | 0.752636 | 0.357576 | 0.674128 | 0.273492 |

Multi-seed summary across selected checkpoints (full table:
`results/tables/ppo_multiseed_summary.csv`):

| KPI (dev window) | Mean ± spread (min–max) |
| --- | --- |
| cost_total | 0.7438 (0.7263 – 0.7526) |
| electricity_consumption_total | 0.7544 (0.7446 – 0.7723) |
| discomfort_proportion | 0.4303 (0.3576 – 0.4848) |
| all_time_peak_average | 0.9252 (0.8786 – 0.9592) |
| ramping_average | 1.0224 (0.8899 – 1.2546) |

Comparison against the week-2 baselines (full table: `results/tables/ppo_vs_baselines.csv`,
figure: `results/figures/ppo_vs_baselines_cost.png`):

| Controller | Dev cost | Final cost | Dev discomfort | Final discomfort |
| --- | --- | --- | --- | --- |
| B0 neutral | 0.4420 | 0.3636 | 0.9152 | 0.9804 |
| B1 fixed schedule | 1.8631 | 1.7259 | 0.8788 | 0.9719 |
| B2 tariff-aware | 1.6709 | 1.5568 | 0.8667 | 0.9691 |
| PPO seed 42 | 0.7524 | 0.6583 | 0.4485 | 0.5288 |
| PPO seed 43 | 0.7263 | 0.6388 | 0.4848 | 0.5400 |
| PPO seed 44 | 0.7526 | 0.6741 | 0.3576 | 0.2735 |

### Honest findings (including negatives)

1. **PPO did not beat B0 on cost, on any seed, on either window.** Dev cost mean
   0.7438 ± 0.013 vs B0's 0.4420 (~+68%); final cost mean 0.6571 vs B0's 0.3636 (~+81%).
   Per the plan this is a **recorded finding, not a failure** — hyperparameters were
   frozen before training and were not retuned after seeing results.
2. **PPO does beat B1/B2 decisively on cost** on both windows (≈2.2× cheaper than the
   best active baseline) while also halving their discomfort, so the learned policy is
   clearly better than naive actuation — it simply pays for comfort improvement with
   higher consumption than doing nothing.
3. **The frozen reward explains the trade.** With `w_C = 10`, comfort dominates the
   return by design: episode means improve from ≈ −803 to ≈ −382 over training
   (monitor logs), driven mostly by cutting hot-discomfort proportion from B0's 91.5%
   to ~36–48% on dev. The policy "buys" that comfort with ~60–70% more net consumption
   than B0, which the cost KPI records as a regression.
4. **Grid-limit discipline got worse, not better.** Grid-limit exceedances on dev:
   PPO 30/17/24 (seeds 42/43/44) vs B0's 9; on final: 123/102/82 vs B0's 38. An
   unconstrained PPO trained on the frozen reward does not respect the 95th-percentile
   import limit — direct motivation for the planned safety shield.
5. **Checkpoint selection lands early.** The cost-optimal checkpoints sit at 20k–100k of
   200k steps for all seeds: later policies keep improving raw return (comfort) while
   drifting further above B0 on cost, so the frozen lowest-cost rule selects earlier,
   less-converged policies. Spread across seeds is small on cost (±0.013) but larger on
   behaviour (seed 44 trades more cost for much lower final discomfort).
6. **No action-space pathologies.** Zero pre-clip violations and zero harness clipping
   events in every selected-checkpoint evaluation; reserve-band events remain frequent
   (27–108 dev), i.e., the battery policy rides the SoC band edges like B2 did.
7. Training budget respected: ≈5 min wall clock per seed on CPU, well under the 4-hour
   ceiling; nothing was cut.

### Carry-forward implications for October

- The frozen PPO foundation is usable: deterministic evaluation through the locked
  harness works end-to-end, artifact sets are complete, and multi-seed variance is now
  quantified. Forecasting and uncertainty-aware inputs should extend the observation
  pipeline built here, not replace it.
- The cost-vs-comfort trade recorded above is the baseline to beat: an uncertainty-aware
  controller should reduce the comfort term *without* paying ~70% more energy than B0,
  and the safety shield has a concrete target — the grid-limit exceedance counts above.
- No scenario/robustness work was attempted in this phase (out of scope per the plan).

## Supervisor update

> The standard PPO controller is trained and evaluated against the locked B0/B1/B2
> baselines across three seeds on the dev and final windows. Forecasting and the
> safety shield have not started; they will be built on this frozen PPO foundation.

## Verification status

- `./.venv/bin/python scripts/14_gate_week3.py` exits 0 (also via
  `/Volumes/code/rp/automation/check_week3.sh`).
- `./.venv/bin/python -m pytest -q` passes: 60 tests.
- `/Volumes/code/rp/automation/check_pytest.sh` exits 0.
