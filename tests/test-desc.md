# Test Guide — What Each Test File Guarantees

66 test functions → 72 collected pytest items. Run: `python -m pytest -q`.
These are **contract tests**: they pin the project's invariants so that neither
code drift nor accidental edits can silently change what a number means.
Four of them read generated evidence under `results/` (noted below) — they pass
once the pipeline in `results/results-desc.md` has been re-run.

| File | Tests | Guarantees |
| --- | --- | --- |
| `test_environment.py` | 3 | CityLearn loading utilities: `describe_space` returns JSON-safe Box bounds; `neutral_actions` yields zeros clipped into each action space; env loading is CWD-independent (relative/null `root_directory` absolutized). |
| `test_observation_names.py` | 5 | The name→index map is *the* observation truth: layout reproduces the parent-scenario inspection JSON positions (49-dim parent, 29-dim single-building); the frozen Building_1 index matches the live environment's observation space; the mapping is immutable (`MappingProxyType`); key control observations (hour, price, SoCs, indoor temp, setpoint, solar) are present and distinct. |
| `test_controllers.py` | 14 fns / 20 items | The B0/B1/B2 decision rules exactly as frozen in `configs/week2-baselines.yaml`: shapes/dtype/finiteness under random observations (×3 controllers); B0 always zero; B1 follows the calendar bands and is price-blind; B2 discharges ≥ τ, charges < τ, τ is inclusive, SoC reserve band respected (edges inclusive), off-peak cooling equals B1's bands; determinism under seed (×3); baselines source free of forbidden imports; full dev-window runs complete 167 steps, NaN-free, with zero clipping through the real environment (×3). |
| `test_metrics.py` | 6 | Derived-metric arithmetic on hand-built traces: hot-side-only comfort counting; SoC min/max and peak; clipping-event counting; reserve-event counting; solar self-consumption (incl. the 0.8 case and grid-limit exceedances, zero-generation edge); missing required columns raise `KeyError`. |
| `test_runner.py` | 6 | The locked harness: a run writes the complete 5-file artifact set with correct metadata (relative schema path, 40-char git commit, step count); identical seeds reproduce KPIs; traces are NaN-free; **B0 through the harness reproduces all six smoke anchors at 1e-9**; invalid windows are rejected; config snapshots land in metadata. *(Two tests read the smoke/B0 evidence under `results/`.)* |
| `test_rl_env.py` | 9 | The Gymnasium adapter: observation/action space shapes and finiteness; deterministic reset per seed; episode length equals the window; **the neutral RL action `[-1,-1,-1]` reproduces the B0 smoke anchors at 1e-9** (the week-3↔week-2 bridge); the `[a0, a1, (a2+1)/2]` action mapping bounds; the reward equals the frozen CMDP formula on hand-computed triples (+ division-guard); pre-clip violations are counted; window overrides/invalid windows; normalisation spot-checks against the frozen `(offset, scale)` pairs. |
| `test_ppo_controller.py` | 12 | The PPO wrapper and selection rule: `act` is deterministic and equals the mapped policy output; the frozen affine action mapping; normalisation matches the frozen transform (float32, [0,1], identity SoCs) and saturates outside frozen ranges; observation-dimension mismatch raises; reset is stateless; reward-constant extraction; `episode_return_from_trace` hand-computed; the frozen selection rule (lowest cost → discomfort tie-break → deterministic on full tie → missing columns raise). |
| `test_forecasting.py` | 11 | Week-4 contracts: source alignment, no-lookahead mutation proof, exact folds, hand-computed metrics, monotone quantiles, persistence/climatology definitions, deterministic linear/GRU fits, conformal widening, provider causality, and 9/36-value Week-5 feature blocks. |

## The two anchors worth knowing

- `test_b0_matches_smoke_kpis` (runner) and `test_neutral_action_reproduces_b0_anchors`
  (adapter) both pin the **same six smoke KPIs at 1e-9** from two different code
  paths — this is what proves the evaluation harness, the RL adapter, and the
  raw environment all measure the same world. If either fails after a code
  change, the change altered measurement semantics, not just behaviour.
- The four evidence-reading tests (`test_b0_matches_smoke_kpis`,
  `test_neutral_action_reproduces_b0_anchors`, and the two inspection-position
  tests in `test_observation_names.py`) require `results/runs/smoke/` and
  `results/inspection/` to exist — currently cleared for regeneration
  (`results/results-desc.md`); the other 56 pass regardless.

## Conventions

- Tests self-manage `sys.path` (`PROJECT_ROOT/src`); no package install needed.
- No test trains anything, touches `configs/`, or writes into `results/`
  (runs are executed into temp directories).
- Phase gates (`scripts/**/05/08/14/17_gate_week*.py`) run this suite *plus*
  artifact-level checks — tests verify code, gates verify evidence.
