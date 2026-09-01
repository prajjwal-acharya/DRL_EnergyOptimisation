# Script Guide — What Each Script Is For

Numbering is sequential in execution order — also the recommended reading
order — and scripts live in phase subfolders: `foundation/` (01–05, Week 1) ·
`cmdp_baselines/` (06–08, Week 2) · `standard_ppo/` (09–14, Week 3) ·
`forecasting/` (15–17, Week 4, planned) · `uncertainty_aware_ppo/`
(18–20, Week 5, planned). The gate names (`gate_weekN`) carry the phase, so
the numbers never skip.

All scripts run from the repository root with the project venv
(`source .venv/bin/activate`, or `.venv/bin/python scripts/<name>`).
None of them takes a research decision for you — constants live in `configs/`,
never in code.

## Week 1 — Foundation

| Script | Significance | Writes | Runtime |
| --- | --- | --- | --- |
| `01_fetch_pinned_dataset.py` | Guarantees the exact dataset this project consumes: verifies the pinned `citylearn_challenge_2023_phase_1` payload (file set, 721-line CSVs, provenance) is on disk; with `--fetch`, clones CityLearn tag `v2.5.0` to a throwaway temp dir, checks the pinned commit, and extracts only the dataset. | `data/raw/citylearn_challenge_2023_phase_1/` | seconds (fetch: network) |
| `02_derive_building_schema.py` | Derives the CP-I single-building scenario (`Building_1` only, central agent) from the read-only parent schema, without touching raw data. The derived schema is the environment contract every run loads. | `configs/schema-building1.json` | seconds |
| `03_inspect_environment.py` | Interface evidence: obs/action spaces, episode length, assets, KPIs of the parent scenario — the recorded basis for the environment-selection decision. | `results/inspection/…json` | seconds |
| `04_run_smoke_test.py` | Proves the full path end-to-end (load → reset → zero actions → terminal → artifacts). Infrastructure check only — **not a research result**. | `results/runs/smoke/` | seconds |
| `05_gate_week1.py` | Week-1 phase gate: repository files, README content, literature thresholds, schema shape, smoke evidence, CityLearn version. | — (exit code) | seconds |

## Week 2 — CMDP + Deterministic Baselines

| Script | Significance | Writes | Runtime |
| --- | --- | --- | --- |
| `06_run_baselines.py` | Runs B0 (neutral), B1 (fixed schedule), B2 (tariff-aware) on both locked windows strictly from `configs/week2-baselines.yaml` through the locked harness. The deterministic comparison evidence. | `results/runs/baselines/<ctrl>/<window>/` | minutes |
| `07_compare_baselines.py` | Builds the Week-2 comparison table and the four figure kinds per window from the run artifacts (never re-runs the env). | `results/tables/baseline_comparison.csv`, `results/figures/{dev,final}_*` | seconds |
| `08_gate_week2.py` | Week-2 phase gate: artifact completeness, B0 regression vs the six smoke anchors at 1e-9, frozen config keys, forbidden-import guard, review doc. | — (exit code) | seconds |

## Week 3 — Standard PPO

| Script | Significance | Writes | Runtime |
| --- | --- | --- | --- |
| `09_compute_normalization_stats.py` | One-shot generator of `configs/week3-ppo.yaml`'s frozen normalisation block (per-feature offset/scale from the B0 dev trace + schema static ranges). Already executed — the config in the repo is the frozen result; re-running is idempotent verification. | `configs/week3-ppo.yaml` | seconds |
| `10_train_ppo.py` | Trains the standard PPO controller for one seed on the Gymnasium adapter under the frozen config; checkpoints every 10k steps; enforces sanity gates (NaN-free monitor, zero pre-clip violations, CPU-only) and records full provenance. Run per seed: `--seed 42`, then 43, 44. | `results/runs/ppo/seed<seed>/` + return-curve figure | ~5 min/seed |
| `11_evaluate_checkpoints.py` | Pushes every checkpoint of one seed through the **locked week-2 harness** on the dev window and executes the frozen selection rule (lowest dev cost, tie-break lower discomfort). Learning-curve + selection evidence. | `results/runs/ppo/seed<seed>/evaluations.csv`, `selected_checkpoint.json`, KPI-curve figure | minutes |
| `12_evaluate_final_window.py` | Evaluates each seed's selected checkpoint on the held-out final window (0–719) with complete artifact sets — the controller evidence that was never selected on. | `results/runs/ppo/seed<seed>/final/`, `results/runs/ppo/final_window_summaries.json` | minutes |
| `13_compare_ppo.py` | Builds the multi-seed summary and the PPO-vs-baselines table/figure (baseline table consumed read-only). | `results/tables/ppo_*.csv`, `results/figures/ppo_vs_baselines_cost.png` | seconds |
| `14_gate_week3.py` | Week-3 phase gate: config SHA vs run metadata (incl. the documented migration legacy hash), independent re-execution of the selection rule, final-window artifact shape, week-2 evidence byte-identity, review doc. | — (exit code) | seconds |

## Weeks 4–5 — Planned (not yet implemented)

| Script (planned) | Significance |
| --- | --- |
| `15_train_forecasters.py` | Backtests the five-rung forecasting ladder (persistence ×2, climatology, linear quantile, GRU quantile) per target under the frozen 12-fold rolling-origin scheme. Spec: `docs/plans/week4-implementation-plan.md`. |
| `16_compare_forecasters.py` | Executes the frozen selection rule mechanically → `selected_models.json`, forecast tables/figures. |
| `17_gate_week4.py` | Week-4 phase gate (config hash, 480×3 row arithmetic, quantile monotonicity, weeks 1–3 byte-identity). |
| `18_compute_forecast_feature_stats.py` | One-shot generator of the frozen forecast-feature normalisation for both Week-5 arms. Spec: `docs/plans/week5-implementation-plan.md`. |
| `19_compare_rq1.py` | The RQ1 matched-pair comparison + pre-registered verdict (`rq1_verdict.json`). |
| `20_gate_week5.py` | Week-5 phase gate (arm-config identity, feature-matrix hashes, verdict rule hash). |

## Conventions

- The agent-conductor wrapper layer was removed (2026-09-01); the gates are run
  directly: `python scripts/<phase-folder>/<NN>_gate_weekN.py`.
- **Research result?** `10/11`, `21/22/23/24` produce research evidence; everything
  else is infrastructure or verification (same convention as the README table).
- Nothing under `results/` is ever hand-edited — regenerate from these scripts only.
