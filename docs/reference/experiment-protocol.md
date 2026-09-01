# Experiment Protocol

## Non-negotiable recording fields

Every experiment must record:

- Dataset or schema version
- Controller variant
- Configuration file
- Random seed
- Training and evaluation time ranges
- Output directory
- Git commit identifier

## Initial protocol

The Day 2 inspection uses CityLearn 2.5.0 data checked out from tag `v2.5.0`. It validates the parent scenario interface only and must not be reported as a research result.

The Day 3 smoke run uses `configs/smoke.yaml`, a derived `Building_1` schema, seed `42`, the CityLearn evaluation range `0-167`, and a deterministic zero-action controller. It runs to the environment terminal state, validates the full simulation path, and must not be reported as a research result.

## Week 2 baseline protocol

The Week 2 baseline evaluations use `configs/week2-baselines.yaml`: the derived `Building_1` schema, seed `42`, windows dev `0-167` and final `0-719`, and the deterministic controllers B0 (neutral), B1 (fixed schedule), and B2 (tariff-aware). Every run writes `run_metadata.json` (resolved config, seed, git commit), `trace.csv`, `district_kpis.csv`, and a run note under `results/runs/baselines/<controller>/<window>/`.

All artifacts are regenerable via `scripts/10_run_baselines.py` and `scripts/11_compare_baselines.py`; `scripts/12_verify_week2.py` is the phase gate. The comparison is single-seed heuristic evidence under one tariff profile and must not be reported as a savings claim.
