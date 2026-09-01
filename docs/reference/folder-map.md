# Folder Map — What Lives Where and Why

The one design rule: **authored files are git-tracked; generated files are disposable.**
`src/ + configs/ + scripts/ + tests/ + docs/` reproduce every number in `results/`;
`data/raw/` is pinned external input. Everything else can be deleted and rebuilt.

```text
RL-Forecast-Energy/
├── README.md            entry point: identity, RQs, status, quickstart
├── requirements.txt     pinned dependencies (CityLearn 2.5.0, SB3 2.3.2, torch 2.8.0, …)
├── pytest.ini           test discovery (tests/)
├── .gitignore           keeps data/, results/, .venv/ out of git
│
├── docs/                all human knowledge — split by how often it changes
│   ├── README.md          this index's index (reading order)
│   ├── plans/             INTEND: semester plan PDF, weekly briefs, binding phase specs
│   ├── reference/         KNOW: folder map (this file), dataset, CMDP spec, protocol, literature
│   └── status/            ARE: current stage, results, issues + research-log + phase-reviews/
│
├── src/energy_optimisation/    the library — all reusable logic, no scripts
│   ├── environment.py            CityLearn loading, inspection, schema derivation
│   ├── observation_names.py      name→index observation map (no magic indices anywhere)
│   ├── baselines/controllers.py  Controller ABC + B0/B1/B2 decision rules
│   ├── evaluation/               runner.py (locked harness), metrics.py, artifacts.py
│   ├── rl/                       env_adapter.py (Gymnasium + reward + normalisation),
│   │                             controller.py (PPOController), checkpoint_selection.py
│   ├── forecasting/            EMPTY — Week 4 target
│   └── safety/                 EMPTY — Week 6+ shield target
│
├── configs/             frozen experiment definitions ("logic-as-data")
│   ├── smoke.yaml         Week-1 smoke-run configuration
│   ├── week2-baselines.yaml  windows, tariff τ, reward weights, B1/B2 rules, grid limit
│   ├── week3-ppo.yaml        all PPO hyperparameters + 29-feature normalisation stats
│   └── schema-building1.json derived single-building CityLearn schema
│
├── scripts/             runnable commands — numbered by phase
│   ├── 01–05_fetch dataset / derive schema / inspect / smoke / gate   Week 1
│   ├── 10–12_run / compare / gate                                 Week 2 baselines
│   ├── 20–25_norm stats / train / eval ckpts / eval final / compare / gate     Week 3
│   └── 30–32, 40–43                                               Week 4/5 (planned)
│
├── tests/               60 contract tests (B0-anchor 1e-9 regression lives in test_rl_env.py)
│
├── data/
│   ├── README.md          provenance and rules for this folder
│   └── raw/citylearn_challenge_2023_phase_1/   the pinned dataset payload (read-only, ~0.5 MB,
│                                              fetched/verified by scripts/01_fetch_pinned_dataset.py)
│
├── results/             generated evidence — never hand-edited, always regenerable
│   ├── runs/               baselines/ · ppo/ · smoke/ — per-run artifact sets
│   │                       (run_metadata.json, trace.csv, district_kpis.csv,
│   │                        derived_metrics.json, README.md)
│   ├── inspection/         environment interface evidence
│   ├── tables/             the comparison CSVs (= "the results")
│   └── figures/            all PNGs
│
└── automation/          agent-conductor wrappers (check_*.sh), the week4→week5 chain
                         script, and notes on running missions against this repo
```

## What goes where — quick rules

| Thing | Goes in | Not in |
| --- | --- | --- |
| Reusable logic | `src/` | `scripts/` |
| A runnable command | `scripts/` (numbered) | anywhere else |
| An experiment decision/constant | `configs/` (frozen, named) | code |
| A number or figure | `results/` (regenerated) | git |
| How something works | `docs/reference/` | README (keep it short) |
| What happened this phase | `docs/status/` | `docs/plans/` |
| What we will do | `docs/plans/` | `docs/status/` |
| Raw external data | `data/raw/` (never edited) | `data/processed/` (only if derived data appears) |
