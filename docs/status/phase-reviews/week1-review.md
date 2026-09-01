# Week 1 Review

**Review date:** 18 August 2026

**Scope:** project foundation, CityLearn environment selection, smoke evidence, and initial literature evidence.
**Research-result status:** no controller was trained and no performance claim was made.

## Completion record

| Week 1 outcome | Evidence | Result |
| --- | --- | --- |
| Reproducible project structure | Versioned source, configuration, documentation, tests, and pinned dependencies | Pass |
| CityLearn installation | `CityLearn==2.5.0` imported and the local CLI version command succeeded | Pass |
| Scenario loading | Tag-pinned CityLearn v2.5.0 source plus the locally generated `Building_1` schema reset successfully | Pass |
| Interface understood | One central observation agent; three valid actions: DHW storage, electrical storage, and cooling device | Pass |
| End-to-end smoke run | Deterministic zero actions completed 167 simulator actions and wrote metadata, KPI CSV, plot, and run note | Pass |
| Research questions and scope | `README.md` records RQ1-RQ3, metrics, non-goals, and reproducibility rules | Pass |
| Literature evidence base | 12 sources; 6 analysed sources explicitly mapped to RQ1-RQ3; four testable gaps recorded | Pass |
| Initial environment decision | `docs/reference/environment-selection.md` documents rejecting the 2020 example and selecting the 2023 Phase 1 parent scenario | Pass |
| Source-control boundary | Virtual environment, downloaded CityLearn data, and generated outputs excluded from Git | Pass |

## Reproducible verification sequence

Run the following from the repository root after installing `requirements.txt`:

```bash
python scripts/foundation/01_fetch_pinned_dataset.py --skip-clone
python scripts/foundation/02_derive_building_schema.py
python scripts/foundation/03_inspect_environment.py
python scripts/foundation/04_run_smoke_test.py
python -m pytest -q
python scripts/foundation/05_gate_week1.py
```

`scripts/foundation/05_gate_week1.py` checks the project structure, fixed research questions, source-control boundary, literature thresholds, single-building action interface, and terminal smoke evidence. It is deliberately not a research evaluation.

## Named-dataset API note

The Week 1 plan originally used CityLearn's named-dataset download path. In this environment, that path depends on GitHub's anonymous API and returned a rate-limit response. The project therefore uses the exact CityLearn `v2.5.0` source tag and commit recorded in `configs/smoke.yaml`, derives the CP-I schema locally, and primes the required CityLearn cache files. This preserves the exact scenario definition while removing a network-rate-limit dependency from reproduction.

The local pinned-schema reset and smoke simulation are the completion evidence; a successful anonymous named-dataset API request is intentionally not claimed.

## Handoff to Week 2

The next implementation step is the plan-approved CMDP specification and a tariff-aware rule-based baseline. PPO, interval forecasting, and the safety shield remain out of scope until that baseline has a reproducible evaluation path.
