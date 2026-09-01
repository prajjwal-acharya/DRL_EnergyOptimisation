# Current Stage

**Last updated:** 1 September 2026 (at the fresh-start migration to this repository).

## Where we are

Weeks 1–3 of the approved semester plan are **complete, verified, and committed**.
The 30-September mid-semester milestone needs only the forecasting module.

| Phase | Status | Evidence |
| --- | --- | --- |
| Week 1 — foundation (env, schema, smoke, literature) | ✅ complete | `status/phase-reviews/week1-review.md` |
| Week 2 — CMDP + locked harness + B0/B1/B2 baselines | ✅ complete | `status/phase-reviews/week2-review.md` |
| Week 3 — standard PPO (seeds 42/43/44), selection, final-window eval | ✅ complete | `status/phase-reviews/week3-review.md` |
| Week 4 — probabilistic forecasting module | ⏸ not started | `plans/week4-implementation-plan.md` (binding spec, ready) |
| Week 5 — uncertainty-aware PPO, RQ1 verdict | ⏸ not started | `plans/week5-implementation-plan.md` (chained behind week 4) |
| October — safety shield + robustness scenarios (RQ2/RQ3) | 🗓 planned | — |
| November — aggregation, dashboard, manuscript | 🗓 planned | — |

Weeks 4–5 were attempted on 26 Aug 2026 in the previous workspace and blocked on an
external model-API failure **before any file was written** — nothing was lost; the record
and resume procedure are in `status/phase-reviews/week4-5-status.md`.

## Implementations checklist

- [x] Pinned CityLearn 2.5.0 environment + local bootstrap, derived single-building schema
- [x] Name-addressed observation index (29 slots, no magic indices)
- [x] Locked evaluation harness (runner / metrics / artifacts) — B0 anchor regression at 1e-9
- [x] Controllers B0 neutral, B1 fixed-schedule, B2 tariff-aware (+ their 6 locked artifact sets)
- [x] Gymnasium RL adapter (frozen normalisation, CMDP reward, action mapping)
- [x] PPOController + frozen checkpoint-selection rule
- [x] 3 PPO training runs (seeds 42/43/44, 200k steps each, ~5 min/seed CPU)
- [x] 21-checkpoint evaluations per seed, final-window (0–719) evaluations, comparison tables
- [x] 60 contract tests; phase gates for weeks 1–3
- [ ] Forecasting package (`src/energy_optimisation/forecasting/` — reserved empty)
- [ ] `ForecastProvider` + backtest + frozen selection (Week 4)
- [ ] Observation variants plain/point/interval + RQ1 matched comparison (Week 5)
- [ ] Safety shield (`src/energy_optimisation/safety/` — reserved empty)

## State note (1 September 2026)

`results/` was intentionally cleared for a from-scratch regeneration by hand —
see `results/README.md` for the exact script order and runtimes. Until the
pipeline is re-run, the Week-1/2/3 verify gates fail by design (they check
on-disk evidence), as do the four tests that read the smoke/inspection
artifacts (`test_b0_matches_smoke_kpis`, `test_neutral_action_reproduces_b0_anchors`,
and the two inspection-position tests in `test_observation_names.py`); the
remaining 56 tests pass. The numbers in `results.md` below were produced by exactly
those scripts in the previous workspace and will reproduce.

## Next action

Execute `plans/week4-implementation-plan.md` top-to-bottom (phases A→D): forecasting
package, frozen `configs/week4-forecasting.yaml`, 12-fold rolling-origin backtest,
mechanical selection, `17_gate_week4.py` gate, `status/phase-reviews/week4-review.md`.
Then Week 5 per its plan. If resuming via the agent-conductor, see `automation/README.md`
(this repository path contains a space — a space-free symlink is required for missions).

**Claim discipline:** nothing so far is a savings claim. B0/B1/B2 is single-seed heuristic
evidence; week-3 PPO is three-seed evidence under one tariff profile; all of it will be
superseded by the planned multi-seed, multi-scenario evaluation.
