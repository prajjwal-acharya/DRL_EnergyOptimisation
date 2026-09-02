# Current Stage

**Last updated:** 2 September 2026 (after the Week-4 forecasting gate).

## Where we are

Weeks 1–4 of the approved semester plan are implemented and verified. The
30-September forecasting milestone now has rolling-origin accuracy and calibration
evidence; Week 5 is the next implementation boundary.

| Phase | Status | Evidence |
| --- | --- | --- |
| Week 1 — foundation (env, schema, smoke, literature) | ✅ complete | `status/phase-reviews/week1-review.md` |
| Week 2 — CMDP + locked harness + B0/B1/B2 baselines | ✅ complete | `status/phase-reviews/week2-review.md` |
| Week 3 — standard PPO (seeds 42/43/44), selection, final-window eval | ✅ complete | `status/phase-reviews/week3-review.md` |
| Week 4 — probabilistic forecasting module | ✅ complete | `phase-reviews/week4-review.md` |
| Week 5 — uncertainty-aware PPO, RQ1 verdict | ⏭ next | `plans/week5-implementation-plan.md` |
| October — safety shield + robustness scenarios (RQ2/RQ3) | 🗓 planned | — |
| November — aggregation, dashboard, manuscript | 🗓 planned | — |

The failed 26 Aug automation attempt remains as historical provenance in
`phase-reviews/week4-5-status.md`; Week 4 was subsequently implemented directly in this
repository on 2 September.

## Implementations checklist

- [x] Pinned CityLearn 2.5.0 environment + local bootstrap, derived single-building schema
- [x] Name-addressed observation index (29 slots, no magic indices)
- [x] Locked evaluation harness (runner / metrics / artifacts) — B0 anchor regression at 1e-9
- [x] Controllers B0 neutral, B1 fixed-schedule, B2 tariff-aware (+ their 6 locked artifact sets)
- [x] Gymnasium RL adapter (frozen normalisation, CMDP reward, action mapping)
- [x] PPOController + frozen checkpoint-selection rule
- [x] 3 PPO training runs (seeds 42/43/44, 200k steps each, ~5 min/seed CPU)
- [x] 21-checkpoint evaluations per seed, final-window (0–719) evaluations, comparison tables
- [x] 72 contract tests; phase gates for weeks 1–4
- [x] Modular forecasting package: causal data, metrics, model ladder, orchestration, API
- [x] `ForecastProvider` + 12-fold backtest + frozen selection (Week 4)
- [ ] Observation variants plain/point/interval + RQ1 matched comparison (Week 5)
- [ ] Safety shield (`src/energy_optimisation/safety/` — reserved empty)

## State note (2 September 2026)

The generated Week 1–4 evidence is present. Week 4 added 21 prediction/metric artifact
pairs (seven variants × three targets), selection/refit records, two tables, and five
figures. The backtest uses 1,434 valid labelled pairs per model/target; the last three
origins cannot supply every horizon because the dataset ends at row 719.

## Next action

Execute `plans/week5-implementation-plan.md` top-to-bottom. Both Week-5 arms must consume
the same frozen Week-4 artifacts: linear quantile regression for non-shiftable load, 24-hour persistence for
solar, and last-value persistence for cooling. The two fallback intervals are degenerate;
Week 5 must report that diagnostic rather than treating them as calibrated uncertainty.

**Claim discipline:** nothing so far is a savings claim. B0/B1/B2 is single-seed heuristic
evidence; week-3 PPO is three-seed evidence under one tariff profile; all of it will be
superseded by the planned multi-seed, multi-scenario evaluation.
