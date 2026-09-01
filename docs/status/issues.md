# Issues, Anomalies, and Technical Debt

**Last updated:** 1 September 2026. None of these invalidate the results; all are worth
knowing before extending the code. Deep detail: [`research-log.md`](research-log.md) §7.

## Blockers

1. **Weeks 4–5 never executed.** The week-4 autonomous mission blocked on 26 Aug 2026 at
   its first worker dispatch — the daemon's model route (`stealth/ox-alpha`) was retired
   by the provider (HTTP 404), exhausting the retry cap before any file was written. The
   chain script correctly refused to start week 5. Root cause, timeline, and the resume
   procedure: [`phase-reviews/week4-5-status.md`](phase-reviews/week4-5-status.md).
   The 30-Sep milestone needs only this forecasting module.

## Known anomalies (verified against primary artifacts)

2. **Seed-recording inconsistency (week 3).** `run_metadata.json` for seeds 43/44 has the
   correct top-level `seed`, but the embedded `config_ppo_block.seed` reads 42 for all
   runs (the YAML hardcodes 42; run seed was a CLI override). Provenance wart only.
3. **Two comfort definitions coexist.** Derived `comfort_violation_hours` counts the
   **hot** band only (matches the CMDP constraint); CityLearn's `discomfort_proportion`
   counts hot + cold. B1/B2 show 0 derived violation hours while CityLearn reports
   ~97 % (cold) discomfort. Always state which definition a number comes from.
4. **Erratum in the week-3 review:** final-window grid exceedances are written
   "123/102/82 (seeds 42/43/44)"; the artifacts say **123/82/102** (seeds 43/44
   transposed). To be fixed in the Week-4 commit alongside the two "49-dim → 29" doc
   corrections scheduled there.
5. **Reward punishes overheating only.** Cold-side discomfort costs nothing in the CMDP
   reward (only via the energy term) — which is exactly how B1/B2 end up over-cooling.
   Deliberate (June, cooling-dominated) but must be stated wherever comfort is reported.
6. **Legacy artifacts retained intentionally** (historical records, never pruned):
   `results/runs/baselines/b0_zero_actions/0-167/` (pre-naming-lock harness-regression run)
   and the `results/figures/0-167_*` set (pre-`dev_*` prefix). Superseded by
   `b0_neutral/dev` and the `dev_*` figures.

## Technical debt / small stuff

7. `train_stdout.log` exists only for seed 42 (seeds 43/44 have none).
8. Stale wording: "49-dim" observation in two week-3 docs (correct = 29 slots); the
   `cmdp-spec.md` hour annotation says h ∈ {0–23} while the dataset/observations run
   1–24 (the frozen normalisation offset 1 / scale 23 is correct).
9. `automation/check_week4.sh` / `check_week5.sh` fail by design until
   `scripts/17_gate_week4.py` / `20_gate_week5.py` exist (they are phase deliverables).
10. This repository path contains a space ("Semester 7") — agent-conductor missions need
    a space-free symlink to run verification wrappers (see `automation/README.md`).
11. Fresh-start notes: this repository began with the single migration commit
    (`3d4cbdb`, September 2026) made with a **placeholder git identity** — set your real
    `user.name`/`user.email` and amend if that matters to you. Weeks 1–3 artifacts copied
    from the previous workspace retain their original commit hashes (`4c2c49f`-era for
    baselines, `fd52a53` for PPO), which refer to the previous repository's history; the
    week-3 gate accepts the pre-migration config hash (`1661674d…`) for exactly these runs
    (see `LEGACY_WEEK3_CONFIG_SHA256` in `scripts/14_gate_week3.py`).
12. `.pytest_cache` from the previous workspace listed 62 tests vs the current 60
    (not copied here; harmless wherever it regenerates).

13. **Suppressed third-party warnings in pytest.ini (2026-09-01).** All 15 former
    pytest warnings were upstream import-time noise, not project code: the
    urllib3/LibreSSL notice (macOS system Python 3.9's ssl module; the project
    makes no runtime network calls) and 14 pyparsing-deprecation notices raised
    by matplotlib 3.9.4's own internals (fixed upstream only in matplotlib 3.10+,
    which needs Python >= 3.10 — incompatible with the frozen stack). Both are
    `ignore`d in `pytest.ini` with the rationale inline; revisit when the pinned
    environment is ever deliberately upgraded.
