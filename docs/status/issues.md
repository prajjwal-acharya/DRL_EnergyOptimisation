# Issues, Anomalies, and Technical Debt

**Last updated:** 2 September 2026. None of these invalidate the results; all are worth
knowing before extending the code. Deep detail: [`research-log.md`](research-log.md) §7.

## Blockers

1. **No active implementation blocker.** Week 4 is complete. Week 5 remains unstarted and
   must preserve the mixed forecast result rather than assuming all selected intervals
   are calibrated. The earlier failed automation attempt remains historical provenance in
   [`phase-reviews/week4-5-status.md`](phase-reviews/week4-5-status.md).

## Known anomalies (verified against primary artifacts)

2. **Seed-recording inconsistency (week 3).** `run_metadata.json` for seeds 43/44 has the
   correct top-level `seed`, but the embedded `config_ppo_block.seed` reads 42 for all
   runs (the YAML hardcodes 42; run seed was a CLI override). Provenance wart only.
3. **Two comfort definitions coexist.** Derived `comfort_violation_hours` counts the
   **hot** band only (matches the CMDP constraint); CityLearn's `discomfort_proportion`
   counts hot + cold. B1/B2 show 0 derived violation hours while CityLearn reports
   ~97 % (cold) discomfort. Always state which definition a number comes from.
4. **Resolved Week-3 documentation errata:** the observation is now documented as 29
   slots and final grid exceedances as 123/82/102 for seeds 42/43/44.
5. **Reward punishes overheating only.** Cold-side discomfort costs nothing in the CMDP
   reward (only via the energy term) — which is exactly how B1/B2 end up over-cooling.
   Deliberate (June, cooling-dominated) but must be stated wherever comfort is reported.
6. **Legacy artifacts retained intentionally** (historical records, never pruned):
   `results/runs/baselines/b0_zero_actions/0-167/` (pre-naming-lock harness-regression run)
   and the `results/figures/0-167_*` set (pre-`dev_*` prefix). Superseded by
   `b0_neutral/dev` and the `dev_*` figures.

## Technical debt / small stuff

7. `train_stdout.log` exists only for seed 42 (seeds 43/44 have none).
8. The `cmdp-spec.md` hour annotation says h ∈ {0–23} while the dataset/observations run
   1–24 (the frozen normalisation offset 1 / scale 23 is correct).
9. The Week-4 gate now exists and passes; the Week-5 gate remains a Week-5 deliverable.
10. Resolved 2026-09-01: the agent-conductor wrapper layer (formerly `automation/`)
    was removed; this repo is executed by hand, so the space in the path no longer
    matters.
11. Fresh-start notes: this repository began with the single migration commit
    (`3d4cbdb`, September 2026) made with a **placeholder git identity** — set your real
    `user.name`/`user.email` and amend if that matters to you. Weeks 1–3 artifacts copied
    from the previous workspace retain their original commit hashes (`4c2c49f`-era for
    baselines, `fd52a53` for PPO), which refer to the previous repository's history; the
    week-3 gate accepts the pre-migration config hash (`1661674d…`) for exactly these runs
    (see `LEGACY_WEEK3_CONFIG_SHA256` in `scripts/standard_ppo/14_gate_week3.py`).
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

14. **Latent bug found by linting and fixed (2026-09-01):** `evaluation/artifacts.py`
    used `Optional` in the `write_run_artifacts` signature without importing it —
    never a runtime error only because `from __future__ import annotations` makes
    annotations lazy. Import added; six unused imports and one dead debug variable
    removed across `src/`, `scripts/`, `tests/`; `ruff.toml` added exempting
    tests/ and scripts/ from E402 (the intentional sys.path bootstrap). Ruff now
    passes clean repo-wide; test outcomes unchanged.

15. **Path-parts migration blind spot, found and fixed (2026-09-02):** the
    September-2026 outputs/→results/ migration replaced string literals
    (`"outputs/ppo"`), but nine executable lines in scripts 10–13 built the same
    path from Path parts (`PROJECT_ROOT / "outputs" / "ppo"`), which string
    replacement cannot see — docstrings were migrated, code was not. The first
    hand-run of the PPO layer therefore wrote seed 42 into a stray `outputs/`
    tree. All nine lines fixed, the generated seed-42 artifacts re-homed under
    `results/runs/ppo/seed42/`, and `11`/`12` re-run to regenerate their records
    with correct paths (values identical to the historical record). The seed-42
    *training* metadata (`run_metadata.json` from `10`) still records the old
    output directory as an informational field — cosmetic; retrain seed 42 only
    if a perfectly uniform artifact set is wanted. Lesson recorded: path
    migrations must grep for constructed paths, not just path strings.

16. **Week-4 plan boundary correction (2026-09-02):** the original plan claimed 480 × 3
    = 1,440 evaluated pairs, but horizons beyond row 719 have no truth labels. The code,
    plan, metadata, and gate now agree on 1,434 honest pairs (479/478/477 by horizon).

17. **Forecast calibration limitation:** only non-shiftable load has a selected calibrated
    learned model. Solar and cooling shipped persistence under the pre-registered fallback,
    so their interval widths are zero. The required review wording is explicitly qualified;
    Week 5 must retain this as a feature-degeneracy finding.

18. **Forecast cold start:** the first 24 control steps have no pre-dataset target history.
    `ForecastProvider` uses a documented current-value persistence fallback with
    degenerate intervals for steps 0–23, then switches to the selected models.
