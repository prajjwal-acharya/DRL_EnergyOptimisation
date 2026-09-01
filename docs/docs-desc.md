# Documentation Index

Reading order for a newcomer — each layer assumes the previous one.

1. [`README.md`](../README.md) — project identity, research questions, scope, quickstart.
2. **Status (where we are):**
   1. [`status/current-stage.md`](status/current-stage.md) — implementations checklist and next action.
   2. [`status/results.md`](status/results.md) — every number and finding so far.
   3. [`status/issues.md`](status/issues.md) — blockers, anomalies, errata, tech debt.
   4. [`status/research-log.md`](status/research-log.md) — the complete cross-phase narrative (the deep record).
   5. [`status/phase-reviews/`](status/phase-reviews/) — frozen per-week completion records.
3. **Reference (how things work — stable, changes rarely):**
   1. [`reference/folder-map.md`](reference/folder-map.md) — what lives where in this repo, and why.
   2. [`reference/dataset.md`](reference/dataset.md) — the CityLearn dataset: files, columns, quirks.
   3. [`reference/cmdp-spec.md`](reference/cmdp-spec.md) — the problem formulation and frozen constants.
   4. [`reference/environment-selection.md`](reference/environment-selection.md) — why this scenario was chosen.
   5. [`reference/experiment-protocol.md`](reference/experiment-protocol.md) — recording protocol per experiment.
   6. [`reference/literature.md`](reference/literature.md) + [`literature-matrix.csv`](reference/literature-matrix.csv) — evidence base.
4. **Plans (what we intend to do — binding once written):**
   - [`plans/semester-plan.pdf`](plans/semester-plan.pdf) — the approved CP-I research plan.
   - [`plans/week1-brief.md`](plans/week1-brief.md), [`plans/week2-brief.md`](plans/week2-brief.md) — original weekly briefs.
   - [`plans/week2-implementation-plan.md`](plans/week2-implementation-plan.md) … [`plans/week5-implementation-plan.md`](plans/week5-implementation-plan.md) — binding per-phase specs.

## Conventions

- **Update discipline:** `status/` changes every phase; `reference/` only when understanding
  changes; `plans/` are frozen once the phase starts.
- **Migration note:** this repository is a polished restructure of the original workspace
  (`/Volumes/code/Research Project`). Weeks 1–3 evidence was produced there and copied here
  with paths updated; see the provenance notes in `status/research-log.md` and
  `status/phase-reviews/week4-5-status.md`.
