# Automation — Agent-Conductor Integration

This folder holds the glue for running implementation weeks as autonomous
agent-conductor missions, plus the phase-verification wrappers they call.

## Files

| File | Purpose |
| --- | --- |
| `check_pytest.sh` | cd to repo root, run `.venv/bin/python -m pytest -q` |
| `check_week1.sh` … `check_week5.sh` | cd to repo root, run the phase gate (`scripts/05/08/14/17/20_gate_week*.py`) |
| `chain_week4_then_week5.sh` | start the week-4 mission, poll status every 60 s, hard-gate on `check_week4.sh`, then start week 5. Stops and refuses week 5 if week 4 is blocked/failed. |

## Before running missions against THIS repository

1. **The path contains a space** (`…/Semester 7/RL-Forecast-Energy`). Conductor
   verification commands are split on spaces, so the wrappers must be reached through a
   space-free symlink, e.g.:

   ```bash
   ln -s "/Users/prajjwalacharya/Documents/Semester 7/RL-Forecast-Energy" /Volumes/rfe
   # then reference /Volumes/rfe/automation/check_week4.sh in mission verification commands
   ```

2. **Create fresh missions** — the historical week-2/3/4/5 mission state lives in the
   previous workspace (`.agent-conductor/missions/…` there) and their recorded commands
   still point at the old paths and the retired `stealth/ox-alpha` model route. The
   week-4 mission objective text is otherwise reusable: it implements
   `docs/plans/week4-implementation-plan.md`, whose script/config/output references were
   migrated to this repository's layout in September 2026.

3. **Use a live model route** — the previous week-4 attempt blocked because
   `stealth/ox-alpha` had been retired (OpenRouter 404). Verify the route before launch.

4. Stop procedure (unchanged): `conductor daemon stop <mission-id>` for each mission, plus
   `pkill -f chain_week4_then_week5.sh`; then remove any stale `.pause.request` files in
   the mission dirs before restarting.

The full blocked-mission post-mortem and resume checklist:
[`../docs/status/phase-reviews/week4-5-status.md`](../docs/status/phase-reviews/week4-5-status.md).
