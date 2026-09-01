"""Artifact writers for baseline evaluation runs.

Plan reference: docs/plans/week2-implementation-plan.md §B3. Every run persists four
files under ``results/runs/baselines/<controller>/<window>/``: run_metadata.json,
trace.csv, district_kpis.csv, and a README.md run note stating purpose,
controller, seed, and window. Nothing under results/ is hand-edited; these
writers are the single source of run artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd


RUN_METADATA_FILE = "run_metadata.json"
TRACE_FILE = "trace.csv"
DISTRICT_KPIS_FILE = "district_kpis.csv"
DERIVED_METRICS_FILE = "derived_metrics.json"
README_FILE = "README.md"

REQUIRED_ARTIFACT_FILES = (
    RUN_METADATA_FILE,
    TRACE_FILE,
    DISTRICT_KPIS_FILE,
    README_FILE,
)


def _district_kpis_frame(kpis: Mapping[str, float]) -> pd.DataFrame:
    """Rebuild the district-level KPI frame written next to each trace."""

    return pd.DataFrame(
        {
            "cost_function": list(kpis.keys()),
            "value": [float(value) for value in kpis.values()],
            "name": ["District"] * len(kpis),
            "level": ["district"] * len(kpis),
        }
    )


def _render_readme(
    metadata: Mapping[str, Any],
    window_label: str,
    has_derived_metrics: bool = False,
) -> str:
    purpose = str(metadata.get("purpose", "research result"))
    artifact_names = [RUN_METADATA_FILE, TRACE_FILE, DISTRICT_KPIS_FILE]
    if has_derived_metrics:
        artifact_names.append(DERIVED_METRICS_FILE)
    lines = [
        f"# Run note: {metadata['controller']} ({window_label})",
        "",
        f"- Purpose: {purpose}",
        f"- Controller: {metadata['controller']}",
        f"- Seed: {metadata['seed']}",
        f"- Window: simulation time steps "
        f"{metadata['simulation_start_time_step']}-{metadata['simulation_end_time_step']}",
        f"- Schema: `{metadata['schema']}`",
        f"- Git commit: {metadata.get('git_commit') or 'unavailable'}",
        f"- Completed steps: {metadata['completed_steps']}",
        f"- Generated (UTC): {metadata['generated_at_utc']}",
        "",
        f"Artifacts: {', '.join(f'`{name}`' for name in artifact_names)}.",
        "Written by the evaluation harness; files under `results/` are never hand-edited.",
        "",
    ]
    return "\n".join(lines)


def write_run_artifacts(
    run_directory: str | Path,
    *,
    metadata: Mapping[str, Any],
    kpis: Mapping[str, float],
    trace: pd.DataFrame,
    window_label: str = "window",
    derived_metrics: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write the complete artifact set for one run and return the directory.

    Parameters
    ----------
    run_directory:
        Destination directory (created if missing), e.g.
        ``results/runs/baselines/<controller>/<window>``.
    metadata:
        Provenance mapping from :func:`evaluation.runner.build_run_metadata`.
    kpis:
        District KPI name/value pairs from :func:`evaluation.runner.run_episode`.
    trace:
        Per-step trace DataFrame with columns ``TRACE_COLUMNS``.
    window_label:
        Human-readable window name used in the README run-note heading.
    derived_metrics:
        Optional §B4 metrics (comfort violations, SoC extremes, clipping /
        reserve event counts, peak, solar self-consumption, grid-limit
        exceedances) persisted as ``derived_metrics.json`` when provided.
    """

    directory = Path(run_directory)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / RUN_METADATA_FILE).write_text(json.dumps(dict(metadata), indent=2) + "\n")
    trace.to_csv(directory / TRACE_FILE, index=False)
    _district_kpis_frame(kpis).to_csv(directory / DISTRICT_KPIS_FILE, index=False)
    if derived_metrics is not None:
        (directory / DERIVED_METRICS_FILE).write_text(
            json.dumps(dict(derived_metrics), indent=2) + "\n"
        )
    (directory / README_FILE).write_text(
        _render_readme(metadata, window_label, has_derived_metrics=derived_metrics is not None)
    )
    return directory
