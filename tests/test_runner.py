"""Phase B harness tests, including the B0 regression against smoke anchors.

Plan reference: docs/plans/week2-implementation-plan.md §B tests. The zero-action
controller used here is the §C0 B0 policy expressed through the shared
interface so the harness can be validated before B1/B2 exist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energy_optimisation.baselines.controllers import Controller
from energy_optimisation.evaluation.artifacts import REQUIRED_ARTIFACT_FILES
from energy_optimisation.evaluation.runner import (
    TRACE_COLUMNS,
    build_run_metadata,
    run_episode,
)


SCHEMA_PATH = PROJECT_ROOT / "configs/schema-building1.json"
SMOKE_ANCHORS_PATH = PROJECT_ROOT / "results/runs/smoke/district_kpis.csv"
SEED = 42
SIMULATION_START_TIME_STEP = 0
SIMULATION_END_TIME_STEP = 167
EXPECTED_STEPS = SIMULATION_END_TIME_STEP - SIMULATION_START_TIME_STEP
TOLERANCE = 1e-9

EXCLUDED_EMPTY_KPIS = (
    "one_minus_thermal_resilience_proportion",
    "power_outage_normalized_unserved_energy_total",
)


class ZeroActionController(Controller):
    """B0 policy: request zero action every step."""

    name = "b0_zero_actions"

    def act(self, observation: np.ndarray) -> np.ndarray:
        return np.zeros(3, dtype=float)


@pytest.fixture(scope="module")
def b0_dev_run():
    kpis, trace = run_episode(
        ZeroActionController(),
        SCHEMA_PATH,
        simulation_start_time_step=SIMULATION_START_TIME_STEP,
        simulation_end_time_step=SIMULATION_END_TIME_STEP,
        seed=SEED,
    )
    return kpis, trace


@pytest.fixture(scope="module")
def smoke_anchors() -> dict[str, float]:
    frame = pd.read_csv(SMOKE_ANCHORS_PATH, keep_default_na=False, float_precision="round_trip")
    district = frame.loc[frame["level"] == "district"]
    return {
        str(cost_function): float(value)
        for cost_function, value in zip(district["cost_function"], district["value"])
        if str(value).strip() != ""
    }


def test_run_produces_all_artifacts(tmp_path: Path) -> None:
    from energy_optimisation.evaluation.runner import run_and_record

    output_directory = tmp_path / "baselines" / ZeroActionController.name / "dev"
    kpis, trace = run_and_record(
        ZeroActionController(),
        SCHEMA_PATH,
        output_directory=output_directory,
        simulation_start_time_step=SIMULATION_START_TIME_STEP,
        simulation_end_time_step=SIMULATION_END_TIME_STEP,
        seed=SEED,
    )

    for file_name in REQUIRED_ARTIFACT_FILES:
        assert (output_directory / file_name).is_file(), f"missing artifact {file_name}"

    metadata = json.loads((output_directory / "run_metadata.json").read_text())
    assert metadata["controller"] == ZeroActionController.name
    assert metadata["seed"] == SEED
    assert metadata["simulation_start_time_step"] == SIMULATION_START_TIME_STEP
    assert metadata["simulation_end_time_step"] == SIMULATION_END_TIME_STEP
    assert metadata["completed_steps"] == EXPECTED_STEPS == len(trace)
    assert not metadata["schema"].startswith(str(PROJECT_ROOT)), "schema path must be project-relative"
    assert metadata["git_commit"] and len(metadata["git_commit"]) == 40

    written_trace = pd.read_csv(output_directory / "trace.csv")
    assert list(written_trace.columns) == list(TRACE_COLUMNS)
    assert len(written_trace) == EXPECTED_STEPS

    written_kpis = pd.read_csv(output_directory / "district_kpis.csv")
    assert {"cost_function", "value", "name", "level"} <= set(written_kpis.columns)
    assert "cost_total" in set(written_kpis["cost_function"])

    readme = (output_directory / "README.md").read_text()
    assert ZeroActionController.name in readme
    assert f"Seed: {SEED}" in readme
    assert "Purpose:" in readme


def test_identical_seed_reproducible_kpis() -> None:
    first_kpis, _ = run_episode(
        ZeroActionController(),
        SCHEMA_PATH,
        simulation_start_time_step=SIMULATION_START_TIME_STEP,
        simulation_end_time_step=SIMULATION_END_TIME_STEP,
        seed=SEED,
    )
    second_kpis, _ = run_episode(
        ZeroActionController(),
        SCHEMA_PATH,
        simulation_start_time_step=SIMULATION_START_TIME_STEP,
        simulation_end_time_step=SIMULATION_END_TIME_STEP,
        seed=SEED,
    )

    assert set(first_kpis) == set(second_kpis)
    assert first_kpis == second_kpis


def test_trace_has_no_nan(b0_dev_run) -> None:
    _, trace = b0_dev_run

    assert list(trace.columns) == list(TRACE_COLUMNS)
    assert len(trace) == EXPECTED_STEPS
    assert not trace.isna().any().any(), "trace must not contain NaN values"


def test_b0_matches_smoke_kpis(b0_dev_run, smoke_anchors) -> None:
    kpis, _ = b0_dev_run

    assert set(smoke_anchors) <= set(kpis), "B0 must produce every non-empty smoke KPI"
    worst_gap = 0.0
    for cost_function, expected in smoke_anchors.items():
        gap = abs(kpis[cost_function] - expected)
        worst_gap = max(worst_gap, gap)
        assert gap <= TOLERANCE, (
            f"{cost_function}: expected {expected!r}, got {kpis[cost_function]!r} "
            f"(|delta|={gap} > {TOLERANCE})"
        )
    assert abs(kpis["cost_total"] - smoke_anchors["cost_total"]) <= TOLERANCE

    for excluded in EXCLUDED_EMPTY_KPIS:
        assert excluded not in kpis, f"{excluded} is empty in this dataset and must be excluded"


def test_runner_rejects_invalid_window() -> None:
    with pytest.raises(ValueError):
        run_episode(
            ZeroActionController(),
            SCHEMA_PATH,
            simulation_start_time_step=10,
            simulation_end_time_step=10,
            seed=SEED,
        )


def test_build_run_metadata_records_relative_schema_path_and_commit() -> None:
    metadata = build_run_metadata(
        "any-controller",
        SCHEMA_PATH,
        simulation_start_time_step=SIMULATION_START_TIME_STEP,
        simulation_end_time_step=SIMULATION_END_TIME_STEP,
        seed=SEED,
        completed_steps=7,
        config={"tau_usd_per_kwh": 0.0439},
    )

    assert metadata["schema"] == "configs/schema-building1.json"
    assert metadata["controller"] == "any-controller"
    assert metadata["completed_steps"] == 7
    assert metadata["git_commit"] and len(metadata["git_commit"]) == 40
    assert metadata["config"] == {"tau_usd_per_kwh": 0.0439}
