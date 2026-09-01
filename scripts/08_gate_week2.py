"""Verify the completed Week 2 phase: CMDP spec, harness, baselines, artifacts.

Plan reference: docs/plans/week2-implementation-plan.md §D3 (mirrors the structure of
``scripts/05_gate_week1.py``). Every check below is a hard pass/fail with a
clear message; the script exits non-zero on the first failure. No controller is
trained and nothing under ``results/`` is modified — this reads recorded phase
evidence only.

Checks:
1. the full pytest suite passes,
2. B0/B1/B2 have complete artifact sets for BOTH windows (dev 0-167, final 0-719),
3. results/tables/baseline_comparison.csv exists with exactly 3 controllers,
4. at least 4 comparison figures exist under results/figures/,
5. B0 dev-window KPIs reproduce the §0 smoke anchors (tolerance 1e-9),
6. traces contain no NaN and metrics carry clipping/reserve event counts,
7. docs/reference/cmdp-spec.md has all 6 sections and no unfilled placeholders,
8. configs/week2-baselines.yaml exists with the frozen constants and baselines/
   sources contain no forbidden imports,
9. docs/status/phase-reviews/week2-review.md exists.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CONTROLLERS = ("b0_neutral", "b1_fixed_schedule", "b2_tariff_aware")
WINDOWS = ("dev", "final")
BASELINES_ROOT = PROJECT_ROOT / "results/runs/baselines"
COMPARISON_TABLE = PROJECT_ROOT / "results/tables/baseline_comparison.csv"
FIGURES_DIR = PROJECT_ROOT / "results/figures"

RUN_METADATA_FILE = "run_metadata.json"
TRACE_FILE = "trace.csv"
DISTRICT_KPIS_FILE = "district_kpis.csv"
DERIVED_METRICS_FILE = "derived_metrics.json"
README_FILE = "README.md"
REQUIRED_ARTIFACT_FILES = (
    RUN_METADATA_FILE,
    TRACE_FILE,
    DISTRICT_KPIS_FILE,
    DERIVED_METRICS_FILE,
    README_FILE,
)

# §0 anchors for B0 (zero actions, window 0-167, seed 42), verbatim from
# docs/plans/week2-implementation-plan.md (sourced from results/runs/smoke).
B0_DEV_ANCHORS = {
    "cost_total": 0.44198876839332574,
    "all_time_peak_average": 0.8618364154405324,
    "electricity_consumption_total": 0.464085898307736,
    "discomfort_proportion": 0.9151515151515152,
    "ramping_average": 0.8571830450575444,
    "zero_net_energy": 0.35004620158879785,
}
ANCHOR_TOLERANCE = 1e-9

FIGURE_KINDS = (
    "cost_by_controller",
    "net_demand_overlay_48h",
    "electrical_soc_trace",
    "indoor_temperature_vs_setpoint",
)

CMDP_SPEC_SECTIONS = (
    "## 1. State",
    "## 2. Actions",
    "## 3. Transition",
    "## 4. Reward",
    "## 5. Constraints",
    "## 6. KPI mapping",
)
UNFILLED_PLACEHOLDER_TOKENS = (
    "pending first b0 run",
    "todo",
    "tbd",
    "placeholder",
    "fill in",
)

FORBIDDEN_BASELINE_TOKENS = ("forecasting", "stable_baselines3", "torch")

REQUIRED_CONFIG_KEYS = (
    "schema_path",
    "random_seed",
    "windows",
    "tariff_threshold_usd_per_kwh",
    "reserve_low_soc",
    "reserve_high_soc",
    "comfort_band_c",
    "reward_weights",
    "b1_fixed_schedule",
    "b2_tariff_aware",
)


def require(condition: bool, message: str) -> None:
    """Fail loudly when a Week 2 invariant is broken."""

    if not condition:
        raise AssertionError(message)


def verify_pytest_suite() -> str:
    command = [str(PROJECT_ROOT / ".venv/bin/python"), "-m", "pytest", "-q"]
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    require(
        completed.returncode == 0,
        f"pytest suite failed (exit {completed.returncode}):\n{completed.stdout[-2000:]}",
    )
    summary = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    return f"pytest suite green ({summary})"


def run_directory(controller: str, window: str) -> Path:
    return BASELINES_ROOT / controller / window


def read_run_kpis(run_directory_path: Path) -> Dict[str, float]:
    kpis_path = run_directory_path / DISTRICT_KPIS_FILE
    require(kpis_path.is_file(), f"Missing district KPIs: {kpis_path}")
    with kpis_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    return {
        row["cost_function"]: float(row["value"])
        for row in rows
        if str(row["value"]).strip() != ""
    }


def verify_baseline_artifacts() -> str:
    checked = 0
    for controller in CONTROLLERS:
        for window in WINDOWS:
            directory = run_directory(controller, window)
            missing = [name for name in REQUIRED_ARTIFACT_FILES if not (directory / name).is_file()]
            require(
                not missing,
                f"Incomplete artifact set for {controller}/{window}: missing {', '.join(missing)}",
            )
            metadata = json.loads((directory / RUN_METADATA_FILE).read_text())
            require(metadata["controller"] == controller, f"Metadata controller mismatch in {directory}")
            require(metadata["seed"] == 42, f"{controller}/{window} must use seed 42")
            bounds = metadata["simulation_start_time_step"], metadata["simulation_end_time_step"]
            expected_bounds = {"dev": (0, 167), "final": (0, 719)}[window]
            require(
                bounds == expected_bounds,
                f"{controller}/{window} window {bounds} != frozen {expected_bounds}",
            )
            trace = (directory / TRACE_FILE).read_text()
            header = trace.splitlines()[0].split(",")
            require("net_electricity_consumption" in header, f"{controller}/{window} trace lacks KPI series")
            checked += 1
    return f"complete artifact sets for {checked} runs ({len(CONTROLLERS)} controllers x {len(WINDOWS)} windows)"


def verify_comparison_table() -> str:
    require(COMPARISON_TABLE.is_file(), f"Missing comparison table: {COMPARISON_TABLE}")
    with COMPARISON_TABLE.open(newline="") as file:
        rows = list(csv.DictReader(file))
    controllers_in_table = {row["controller"] for row in rows}
    require(
        controllers_in_table == set(CONTROLLERS),
        f"Comparison table must contain exactly the 3 baseline controllers, found {sorted(controllers_in_table)}",
    )
    for controller in CONTROLLERS:
        for window in WINDOWS:
            matching = [row for row in rows if row["controller"] == controller and row["window"] == window]
            require(len(matching) == 1, f"Comparison table needs exactly one {controller} row for {window}")
            require(matching[0]["cost_total"] != "", f"{controller}/{window} cost_total missing")
    return f"comparison table covers {len(CONTROLLERS)} controllers across {len(WINDOWS)} windows ({len(rows)} rows)"


def verify_figures() -> str:
    require(FIGURES_DIR.is_dir(), f"Missing figures directory: {FIGURES_DIR}")
    figures = sorted(FIGURES_DIR.glob("*.png"))
    require(len(figures) >= 4, f"Expected at least 4 comparison figures, found {len(figures)}")
    figure_names = " ".join(path.name for path in figures)
    for kind in FIGURE_KINDS:
        require(kind in figure_names, f"No comparison figure of kind '{kind}' under {FIGURES_DIR.name}")
    return f"{len(figures)} comparison figures present incl. all 4 required kinds"


def verify_b0_dev_regression() -> str:
    kpis = read_run_kpis(run_directory("b0_neutral", "dev"))
    worst_gap = 0.0
    for cost_function, expected in B0_DEV_ANCHORS.items():
        require(cost_function in kpis, f"B0 dev run lacks anchor KPI '{cost_function}'")
        gap = abs(kpis[cost_function] - expected)
        worst_gap = max(worst_gap, gap)
        require(
            gap <= ANCHOR_TOLERANCE,
            f"B0 dev {cost_function}: expected {expected!r}, got {kpis[cost_function]!r} "
            f"(|delta|={gap} > {ANCHOR_TOLERANCE})",
        )
    return f"B0 dev-window KPIs match all {len(B0_DEV_ANCHORS)} §0 anchors (max |delta|={worst_gap:g})"


def verify_traces_and_metrics() -> str:
    import pandas as pd

    checked = 0
    for controller in CONTROLLERS:
        for window in WINDOWS:
            directory = run_directory(controller, window)
            trace = pd.read_csv(directory / TRACE_FILE)
            require(not trace.isna().any().any(), f"NaN values in {controller}/{window} trace")
            # CityLearn 2.5.0 runs exactly (end - start) hourly steps per
            # episode (terminated when time_step == end - start - 1); the
            # same convention that produced the §0 smoke anchors.
            expected_rows = {"dev": 167, "final": 719}[window]
            require(
                len(trace) == expected_rows,
                f"{controller}/{window} trace has {len(trace)} rows, expected {expected_rows}",
            )
            metrics = json.loads((directory / DERIVED_METRICS_FILE).read_text())
            for key in ("clipping_event_count", "reserve_event_count"):
                require(key in metrics, f"{controller}/{window} derived metrics lack '{key}'")
                require(
                    isinstance(metrics[key], int),
                    f"{controller}/{window} metric '{key}' must be an integer count",
                )
            checked += 1
    return f"{checked} traces NaN-free with clipping/reserve event counts present"


def verify_cmdp_spec() -> str:
    path = PROJECT_ROOT / "docs/reference/cmdp-spec.md"
    require(path.is_file(), f"Missing CMDP specification: {path}")
    text = path.read_text()
    for heading in CMDP_SPEC_SECTIONS:
        require(heading in text, f"cmdp_spec.md is missing section '{heading}'")
    lowered = text.lower()
    for token in UNFILLED_PLACEHOLDER_TOKENS:
        require(token not in lowered, f"cmdp_spec.md contains unfilled placeholder token '{token}'")
    for constant in ("Ē_B0", "P_ref", "P_grid,max"):
        require(constant in text, f"cmdp_spec.md does not record {constant}")
    return "all 6 sections present, constants recorded, no unfilled placeholders"


def verify_config_and_imports() -> str:
    config_path = PROJECT_ROOT / "configs/week2-baselines.yaml"
    require(config_path.is_file(), f"Missing frozen config: {config_path}")
    config: Mapping = yaml.safe_load(config_path.read_text())
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    require(not missing, f"week2.yaml is missing frozen key(s): {', '.join(missing)}")
    require(config["random_seed"] == 42, "week2.yaml seed must be the frozen 42")
    require(float(config["tariff_threshold_usd_per_kwh"]) == 0.0439, "week2.yaml tau must stay frozen at 0.0439")

    baselines_directory = PROJECT_ROOT / "src/energy_optimisation/baselines"
    sources = sorted(baselines_directory.glob("*.py"))
    require(bool(sources), "baselines package sources not found")
    for source in sources:
        text = source.read_text().lower()
        for token in FORBIDDEN_BASELINE_TOKENS:
            require(token not in text, f"{source.name} mentions forbidden dependency '{token}'")
    return "week2.yaml locked with frozen constants; baselines/ free of forbidden imports"


def verify_review_document() -> str:
    path = PROJECT_ROOT / "docs/status/phase-reviews/week2-review.md"
    require(path.is_file(), f"Missing review document: {path}")
    text = path.read_text()
    require("supervisor update" in text.lower(), "week2-review.md must include the supervisor update verbatim")
    return "week2-review.md present with supervisor update"


CHECKS = (
    ("Pytest", verify_pytest_suite),
    ("Baseline artifacts", verify_baseline_artifacts),
    ("Comparison table", verify_comparison_table),
    ("Figures", verify_figures),
    ("B0 regression", verify_b0_dev_regression),
    ("Traces & metrics", verify_traces_and_metrics),
    ("CMDP spec", verify_cmdp_spec),
    ("Config & imports", verify_config_and_imports),
    ("Review doc", verify_review_document),
)


def main() -> None:
    passed: List[str] = []
    failed: List[str] = []
    for name, check in CHECKS:
        try:
            detail = check()
        except AssertionError as error:
            failed.append(name)
            print(f"FAIL  {name}: {error}")
            continue
        passed.append(name)
        print(f"PASS  {name}: {detail}")

    require(not failed, f"Week 2 verification failed check(s): {', '.join(failed)}")
    print("Week 2 verification passed: CMDP spec, harness, baselines, and artifacts are locked.")


if __name__ == "__main__":
    main()
