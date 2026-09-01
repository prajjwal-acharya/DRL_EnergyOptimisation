"""Verify the reproducible Week 1 foundation without training a controller.

Run this after the bootstrap, derived-schema, and smoke-run commands described
in the README.  It verifies project-owned evidence only; CityLearn's named
dataset API is intentionally not contacted because it depends on GitHub's
anonymous API availability.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from citylearn import __version__ as citylearn_version

from energy_optimisation.environment import load_environment, neutral_actions


REQUIRED_PATHS = (
    ".gitignore",
    "README.md",
    "requirements.txt",
    "pytest.ini",
    "configs/smoke.yaml",
    "configs/schema-building1.json",
    "docs/reference/environment-selection.md",
    "docs/reference/experiment-protocol.md",
    "docs/reference/literature-matrix.csv",
    "docs/reference/literature.md",
    "scripts/01_bootstrap_data.py",
    "scripts/02_make_schema.py",
    "scripts/03_inspect_environment.py",
    "scripts/04_smoke_run.py",
    "src/energy_optimisation/environment.py",
    "tests/test_environment.py",
)


def require(condition: bool, message: str) -> None:
    """Fail with a helpful explanation when a Week 1 invariant is broken."""

    if not condition:
        raise AssertionError(message)


def verify_required_paths() -> str:
    missing = [path for path in REQUIRED_PATHS if not (PROJECT_ROOT / path).is_file()]
    require(not missing, f"Missing required Week 1 path(s): {', '.join(missing)}")
    return f"{len(REQUIRED_PATHS)} required repository files present"


def verify_documentation() -> str:
    readme = (PROJECT_ROOT / "README.md").read_text()
    for heading in ("## Research questions", "**RQ1**", "**RQ2**", "**RQ3**", "## Research boundary"):
        require(heading in readme, f"README is missing required content: {heading}")

    cmdp = (PROJECT_ROOT / "docs/reference/environment-selection.md").read_text()
    require("### Evidence-based environment decision" in cmdp, "CMDP environment decision is missing")

    gitignore = (PROJECT_ROOT / ".gitignore").read_text()
    for ignored_path in (".venv/", "data/raw/", "data/processed/", "results/"):
        require(ignored_path in gitignore, f".gitignore is missing {ignored_path}")
    return "README scope/RQ1-RQ3 and CMDP decision recorded; generated data ignored"


def verify_literature_matrix() -> str:
    matrix_path = PROJECT_ROOT / "docs/reference/literature-matrix.csv"
    with matrix_path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    require(len(rows) >= 12, f"Expected at least 12 literature rows, found {len(rows)}")
    analysed = [row for row in rows if row["status"].strip().lower() == "analysed"]
    require(len(analysed) >= 6, f"Expected at least 6 analysed rows, found {len(analysed)}")
    require(
        all(row["relevance_to_rq"].strip() for row in analysed),
        "Every analysed source must map to at least one research question",
    )
    return f"literature matrix has {len(rows)} sources and {len(analysed)} analysed rows"


def verify_schema_and_interface() -> str:
    schema_path = PROJECT_ROOT / "configs/schema-building1.json"
    schema = json.loads(schema_path.read_text())
    require(list(schema["buildings"]) == ["Building_1"], "Derived schema must contain only Building_1")

    environment = load_environment(
        schema_path,
        central_agent=True,
        simulation_start_time_step=0,
        simulation_end_time_step=167,
    )
    observations, _ = environment.reset(seed=42)
    actions = neutral_actions(environment)
    require(len(observations) == 1, "Expected a single central-agent observation")
    require(len(actions) == 1 and len(actions[0]) == 3, "Expected three valid Building_1 actions")
    return "pinned one-building schema resets with one observation agent and three valid actions"


def verify_smoke_evidence() -> str:
    output_directory = PROJECT_ROOT / "results/runs/smoke"
    expected_outputs = ("run_metadata.json", "district_kpis.csv", "district_kpis.png", "README.md")
    missing = [name for name in expected_outputs if not (output_directory / name).is_file()]
    require(not missing, f"Missing smoke output(s): {', '.join(missing)}")

    metadata = json.loads((output_directory / "run_metadata.json").read_text())
    require(metadata["terminated"] is True, "Smoke run did not reach the terminal state")
    require(metadata["truncated"] is False, "Smoke run was unexpectedly truncated")
    require(metadata["completed_steps"] == 167, "Smoke run must complete the configured 167 actions")
    require(metadata["purpose"] == "infrastructure smoke test only; not a research result", "Smoke purpose changed")
    return "terminal smoke run wrote metadata, KPI CSV, KPI plot, and run note"


def main() -> None:
    checks = (
        ("Repository", verify_required_paths),
        ("Documentation", verify_documentation),
        ("Literature", verify_literature_matrix),
        ("Environment", verify_schema_and_interface),
        ("Smoke evidence", verify_smoke_evidence),
    )
    require(citylearn_version == "2.5.0", f"Expected CityLearn 2.5.0, found {citylearn_version}")
    print(f"PASS  CityLearn version: {citylearn_version}")
    for name, check in checks:
        print(f"PASS  {name}: {check()}")
    print("Week 1 verification passed. No controller training was performed.")


if __name__ == "__main__":
    main()
