"""Run a deterministic, short CityLearn smoke simulation and save evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energy_optimisation.environment import load_environment, neutral_actions


DEFAULT_SCHEMA = PROJECT_ROOT / "configs/schema-building1.json"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "results/runs/smoke"


def write_kpi_plot(district_kpis: pd.DataFrame, output_path: Path) -> None:
    """Write a compact KPI bar chart for the smoke-run handoff."""

    selected_names = ["cost_total", "all_time_peak_average", "carbon_emissions_total"]
    selected = district_kpis.set_index("cost_function").reindex(selected_names).dropna(subset=["value"])

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(selected.index, selected["value"], color="#2b6cb0")
    axis.set_title("CityLearn smoke-run district KPIs")
    axis.set_ylabel("Normalised value")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-time-step", type=int, default=0)
    parser.add_argument("--end-time-step", type=int, default=167)
    arguments = parser.parse_args()

    if arguments.end_time_step <= arguments.start_time_step:
        raise ValueError("end-time-step must be greater than start-time-step")

    environment = load_environment(
        arguments.schema,
        central_agent=True,
        simulation_start_time_step=arguments.start_time_step,
        simulation_end_time_step=arguments.end_time_step,
    )
    observations, _ = environment.reset(seed=arguments.seed)
    total_reward = 0.0
    completed_steps = 0

    while not environment.terminated:
        observations, rewards, terminated, truncated, _ = environment.step(neutral_actions(environment))
        total_reward += float(sum(rewards))
        completed_steps += 1
        if terminated or truncated:
            break

    if not environment.terminated:
        raise RuntimeError("Smoke run finished without the expected terminal state")

    output_directory = arguments.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    kpis = environment.evaluate()
    district_kpis = kpis.loc[kpis["level"] == "district"].copy()
    district_kpis.to_csv(output_directory / "district_kpis.csv", index=False)
    write_kpi_plot(district_kpis, output_directory / "district_kpis.png")

    metadata = {
        "schema": str(arguments.schema),
        "controller": "deterministic_zero_action",
        "seed": arguments.seed,
        "simulation_start_time_step": arguments.start_time_step,
        "simulation_end_time_step": arguments.end_time_step,
        "completed_steps": completed_steps,
        "terminated": environment.terminated,
        "truncated": environment.truncated,
        "observation_agent_count": len(observations),
        "action_dimensions": len(neutral_actions(environment)[0]),
        "total_reward": total_reward,
        "purpose": "infrastructure smoke test only; not a research result",
    }
    (output_directory / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_directory / "README.md").write_text(
        "# CityLearn Smoke Run\n\n"
        f"- Schema: `{arguments.schema}`\n"
        "- Controller: deterministic zero action\n"
        f"- Seed: {arguments.seed}\n"
        f"- Simulation time range: {arguments.start_time_step}-{arguments.end_time_step}\n"
        f"- Completed steps: {completed_steps}\n"
        "- Purpose: installation and environment smoke test only\n"
        "- Not a research result: yes\n"
    )
    print(f"Completed {completed_steps} simulator steps; evidence written to {output_directory}")


if __name__ == "__main__":
    main()
