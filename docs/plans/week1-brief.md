# Week 1: Project Foundation and CityLearn Setup

## Objective

At week end, you must have a reproducible repository, a CityLearn smoke simulation that writes output, the approved research questions in the README, and an initial literature evidence base. Do not train PPO yet.

## Scope boundary

- Simulation only.
- Initial control assets: HVAC-related control and battery charging/discharging.
- Do not add physical deployment, EV charging, or multi-building coordination to CP-I.
- Do not claim real-world savings from simulator results.

## Day 1 - Create the repository

Create this structure:

```text
building-energy-risk-aware-rl/
|- README.md
|- requirements.txt
|- .gitignore
|- configs/
|  `- baseline.yaml
|- docs/
|  |- cmdp-formulation.md
|  |- literature-matrix.csv
|  `- experiment-protocol.md
|- data/
|  |- raw/                 # never edit downloaded data
|  `- processed/
|- notebooks/
|  `- 01_citylearn_exploration.ipynb
|- src/
|  `- energy_optimisation/
|     |- __init__.py
|     |- environment.py
|     |- baselines/
|     |- forecasting/
|     |- safety/
|     `- evaluation/
|- scripts/
|  `- 04_run_smoke_test.py
|- tests/
|  `- test_environment.py
`- results/
   |- figures/
   |- logs/
   `- tables/
```

Rules:

- `src/` is reusable Python code; `notebooks/` is only for exploration and plots.
- Put every experiment setting - dataset, seed, time range, controller, tariff scenario - in `configs/`.
- `results/` is generated data; do not edit it manually.
- Never hard-code paths or random seeds in experiment code.

Create the environment and install only initial dependencies:

```bash
cd "/path/to/building-energy-risk-aware-rl"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install CityLearn pandas numpy matplotlib jupyter pytest
python -m pip freeze > requirements.txt
```

Create `.gitignore`:

```gitignore
.venv/
__pycache__/
.ipynb_checkpoints/
.DS_Store
data/raw/
data/processed/
results/
*.pyc
```

Commit code, configuration, README, literature matrix, and small result summaries. Do not commit downloaded datasets, model checkpoints, or large outputs.

## Day 2 - Confirm CityLearn works

Verify the package and see the available datasets:

```bash
python -m citylearn --version
python -m citylearn list_datasets
```

Load the official example dataset:

```python
from citylearn.citylearn import CityLearnEnv

env = CityLearnEnv("citylearn_challenge_2020_climate_zone_1")
```

CityLearn downloads and caches a named dataset on its first use. Use `notebooks/01_citylearn_exploration.ipynb` to inspect these items before writing a controller:

1. Number of buildings.
2. Observation-space shape and feature names.
3. Action-space shape and action bounds.
4. Simulation length and timestep.
5. The controllable assets present in the dataset.
6. CityLearn KPI and metric methods.

Record the evidence in `docs/reference/environment-selection.md`:

```md
## Environment inspection

- Dataset:
- Number of buildings:
- Control timestep:
- Available observations:
- Available controllable assets:
- Available actions:
- Default reward/KPIs:
- Decision for CP-I prototype:
```

Do not assume every CityLearn dataset exposes direct thermostat/setpoint control. Verify the action space before starting PPO. If the selected scenario lacks the required HVAC action, record the issue now and select or customise an appropriate schema later.

## Day 3 - Run a smoke simulation

The first run must prove the complete path:

```text
load environment
-> reset it
-> choose valid actions
-> step through a short episode
-> save output
-> produce one KPI summary or plot
```

Run a short CityLearn CLI evaluation:

```bash
python -m citylearn simulate \
  citylearn_challenge_2020_climate_zone_1 \
  evaluate \
  --output_directory results/runs/smoke \
  --simulation_id smoke-run \
  --random_seed 42 \
  --evaluation_episode_time_steps 0 167
```

Create `results/runs/smoke/README.md`:

```md
Dataset: citylearn_challenge_2020_climate_zone_1
Controller: CityLearn default evaluation agent
Seed: 42
Steps: 0-167
Purpose: installation and environment smoke test only
Not a research result: yes
```

This smoke test is complete only when it completes without errors, writes output files, and you can state the action space and KPIs that CityLearn provides.

## Day 4 - Write the README and research questions

Start `README.md` with this project definition:

```md
# Risk-Aware Deep Reinforcement Learning for Building Energy Optimisation

## Problem

Grid-interactive buildings must make sequential HVAC and battery decisions under uncertain electricity demand, solar generation, weather, occupancy, and time-of-day tariffs. This project tests whether uncertainty-aware, safety-constrained reinforcement learning can reduce operating cost and peak grid demand without unacceptable comfort or operational violations.

## Scope

- Simulation-based prototype using CityLearn.
- Initial control assets: HVAC-related control and battery charging/discharging.
- Main RL algorithm: PPO.
- Forecast outputs: point forecasts and prediction intervals.
- Safety constraints: comfort, battery state of charge, charge/discharge bounds, and grid-import limits.

## Out of scope for CP-I

- Physical building deployment
- EV charging control
- Multi-building coordination
- Claims of real-world savings without simulator evidence

## Research questions

### RQ1
Can forecast uncertainty improve the cost-comfort trade-off compared with a controller that uses only point forecasts?

### RQ2
Can a safety shield reduce comfort, battery, and grid-limit violations during forecast error and high-demand events while preserving most cost savings?

### RQ3
How robust is the proposed controller under changed tariffs, solar availability, occupancy patterns, and weather conditions?

## Evaluation metrics

- Electricity cost
- Peak grid demand
- Comfort-violation duration
- Battery and grid constraint violations
- Solar self-consumption
- Worst-case performance under forecast error

## Reproducibility

Every experiment records its dataset/schema, configuration, seed, code version, controller variant, and output location.
```

Keep these research questions fixed unless the supervisor approves a change. Every later experiment must answer at least one of them.

## Days 5-6 - Start the literature matrix

Create `docs/reference/literature-matrix.csv` with this header:

```csv
citation_key,year,category,building_or_dataset,control_assets,forecast_method,uncertainty_method,rl_or_control_method,safety_or_constraints,baselines,metrics,main_result,limitation,relevance_to_rq,link,status,notes
```

Find sources across four categories:

| Category | Minimum papers | Extract |
| --- | ---: | --- |
| Building energy RL | 3 | Environment, controllable assets, reward, RL method, baselines |
| Probabilistic load/PV forecasting | 3 | Forecast horizon, interval/quantile method, MAE/RMSE, calibration |
| Safe/constrained RL | 3 | Constraint type, shield/fallback/CMDP method, violation reporting |
| Tariff-aware demand response | 2 | Tariff design, cost/peak metrics, operating assumptions |

Suggested searches:

```text
"building energy management" reinforcement learning CityLearn PPO
probabilistic load forecasting quantile regression prediction interval calibration
safe reinforcement learning action shielding constrained Markov decision process
building battery HVAC time-of-day tariff demand response
```

For every paper, immediately answer:

1. What decision is controlled?
2. What data or simulator is used?
3. Does it use point forecasts or uncertainty information?
4. How are safety constraints enforced: penalty, clipping, shield, or constrained optimisation?
5. What baselines and metrics are used?
6. What limitation creates room for this project?

Week 1 literature target:

- 12 papers discovered.
- 6 rows fully analysed.
- 3-5 concrete gaps recorded.
- Every analysed paper mapped to RQ1, RQ2, or RQ3.

## Day 7 - Review and commit

Before the week ends, verify:

```text
[ ] Repository structure exists.
[ ] Virtual environment installs CityLearn successfully.
[ ] A named CityLearn dataset loads.
[ ] A short simulation completes and writes results.
[ ] Observation and action spaces are understood for the selected environment.
[ ] README contains scope, non-goals, RQ1-RQ3, and metrics.
[ ] Literature matrix has at least 12 sources and 6 analysed rows.
[ ] cmdp-formulation.md records the initial environment decision.
[ ] Everything except generated data/results is committed to Git.
```

## Next week

Formulate the CMDP and implement a tariff-aware rule-based baseline. Do not begin PPO or LSTM/GRU development until the environment, metrics, and baseline run are reproducible.

## References

- CityLearn installation: https://www.citylearn.net/installation.html
- CityLearn environment loading: https://www.citylearn.net/usage/load_environment.html
- CityLearn CLI: https://www.citylearn.net/usage/cli.html
