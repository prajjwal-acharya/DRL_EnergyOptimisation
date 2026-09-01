# Data folder — pointer-wise summary

- **Dataset source:** CityLearn Challenge 2023 Phase 1 dataset (included under `data/raw/citylearn_challenge_2023_phase_1`).
- **Purpose:** Time-series simulation data used by the project for building energy modelling, forecasting, baseline algorithms and RL agent training/evaluation.
- **Time resolution:** hourly time steps (see `seconds_per_time_step` in the schema: 3600.0 seconds).
- **Simulation range:** short episode of 720 timesteps in the provided schema (simulation_start_time_step 0 → simulation_end_time_step 719).

- **Raw files (path: data/raw/citylearn_challenge_2023_phase_1):**
	- **Building_1.csv / Building_2.csv / Building_3.csv:** per-building hourly time-series CSVs. Columns include time/context (month, hour, day_type, daylight_savings_status), outdoor/indoor temperatures, humidity, non-shiftable load, DHW/cooling/heating demand, solar generation, occupant count, setpoints and hvac mode. These are the energy-simulation traces consumed by the environment.
	- **Building_1.pth / Building_2.pth / Building_3.pth:** PyTorch binary files referenced by the LSTM dynamics in the schema (LSTMDynamics models / saved tensors used to simulate building dynamics). See `dynamics.filename` entries in the schema.
	- **weather.csv:** hourly weather observations and short-term forecasts (e.g., outdoor temperature, relative humidity, solar irradiance and predicted values). Used by buildings via `weather` entries in the schema.
	- **pricing.csv:** electricity pricing time-series used as `electricity_pricing` (spot / time-varying price signal for reward/cost calculations).
	- **carbon_intensity.csv:** hourly carbon intensity time-series used by the environment (used by reward components or metrics).
	- **schema.json:** full environment configuration and metadata. Contains: active observations, actions, agent config, reward function, building-specific settings (devices, storage, PV, dynamics configuration, normalization ranges, input observation order), simulation timesteps and seeds. This is the authoritative source for how the CSVs are interpreted by the environment.
	- **PROVENANCE.txt:** provenance / license / source notes for the raw dataset.

- **Processed folder (data/processed):** currently empty — intended target for any derived artefacts (normalized CSVs, train/test splits, aggregated summaries, or precomputed feature files). Populate during preprocessing steps if needed.

- **How files are used in this repo:**
	- `schema.json` maps each building CSV and shared CSVs (weather/pricing/carbon_intensity) into the CityLearn environment. The project scripts and `src/` code load these files to construct the RL environment and run baselines/experiments.
	- The `.csv` files are plain text, comma-separated, with one row per hourly timestep. The `.pth` files are binary PyTorch artifacts invoked by the LSTM dynamics.

- **Notes & tips:**
	- Inspect `data/raw/citylearn_challenge_2023_phase_1/schema.json` for exact observation names and which columns are active/used by agents. This clarifies column meanings and input ordering for ML models.
	- When adding processed data, place it under `data/processed/` with a short README or consistent naming convention so downstream scripts can find it.

---
Generated: concise pointer-wise dataset summary for quick onboarding and reproducible use.
