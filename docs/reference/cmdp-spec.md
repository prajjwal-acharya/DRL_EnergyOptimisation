# Week 2 CMDP Specification — CityLearn 2023 Phase 1, Building 1 (central agent)

Formal constrained-MDP specification for the Week 2 deterministic-baseline phase.
Binding sources: `docs/plans/week2-implementation-plan.md` (§0 ground truth), `configs/schema-building1.json`.
Observations are addressed **by name** through `src/energy_optimisation/observation_names.py`; no magic indices anywhere.

Recorded constants (provisional Phase-A measurement, see §§4–5): `Ē_B0 = 0.477229108554339`, `P_ref = 7.694016456604004`, `P_grid,max = 1.8084184527397151`.

---

## 1. State

The state vector is the central-agent observation. Per the week-1 inspection of the 3-building parent schema (`results/inspection/citylearn_2023_phase_1.json`) it is a 49-dim combined vector: 19 shared observations counted once plus 10 unshared observations per building (3 buildings). The Week 2 evaluation environment uses the derived single-building schema, whose central-agent observation contains exactly Building_1's 29 active observations (19 shared counted once) in the identical order; controllers resolve every entry by name via `BUILDING_1_OBSERVATION_INDEX`.

All 29 Building_1 observations, in central-agent order:

| # | Symbol | Name | Unit | Group | Source | Measurement |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `d_type` | `day_type` | categorical {1–8} | calendar | dataset calendar (`energy_simulation.csv`) | central-agent observation, name-addressed |
| 1 | `h` | `hour` | h ∈ {0–23} | calendar | dataset calendar (`energy_simulation.csv`) | central-agent observation, name-addressed |
| 2 | `T_out` | `outdoor_dry_bulb_temperature` | °C | weather-solar | `weather.csv` | central-agent observation, name-addressed |
| 3 | `T_out,p1` | `outdoor_dry_bulb_temperature_predicted_1` | °C | weather-solar | `weather.csv` | central-agent observation, name-addressed |
| 4 | `T_out,p2` | `outdoor_dry_bulb_temperature_predicted_2` | °C | weather-solar | `weather.csv` | central-agent observation, name-addressed |
| 5 | `T_out,p3` | `outdoor_dry_bulb_temperature_predicted_3` | °C | weather-solar | `weather.csv` | central-agent observation, name-addressed |
| 6 | `I_diff` | `diffuse_solar_irradiance` | W/m² | weather-solar | `weather.csv` | central-agent observation, name-addressed |
| 7 | `I_diff,p1` | `diffuse_solar_irradiance_predicted_1` | W/m² | weather-solar | `weather.csv` | central-agent observation, name-addressed |
| 8 | `I_diff,p2` | `diffuse_solar_irradiance_predicted_2` | W/m² | weather-solar | `weather.csv` | central-agent observation, name-addressed |
| 9 | `I_diff,p3` | `diffuse_solar_irradiance_predicted_3` | W/m² | weather-solar | `weather.csv` | central-agent observation, name-addressed |
| 10 | `I_dir` | `direct_solar_irradiance` | W/m² | weather-solar | `weather.csv` | central-agent observation, name-addressed |
| 11 | `I_dir,p1` | `direct_solar_irradiance_predicted_1` | W/m² | weather-solar | `weather.csv` | central-agent observation, name-addressed |
| 12 | `I_dir,p2` | `direct_solar_irradiance_predicted_2` | W/m² | weather-solar | `weather.csv` | central-agent observation, name-addressed |
| 13 | `I_dir,p3` | `direct_solar_irradiance_predicted_3` | W/m² | weather-solar | `weather.csv` | central-agent observation, name-addressed |
| 14 | `c_t` | `carbon_intensity` | kg_CO₂e/kWh | tariff | `carbon_intensity.csv` | central-agent observation, name-addressed |
| 15 | `T_in` | `indoor_dry_bulb_temperature` | °C | building | `energy_simulation.csv` (dynamics-driven) | central-agent observation, name-addressed |
| 16 | `L_nsl` | `non_shiftable_load` | kWh | building | `energy_simulation.csv` | central-agent observation, name-addressed |
| 17 | `E_pv` | `solar_generation` | kWh | building | PV model (2.4 kW nominal) over `weather.csv` | central-agent observation, name-addressed |
| 18 | `S_dhw` | `dhw_storage_soc` | ratio [0, 1] | storage | DHW tank model (capacity 2.2827 kWh, loss coefficient 0.0032) | central-agent observation, name-addressed |
| 19 | `S_bat` | `electrical_storage_soc` | ratio [0, 1] | storage | battery model (4.0 kWh capacity, 3.32 kW, η = 0.95, DoD = 0.8) | central-agent observation, name-addressed |
| 20 | `E_obs` | `net_electricity_consumption` | kWh | building | device-model aggregation (see lag note below) | central-agent observation, name-addressed; **reads 0.0 in the post-step vector under CityLearn 2.5.0** |
| 21 | `p_t` | `electricity_pricing` | USD/kWh | tariff | `pricing.csv` | central-agent observation, name-addressed |
| 22 | `p_t,p1` | `electricity_pricing_predicted_1` | USD/kWh | tariff | `pricing.csv` | central-agent observation, name-addressed |
| 23 | `p_t,p2` | `electricity_pricing_predicted_2` | USD/kWh | tariff | `pricing.csv` | central-agent observation, name-addressed |
| 24 | `p_t,p3` | `electricity_pricing_predicted_3` | USD/kWh | tariff | `pricing.csv` | central-agent observation, name-addressed |
| 25 | `Q_cool` | `cooling_demand` | kWh | building | `energy_simulation.csv` + cooling-device model | central-agent observation, name-addressed |
| 26 | `Q_dhw` | `dhw_demand` | kWh | building | `energy_simulation.csv` + DHW-device model | central-agent observation, name-addressed |
| 27 | `n_occ` | `occupant_count` | persons | building | `energy_simulation.csv` | central-agent observation, name-addressed |
| 28 | `T_set` | `indoor_dry_bulb_temperature_cooling_set_point` | °C | building | `energy_simulation.csv` | central-agent observation, name-addressed |

**Measurement note (verified empirically, CityLearn 2.5.0).** The `net_electricity_consumption`
entry of the returned post-step observation vector is 0.0 throughout an episode because
CityLearn writes the underlying array after composing the observation. `E_t` for reward,
constraint, and trace purposes is therefore read from `Building_1.net_electricity_consumption[t]`,
the same series consumed by `env.evaluate()`; on the dev window the executed hours t = 0…166
carry valid values (167 hourly entries). Any harness/controller code must apply this workaround;
it is a measurement convention, not a modification of the environment.

Tariff structure (data-derived, from `pricing.csv`): off-peak 0.02893 USD/kWh (hours 0–7, 23; 405 steps),
mid 0.02915 USD/kWh (hours 8–15, 19–22; 252 steps), peak 0.05867 USD/kWh (hours 16–18; 63 steps);
frozen mid→peak threshold **τ = 0.0439 USD/kWh**.

## 2. Actions

Three-dimensional continuous action a_t = (a_DHW, a_bat, a_cool), fixed order, one action per 1-hour step:

| Symbol | Name | Bounds | Sign convention / physical meaning | Unit |
| --- | --- | --- | --- | --- |
| `a_DHW` | `dhw_storage` | [−1, 1] | negative = charge DHW tank, positive = discharge DHW tank | dimensionless (normalised by device rating) |
| `a_bat` | `electrical_storage` | [−1, 1] | negative = charge battery, positive = discharge battery | dimensionless (normalised by 3.32 kW nominal power) |
| `a_cool` | `cooling_device` | [0, 1] | ratio of cooling-device nominal electrical input — **not** a temperature setpoint | dimensionless ratio |

Requested actions outside the Box bounds are clipped by CityLearn; clipping events are counted (§5).

## 3. Transition

P(s_{t+1} | s_t, a_t) is given by CityLearn's dynamics: the trained building dynamics model
(`Building_1.pth`) advances the building state in fixed 1-hour steps; storage SoCs follow the
device models (battery: 4.0 kWh, 3.32 kW, η = 0.95, DoD = 0.8; DHW tank: 2.2827 kWh, loss
coefficient 0.0032). No claim is made that this transition kernel matches a real building; it is
the simulator's model, used as-is. Episode: 720 hourly steps (30 days) per the pinned dataset;
dev window 0–167, final window 0–719, seed 42 (all frozen).

## 4. Reward

Declared now, frozen before training, consumed **only** by PPO in September (baselines never use it):

```
r_t = − w_E · (E_t / Ē_B0) − w_P · max(0, E_t − P_ref) / P_ref − w_C · D_t
```

| Term | Definition | Unit | Measurement method |
| --- | --- | --- | --- |
| `E_t` | `net_electricity_consumption` at step t | kWh | `Building_1.net_electricity_consumption[t]` (§1 lag note) |
| `Ē_B0` | mean `E_t` of the B0 (zero-action) run on the dev window 0–167, seed 42 | kWh | recorded value below |
| `P_ref` | max `E_t` of the same B0 dev-window run (at simulated hour 140) | kWh | recorded value below |
| `D_t` | max(0, `T_in` − (`T_set` + ΔT)) | °C | per-step trace of observations 15 and 28 |
| ΔT | comfort-band half-width above cooling setpoint | °C | constant below |

| Constant | Value | Status |
| --- | --- | --- |
| `Ē_B0` | **0.477229108554339 kWh** (mean of 167 hourly values) | recorded — provisional Phase-A B0 measurement, reproduced all six §0 smoke anchors exactly (tolerance 1e-9); re-verify via the Phase-B harness regression |
| `P_ref` | **7.694016456604004 kWh** | recorded — same provenance as `Ē_B0` |
| `w_E` | 1.0 | research assumption (declared now; one-line justification: equal footing for energy cost against peak and comfort terms) |
| `w_P` | 1.0 | research assumption (declared now; justification: peak excursions weighted equally with mean energy) |
| `w_C` | 10.0 | research assumption (declared now; justification: comfort violations dominate energy terms to reflect the hard constraint in §5) |
| ΔT | 2.0 °C | research assumption (justification: conventional symmetric comfort allowance around the cooling setpoint) |

B0 provenance: zero actions, single-building schema, `simulation_start_time_step=0`,
`simulation_end_time_step=167`, `reset(seed=42)`; statistics computed in float64 over the 167
valid hourly entries. Rule: any later change to any constant in this section requires a new named
config (e.g. `week2b.yaml`), never a silent edit.

## 5. Constraints

Constraints are separate from the reward penalties; each is monitored and reported per run:

| Constraint | Definition | Threshold / unit | Status | Measurement method |
| --- | --- | --- | --- | --- |
| Comfort (overheating) | `T_in` ≤ `T_set` + 2.0 °C at every step | °C violation magnitude `D_t` ≥ 0 | ΔT = 2.0 °C is a research assumption (conventional comfort allowance) | discomfort KPIs (`discomfort_hot_*`, `discomfort_proportion`) + per-step `T_in`/`T_set` trace |
| SoC reserve band | a *requested* action that would push either SoC outside [0.2, 0.9] is a reserve event | SoC ratio, band [0.2, 0.9] | research assumption (keeps both storages inside a usable reserve margin; CityLearn enforces physical bounds internally regardless) | per-step comparison of pre-clip request × current SoC against the band; event counts in run metrics |
| Action clipping | any requested action outside the Box bounds (§2) before clipping is counted | dimensionless bounds [−1, 1], [−1, 1], [0, 1] | CityLearn-defined (no assumption) | trace comparison of requested (pre-clip) vs applied (post-clip) actions |
| Grid import limit | `E_t` should not exceed `P_grid,max` | kW-equivalent kWh per hour; limit below | data-derived assumption (95th percentile of reference B0 demand; not a physical interconnection limit) | per-step count of `E_t` > `P_grid,max` from the trace |

Recorded value: **P_grid,max = 1.8084184527397151 kWh** (95th percentile, NumPy linear
interpolation, float64, over the same 167-value B0 dev-window series as §4).

## 6. KPI mapping

Primary KPIs (decision-making for RQ comparisons):

| KPI | Role |
| --- | --- |
| `cost_total` | primary — electricity cost objective |
| `all_time_peak_average` | primary — peak-demand objective |
| `electricity_consumption_total` | primary — total-energy objective |
| `discomfort_hot_proportion`, `discomfort_hot_delta_average` | primary — thermal-comfort constraint violation rate/magnitude |
| `discomfort_proportion` | primary — aggregate comfort constraint summary |
| `ramping_average` | primary — grid-stress / flexibility objective |
| `zero_net_energy` | primary — net-energy balance summary |

Contextual KPIs (reported, not decision-driving):
`carbon_emissions_total`, `daily_peak_average`, `daily_one_minus_load_factor_average`,
`monthly_one_minus_load_factor_average`, `annual_normalized_unserved_energy_total`,
`discomfort_cold_*` (heating-dominated dataset), `discomfort_hot_delta_minimum`,
`discomfort_hot_delta_maximum`.

Excluded (expected empty — no outage scenarios in this dataset, per §0 of the plan):
`one_minus_thermal_resilience_proportion`, `power_outage_normalized_unserved_energy_total`.
