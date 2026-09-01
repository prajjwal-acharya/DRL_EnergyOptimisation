# CMDP Formulation

## Environment inspection - completed on Day 2

- CityLearn version: `2.5.0`.
- Local source data: CityLearn tag `v2.5.0`, commit `29062af6d077409e1c37a3e53a6cac30fd4d02bc`.
- Selected parent dataset: `citylearn_challenge_2023_phase_1`.
- Parent scenario: 3 buildings and 720 hourly timesteps.
- Controller interface: one central agent; a 49-feature combined observation and a 9-dimensional action vector.
- Building 1 observations include outdoor weather and three-step forecasts, solar irradiance and forecasts, indoor temperature, non-shiftable load, solar generation, battery state of charge, net electricity consumption, electricity price and three-step price forecasts, cooling demand, DHW demand, occupancy, and cooling set point.
- Building 1 actions: `dhw_storage`, `electrical_storage`, and `cooling_device`.
- Building 1 assets: 2.4 kW nominal PV and a 4.0 kWh battery.
- District KPIs include cost, peak, carbon, load factor, ramping, electricity use, thermal discomfort, resilience, and outage metrics.

### Evidence-based environment decision

`citylearn_challenge_2020_climate_zone_1` was rejected as the CP-I parent scenario: it has 9 buildings, its electricity-pricing observation is inactive, and it exposes cooling/DHW/electrical storage rather than cooling-device control.

The selected 2023 Phase 1 parent scenario contains the tariff, solar, battery, indoor-temperature, cooling-device, and price-forecast signals required by the research plan. It is still a 3-building scenario, so the next environment task is to derive a version that retains only `Building_1`. That will satisfy the CP-I single-building scope without editing the raw source data.

## Decision problem - superseded by the formal specification

The formal CMDP (state, action, reward, transition, and constraints, with frozen constants) now lives in [`cmdp_spec.md`](cmdp_spec.md), which is the mathematical source of truth for the project. This file is retained as the Week 1 environment-selection record.
