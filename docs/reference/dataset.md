# Dataset Reference — `citylearn_challenge_2023_phase_1`

Everything about the data this project runs on. All facts below were verified against the
raw files on disk and against recorded environment traces.

## Identity and provenance

- **What:** a 30-day, hourly-resolution (720 steps) simulation dataset for a small
  residential building in a hot climate — June, outdoor 21.4–40.3 °C, cooling-dominated
  (heating demand is zero all month, `hvac_mode` = 1 throughout).
- **Where from:** ships inside the pinned CityLearn v2.5.0 source clone at
  `data/raw/citylearn_challenge_2023_phase_1/`
  (source tag `v2.5.0`, commit `29062af6d077409e1c37a3e53a6cac30fd4d02bc`).
  The clone is fetched/verified by `scripts/01_fetch_pinned_dataset.py`, which avoids
  CityLearn's named-dataset download path (GitHub anonymous-API rate limits).
- **Why this scenario:** chosen in Week 1 after rejecting the 2020 climate-zone example
  (9 buildings, inactive price observations, no cooling-device action) — rationale in
  [`environment-selection.md`](environment-selection.md).
- **What the project uses:** only **Building_1** of the three buildings, via the derived
  schema `configs/schema-building1.json` (central agent, 720-step episode).
- Row *i* (0-based after the header) is simulation step *t = i*. No timestamps — time is
  calendar-categorical. Hours run **1–24** (step 0 is hour 1); `day_type` ∈ 1–7
  (8 = holiday, never occurs); `month` = 6 only; `daylight_savings_status` = 0 always.

## Files

| File | Rows | Role |
| --- | --- | --- |
| `Building_1.csv` | 720 | The building's recorded energy simulation — the world the controller acts in, and the KPI normalization baseline |
| `Building_2.csv`, `Building_3.csv` | 720 each | Other district buildings — identical 16-column layout, unused by this project |
| `weather.csv` | 720 | 4 weather actuals + 12 dataset-issued forecast columns (3 horizons × 4 variables) |
| `pricing.csv` | 720 | Electricity price + 3 forecast columns |
| `carbon_intensity.csv` | 720 | Grid carbon intensity |
| `Building_{1,2,3}.pth` | — | Pretrained LSTM building-dynamics models (torch), one per building |
| `schema.json` | — | Parent scenario definition (assets, active observations/actions, bounds) |

## `Building_1.csv` — the 16 columns, with verified ranges

| Column | Range / mean | Notes |
| --- | --- | --- |
| `month`, `hour`, `day_type`, `daylight_savings_status` | 6; 1–24; 1–7; 0 | Calendar |
| `indoor_dry_bulb_temperature` | 20.00–27.22 °C | In-env this slot is **replaced by the LSTM dynamics model's** prediction (actions change it); the CSV column is the original baseline operation |
| `average_unmet_cooling_setpoint_difference` | tiny (σ 0.04) | Diagnostic; not an active observation |
| `indoor_relative_humidity` | ~46–51 % | Context only (inactive) |
| `non_shiftable_load` | mean 0.626 kWh | Fixed electrical load (≈ 15 kWh/day) |
| `dhw_demand` | mean 0.214 kWh | Hot-water thermal demand |
| `cooling_demand` | mean 3.333 kWh | **Thermal** cooling demand — ≈ 80 % of the demand mix; serving it through the heat pump (efficiency ≈ 0.254) is why active controllers' consumption explodes |
| `heating_demand` | 0.0 everywhere | June, cooling-only |
| `solar_generation` | 0–703.6, mean 208.3 | **Raw original-building scale** — the env's PV output is exactly this × 0.0024 (2.4 kW nominal / 1000); max 1.649 kWh/h, mean 0.50 (corr = 1.000 with the CSV column) |
| `occupant_count` | 0–3, mean 2.27 | Small residential occupancy |
| `…_cooling_set_point`, `…_heating_set_point` | 20.00–27.22 °C | Scheduled setpoints (identical tracks); the comfort band is this + ΔT = 2 °C |
| `hvac_mode` | 1 | Cooling mode |

## `weather.csv`

Actuals: `outdoor_dry_bulb_temperature` (21.38–40.32 °C, mean 29.6), `outdoor_relative_humidity`
(24–100 %), `diffuse_solar_irradiance` (max 466.6 W/m²), `direct_solar_irradiance`
(max 908.5 W/m²). The sun is fully down on 316 of 720 steps.

The 12 `*_predicted_1/2/3` columns are forecasts issued at row *t* for *t+1/2/3* — and they
are **deliberately poor** (the 2023 challenge's premise):

- Temperature h+1 forecast: MAE **4.83 °C** vs a trivial persistence forecast's **1.04 °C**.
- Correlation peaks at the wrong lag (pred₁ correlates 0.86 with t+4, 0.30 with t+1); the
  forecast's diurnal cycle is phase-shifted vs actuals.
- Price h+1 forecasts: exact-match rate 56 %.

This is precisely why Week 4 builds its own forecasting module with a persistence floor.

**Causality rule (binding):** reading any actual (non-`*predicted*`) column at a row > t is
lookahead and forbidden everywhere; `*predicted*` columns at row t are always legal.

## `pricing.csv` — the tariff

Three levels (USD/kWh): **off-peak 0.02893** (405 steps), **mid 0.02915** (252),
**peak 0.05867** (63 = 3 h × 21 days). Per day (hours 1-indexed): off-peak 1–7/23/24,
mid 8–15 and 19–22, **peak 16–18**. The 3-tier schedule applies on 21 of 30 days
(day_types 1–5); the other 9 days (day_types 6–7, the weekend class) are flat off-peak.
Peak/mid ratio ≈ 2.01× — too small for battery arbitrage to recover round-trip losses
(a core recorded finding).

## `carbon_intensity.csv`

0.338–0.556 kgCO₂/kWh, mean 0.454 — contextual observation, not in the reward.

## Building assets (from the schema)

| Asset | Parameters |
| --- | --- |
| PV | 2.4 kW nominal; env output = dataset column × 0.0024 |
| Battery | 4.0 kWh, 3.32 kW, η 0.95 (power-efficiency curve 0.88–0.95), DoD 0.8 → idles at SoC ≈ 0.2 |
| DHW tank | 2.2827 kWh, loss coeff 0.0032 — tiny vs demand; every controller so far leaves it at SoC 0 |
| Cooling device | heat pump, 4.11 kW nominal, efficiency 0.2535, target 8.0 °C — the action is a ratio of nominal electrical input |
| DHW device | electric heater, 4.86 kW, η 0.939 |
| Dynamics | pretrained LSTM: input 13, hidden 16, 2 layers, lookback 12 |

## How the environment composes observations from these files

The 29-slot observation = CSV columns (calendar, non-shiftable load, demands, occupancy,
setpoints) + weather/pricing/carbon files at row t (actuals + forecasts) + **computed slots**
(both SoCs from device models, solar from the PV model, net consumption from device
aggregation, indoor temperature from the LSTM). The computed slots are exactly the ones with
the CityLearn 2.5.0 zero-write quirk — the post-step observation vector carries 0.0 for them;
the harness repairs inputs causally from the building time series (see
[`cmdp-spec.md`](cmdp-spec.md) §1 measurement note).

**KPI normalization reference = this same dataset:** CityLearn's normalized KPIs divide by
the original recorded operation, i.e. 1.0 = "business as usual". That is why B0 (never
cooling) scores 0.44 on cost.

## How the project slices it

| Slice | Value | Use |
| --- | --- | --- |
| Dev window | steps 0–167 (first ~7 days) | training, checkpoint selection, every tuning decision |
| Final window | steps 0–719 (full month) | evaluation only — never selected on |
| Seeds | 42 (baselines/smoke), 42/43/44 (PPO) | variance quantification |
| Normalisation | from the B0 dev trace + schema static ranges | frozen in `configs/week3-ppo.yaml` |
