# Week 4 Review — Probabilistic Demand and Solar Forecasting

**Completed:** 2 September 2026  
**Scope:** offline forecasting only; no control or RL integration  
**Config:** `configs/week4-forecasting.yaml` (SHA-256 `ba8ac087e76b3dcdf656a85b30987a9d855dd10eb95562e795ddf12317cd09bf`)

## 1. Infrastructure evidence

- Added a causal forecasting library split into `data.py` (aligned input and folds),
  `metrics.py` (NaN-strict point/interval scores), `models.py` (five-rung ladder and
  conformal wrapper), `pipeline.py` (backtest/selection/reporting), and `api.py`
  (`ForecastProvider`, the frozen Week-5 interface).
- The feature builder is the sole CSV-to-feature path. It reads actuals only through
  origin `t`, reads dataset-issued predicted columns only at `t`, and is protected by a
  mutation test that changes every future actual row.
- The frozen 12-fold expanding-window backtest ran on CPU with seed 42. It scored 480
  origins and 1,434 valid labelled `(t,h)` pairs per model/target. At the dataset boundary,
  h=1/2/3 have 479/478/477 labels; no values beyond row 719 were invented.
- Seven variants were recorded per target: the five raw ladder rungs and conformal
  variants of the linear and GRU models. All prediction files are finite and quantile
  monotonicity holds row-by-row.
- The suite now contains 72 passing tests (61 prior contracts plus 11 forecasting
  contracts). The three pre-Week-4 result tables retain their exact byte hashes.

The dev window (0–167) lies inside every training segment by construction. Therefore
**no dev-window forecast-accuracy claim is made** — the dev window's role is control
evaluation.

## 2. Forecasting results and frozen selection

The table uses the selection scope: daylight-only for solar and pooled for the two demand
targets. MAE is the median (`q50`) point error; raw solar is on the source CSV's original
generation scale, before CityLearn's 0.0024 PV conversion.

| Target | Selected variant | MAE | RMSE | Mean pinball | 90% coverage | 50% coverage | Mean 90% width |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `solar_generation` | `persistence_24h` | 44.1302 | 73.7495 | 22.0651 | 0.1241 | 0.1241 | 0.0000 |
| `non_shiftable_load` | `linear_quantile` | 0.2392 | 0.8270 | 0.0993 | 0.8849 | 0.4812 | 1.6543 |
| `cooling_demand` | `persistence_last` | 1.2230 | 1.6108 | 0.6115 | 0.0000 | 0.0000 | 0.0000 |

- **Non-shiftable load is the only calibrated learned winner.** Linear quantile passed
  every per-horizon coverage bar and the naive floor. The GRU had slightly lower pooled
  pinball, but failed at least one per-horizon coverage bar, so it was ineligible under
  the frozen selection rule.
- **Solar is a negative result.** Neither learned model was competitive with 24-hour
  persistence on daylight MAE, and neither raw nor conformal learned interval met the
  per-horizon coverage bars. The frozen fallback therefore ships `persistence_24h`.
- **Cooling is a calibration failure, not a point-accuracy failure.** Both learned models
  beat the persistence MAE floor, but raw intervals missed at least one per-horizon
  coverage bar and the declared conformal variants over-covered. The frozen rule therefore
  ships `persistence_last` rather than allowing post-hoc tuning.
- The conformal variants widened every learned interval mechanically. For load and
  cooling they over-covered; this is evidence that the declared absolute-residual
  widening is conservative on this short series.

solar_generation is exactly 0 at night. Pooled MAE/coverage are dominated by trivial
night zeros. All solar metrics are therefore reported twice: pooled over all OOS steps,
and restricted to daylight origins; selection uses the daylight-restricted scope.

## 3. Integration artifact and leakage disclosure

Each selected target variant was refit into `results/runs/forecasting/refit/` and is
loaded only through `ForecastProvider`. The provider returns the fixed 9-value point block
or 36-value interval block in target-then-horizon order for Week 5.

The learned feature contract needs 24 hours of history, while the control window starts
at row 0 and has no pre-dataset history. For steps 0–23 the provider therefore uses a
declared causal cold start: current-value persistence for every target/horizon with
degenerate intervals. From step 24 onward it uses the frozen selected models. No
pre-dataset values are invented.

The full-series refit for RL integration uses all 720 steps. That artefact is not the
source of the accuracy claim: the rolling-origin out-of-sample backtest above is the
honest forecast evidence. Any residual in-sample leakage in Week 5 is disclosed and is
symmetric across its point and interval comparison arms.

## 4. Plan-required supervisor wording and factual qualification

> The probabilistic forecasting module is complete: a five-rung model ladder is
> evaluated under a frozen rolling-origin backtest, with point accuracy (MAE/RMSE) and
> interval calibration (coverage, width, pinball, Winkler) reported per target and
> horizon, and a selected, calibrated forecaster per target frozen for controller
> integration. No controller, RL, or safety-shield work was touched.

The implementation and reporting portions of that update are satisfied. Its phrase
"calibrated forecaster per target" is not supported for solar or cooling: the frozen plan
explicitly requires persistence to ship when learned candidates fail, and that is the
observed outcome. Only non-shiftable load has a selected variant that passes both declared
coverage bars. This qualification prevents the required status wording from becoming an
unsupported result claim.

## 5. Verification and next boundary

- `python -m pytest -q`: 72 passed.
- `python scripts/forecasting/17_gate_week4.py`: the phase gate verifies config hashes,
  selection traces, all 21 prediction/metric pairs, 1,434-row boundary arithmetic,
  monotonicity, tables, five figures, documentation disclosures, and protected evidence.
- Re-running `16_compare_forecasters.py` reproduced `selected_models.json`
  byte-identically.
- Week 5 may consume these frozen artifacts. It must not reinterpret the persistence
  fallbacks as calibrated uncertainty; degenerate interval widths are a required
  diagnostic and an honest limitation of the RQ1 experiment.
