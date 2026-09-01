# Literature Evidence Review - Days 5-6

## Purpose and boundary

This is an implementation-oriented evidence review for the CP-I research questions, not a claim of a completed systematic review. The companion [`literature-matrix.csv`](literature-matrix.csv) contains 12 screened sources: six deeply analysed and six screening-level entries. The set spans building-energy RL, probabilistic forecasting, safe RL, and tariff-aware demand response.

The sources establish methodological possibilities. They do **not** establish that this project will save a particular percentage of energy or transfer safely to a real building.

## Selection protocol

- Prefer primary papers, official framework documentation, and official policy sources.
- Include a source only if it informs an implemented choice, a planned comparison, a safety mechanism, a forecast-evaluation method, or an India-context boundary.
- Mark a source `analysed` only after recording its setting, control assets, forecast/uncertainty method, safety treatment, baselines, metrics, limitation, and RQ relevance.
- Mark a source `screened` when its title, abstract, and bibliographic metadata establish relevance but it has not yet received that full extraction.
- Treat policy material as context, not as empirical performance evidence.

## Coverage

| Theme | Sources | Fully analysed | Use in CP-I |
| --- | ---: | ---: | --- |
| Building-energy RL | 4 | 2 | Select a simulator, state/action boundary, and fair baselines. |
| Probabilistic forecasting | 4 | 2 | Build point/quantile forecasts and evaluate interval quality. |
| Safe RL | 3 | 2 | Separate hard operational constraints from reward shaping. |
| Tariff-aware demand response | 2 | 1 | Define price scenarios without claiming a particular real tariff. |

## What the evidence supports

1. **A simulator-first comparison is appropriate.** CityLearn supports controlled comparisons among rule-based, model-predictive, and RL strategies. The current project narrows that broader setting to one Building 1 schema. [Nweye et al., 2024](https://arxiv.org/abs/2405.03848)
2. **Comfort and flexible-resource control must be measured together.** The household DRL study by Lissa et al. pairs energy/PV outcomes with a comfort boundary rather than treating consumption alone as success. It motivates the project's paired cost-comfort evaluation, not a reuse of its reported savings. [Lissa et al., 2021](https://doi.org/10.1016/j.egyai.2020.100043)
3. **Forecast-informed energy control exists, but point forecasts do not answer RQ1.** Ren et al. combine forecasts with RL scheduling, whereas the planned contribution is to compare point forecasts with calibrated intervals under the same controller and scenarios. [Ren et al., 2022](https://doi.org/10.1016/j.scs.2021.103207)
4. **Prediction intervals need their own quality evaluation.** Quantile regression supplies interval endpoints; CQR supplies a principled calibration route under its stated exchangeability assumptions. The project will report point accuracy, empirical coverage, and interval width before sending interval features to PPO. [Koenker and Bassett, 1978](https://doi.org/10.2307/1913643), [Romano et al., 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html)
5. **A reward penalty alone is an inadequate safety argument.** CPO motivates explicit constraints; Dalal et al. demonstrate action correction. CP-I will begin with an auditable projection/clipping and fallback design, then log every intervention. It will not claim theoretical guarantees that have not been proved for this schema. [Achiam et al., 2017](https://arxiv.org/abs/1705.10528), [Dalal et al., 2018](https://arxiv.org/abs/1801.08757)

## Evidence-backed gaps to test

These are hypotheses to evaluate, not claimed contributions until experiments are complete.

| Gap | Testable CP-I response | Research question |
| --- | --- | --- |
| Forecast-informed building-energy control often uses point forecasts. | Compare the same PPO architecture with point-only input and with point-plus-calibrated-interval input. | RQ1 |
| Safe RL concepts are not automatically operational building safeguards. | Add explicit comfort, SoC, action-power, and grid-import checks that project or replace unsafe actions. | RQ2 |
| Average performance can conceal poor behaviour under demand, PV, price, or forecast shifts. | Run a fixed scenario matrix across seeds and report spread and worst-case results. | RQ3 |
| Controller comparisons may over-credit RL when rule baselines are weak. | Implement fixed-schedule and tariff-aware rule controllers before PPO, then ablate forecast uncertainty and shield independently. | RQ1-RQ3 |

## Resulting implementation decisions

1. **Forecast baseline first:** persistence and transparent quantile regression precede GRU/LSTM experimentation.
2. **Forecast metrics:** MAE and RMSE for point quality; empirical interval coverage and mean interval width for interval quality.
3. **Controller sequence:** fixed schedule -> tariff-aware rule -> point-forecast PPO -> interval-aware PPO -> interval-aware PPO plus shield.
4. **Safety evidence:** report violation count/duration and every shield/fallback activation, in addition to cost.
5. **Generalisation evidence:** vary tariff profile, solar availability, demand/occupancy proxy, and forecast bias/noise under fixed random seeds.

## Literature limitations and next actions

- The current CityLearn scenario is not India-specific. The Ministry of Power source informs motivation only; no simulated tariff will be labelled as a real tariff.
- CQR's coverage statement depends on its assumptions and must be verified empirically under the project's shifts.
- Safe-RL methods cited here were not designed for the exact CityLearn Building 1 dynamics; their use is a design inspiration, not a transferred guarantee.
- The research plan's incomplete "Deep reinforcement learning for energy management in buildings: A review" citation is not entered as a formal source because it lacks verifiable bibliographic metadata. It must be resolved before inclusion in the final manuscript.
- The next literature action is to convert the six screened entries to analysed entries only when their full methods and limitations have been extracted.
