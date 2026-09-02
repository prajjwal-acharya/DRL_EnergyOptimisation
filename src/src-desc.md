# src/ — the library

Reusable logic only. Runnable commands live in `scripts/`, decisions/constants in `configs/`.

| File | ~Lines | Significance |
| --- | --- | --- |
| `environment.py` | 110 | CityLearn interface: load env, derive single-building schema, inspect scenario. |
| `observation_names.py` | 97 | Name→index map for the 29 observations — no magic indices anywhere. |
| `baselines/controllers.py` | 282 | `Controller` ABC (the universal contract) + B0 neutral, B1 schedule, B2 tariff-aware. |
| `evaluation/runner.py` | 346 | **The locked harness** — every reported number comes through `run_episode()`. Owns the causal observation-repair and executed-value conventions. |
| `evaluation/metrics.py` | 121 | Pure trace→metrics functions: comfort, reserve, clipping, peak, solar, grid limit. |
| `evaluation/artifacts.py` | 118 | Writes the standard 5-file run set every run shares. |
| `rl/env_adapter.py` | 422 | **The Gymnasium bridge**: normalisation, action mapping, frozen CMDP reward, truncation contract. |
| `rl/controller.py` | 205 | `PPOController` — SB3 model on the Controller ABC + exact training-return replay from traces. |
| `rl/checkpoint_selection.py` | 52 | The frozen selection rule: lowest dev cost, tie-break lower discomfort. |
| `forecasting/data.py` | 300 | Aligned CSV loader, frozen folds, causal 22-value static and 24×5 sequence features. |
| `forecasting/metrics.py` | 90 | NaN-strict MAE/RMSE, pinball, coverage, width, Winkler, and monotonicity repair. |
| `forecasting/models.py` | 520 | Persistence/climatology/linear/GRU ladder, conformal wrapper, deterministic serialization. |
| `forecasting/pipeline.py` | 560 | Rolling-origin backtest, scoring, frozen selection, full-series refit, tables and figures. |
| `forecasting/api.py` | 90 | `ForecastProvider` — selected-model loading and Week-5 point/interval feature blocks. |
| `safety/` | empty | Week 6+ target: the safety shield wrapping actions before clip. |

```text
configs ─▶ controllers/rl-controller ─▶ runner.run_episode ─▶ metrics + artifacts
       └▶ forecasting pipeline ─▶ ForecastProvider ─▶ Week-5 feature blocks
                 (Controller ABC)          (name-addressed, causal)
```

Rules: no side effects at import; constants injected from configs, never embedded;
one measurement path for every controller; causality enforced (t−1 values, no lookahead).

Deep walkthrough: `docs/reference/folder-map.md` · contracts: `tests/test-desc.md`.
