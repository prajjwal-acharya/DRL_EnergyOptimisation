# Results and Findings

**Last updated:** 1 September 2026. All KPIs CityLearn-normalised (lower = better;
1.0 = the dataset's recorded business-as-usual operation). Windows: dev = steps 0–167
(selection), final = 0–719 (evaluation-only). Deep narrative with per-phase detail:
[`research-log.md`](research-log.md). **None of this is a savings claim.**

## Controllers in one table

| Controller | Dev cost | Final cost | Dev discomfort | Final discomfort | Grid exceedances (dev/final) |
| --- | --- | --- | --- | --- | --- |
| B0 neutral (do nothing) | **0.4420** | **0.3636** | 0.9152 | 0.9804 | 9 / 38 |
| B1 fixed schedule | 1.8631 | 1.7259 | 0.8788 | 0.9719 | 101 / 425 |
| B2 tariff-aware | 1.6709 | 1.5568 | **0.8667** | **0.9691** | 100 / 417 |
| PPO seed 42 (ckpt 40k) | 0.7524 | 0.6583 | 0.4485 | 0.5288 | 30 / 123 |
| PPO seed 43 (ckpt 20k) | 0.7263 | 0.6388 | 0.4848 | 0.5400 | 17 / 82 |
| PPO seed 44 (ckpt 100k) | 0.7526 | 0.6741 | **0.3576** | **0.2735** | 24 / 102 |

PPO multi-seed dev means (min–max): cost 0.7438 (0.7263–0.7526), consumption 0.7544,
discomfort 0.4303 (0.3576–0.4848), peak 0.9252, ramping 1.0224.

Training itself: 200,704 steps/seed, 1201 episodes, ≈ 5 min/seed on CPU, zero NaNs, zero
pre-clip action violations. Episode returns improve ≈ −845 → ≈ −380 (comfort-driven) while
cost bottoms early (20k–40k) and drifts up — training trades cost for comfort.

## The findings (in order of importance)

1. **No controller beats doing nothing on cost.** B0 is cheapest on both windows because
   it never runs the cooling device — paying with 91.5–98.0 % hot discomfort. Every active
   controller buys comfort with 60–180 % more consumption. This is the frontier the
   proposed method must improve.
2. **PPO is decisively better than the naive active baselines** — ≈ 2.2× cheaper than
   B1/B2 on both windows while roughly halving their discomfort. The learned policy is
   genuinely controlling; it just cannot beat doing nothing.
3. **The frozen reward (w_C = 10) explains the trade.** Comfort dominates the return by
   design; the policy cuts hot discomfort from 91.5 % to ~29–48 % at the cost of ~70 %
   more consumption. A useful intuition: the reward prices each degree-hour of overheating
   at ≈ 4.8 kWh of energy.
4. **Grid-limit discipline is the open safety problem.** Exceedances vs
   P_grid,max = 1.8084 kWh: B0 9/38, B1 101/425, B2 100/417, PPO 30/17/24 (dev) and
   123/82/102 (final). Penalty-only training does not enforce the import limit — the
   direct motivation for the planned safety shield (RQ2).
5. **Storage arbitrage is a trap at this tariff spread.** Peak/mid ratio < 2.03 cannot
   recover battery round-trip losses; B2's battery drains to the reserve edge and stays;
   the DHW tank never charges above 0. Week-3 PPO independently learned to ignore the
   battery (SoC within 0.186–0.200 in every run) — two different controller families
   converging on "don't cycle storage".
6. **`cooling_device` is the dominant cost lever** and it is an electrical-input ratio,
   not a setpoint. B1/B2 driving it at 0.5–0.8 daytime over-cools the building
   (cold-side discomfort ≈ 97 %) — fixed open-loop cooling without indoor-temperature
   feedback is the wrong controller shape.
7. **Tariff awareness helps where it can act** (B2 > B1 on cost, consumption, ramping,
   discomfort on both windows) — supports the RQ3 contrast price-adaptive vs calendar-only.
8. **Cost-based checkpoint selection is fragile** — it picks early, low-return policies
   (20k–100k of 200k) and behaves differently per seed; seed 44 shows the largest
   behavioural spread (much lower discomfort at similar cost).
9. **The discipline works.** Byte-frozen configs, 1e-9 anchor regressions, gates that
   independently re-execute selection rules and check byte-identity of prior evidence
   have kept three phases of autonomous execution reproducible — which is why the
   negative results above are trustworthy.

## Where the evidence lives

> `results/` is currently **cleared for regeneration** (see `results/README.md`).
> The paths below describe where each script re-creates the evidence when run;
> the original artifacts are preserved in the previous workspace
> (`/Volumes/code/Research Project/code/outputs/`).

- Baseline runs: `results/runs/baselines/<controller>/{dev,final}/` · PPO:
  `results/runs/ppo/seed{42,43,44}/` (checkpoints, evaluations, selections, final runs).
- Tables: `results/tables/{baseline_comparison,ppo_multiseed_summary,ppo_vs_baselines}.csv`.
- Figures: `results/figures/` (cost bars, 48 h demand overlays, SoC and temperature traces,
  return and KPI curves per seed).
- Forecasting (Week 4) and uncertainty-aware PPO (Week 5): **no results yet** — not started.
