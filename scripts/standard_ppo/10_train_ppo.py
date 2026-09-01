"""Phase B single-seed PPO bring-up trainer.

Plan reference: docs/plans/week3-implementation-plan.md §B1–§B2. Trains SB3 ``PPO``
on the Phase A :class:`CityLearnRLEnv` adapter for the dev window (0–167)
pinned to the configured device (``cpu``), writing under
``results/runs/ppo/seed<seed>/``:

- numbered checkpoints every ``checkpoint_every`` steps plus a ``final.zip``,
- the SB3 monitor CSV (``monitor.csv``),
- ``run_metadata.json`` (git commit, config hash, torch/SB3 versions,
  device, seed, wall time, sanity-gate outcomes).

Every hyperparameter is read from the frozen ``ppo`` block of the config;
nothing is tuned here. The return-curve figure
``results/figures/ppo_seed<seed>_return_curve.png`` (raw episode return vs
environment steps) is written from the monitor log after training.

Usage:

    ./.venv/bin/python scripts/standard_ppo/10_train_ppo.py --config configs/week3-ppo.yaml --seed 42

``--output-dir`` / ``--figures-dir`` / ``--total-timesteps`` exist solely for
short code-validation smoke runs outside the real artifact tree; real runs use
the defaults, which honour the config exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stable_baselines3
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from energy_optimisation.evaluation.runner import resolve_git_commit
from energy_optimisation.rl import CityLearnRLEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/week3-ppo.yaml",
        help="Frozen week-3 PPO config (hyperparameters live here, not in code).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Training seed; defaults to the config's ppo.seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Artifact directory; defaults to results/runs/ppo/seed<seed> (smoke runs only).",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures",
        help="Figure directory (smoke runs may redirect it).",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total timesteps (code-validation smoke runs only; "
        "real runs must use the frozen config value).",
    )
    return parser.parse_args()


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_project_root(path: Path) -> str:
    """Project-relative path for artifacts inside the repo; absolute otherwise.

    Real runs always write inside the project root. Smoke runs redirected to a
    temp directory fall back to the absolute path rather than failing.
    """

    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


class BringupCallback(BaseCallback):
    """Numbered checkpointing + pre-clip violation bookkeeping (plan §B1/§B2)."""

    def __init__(self, checkpoint_dir: Path, checkpoint_every: int) -> None:
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_every = int(checkpoint_every)
        self.saved_checkpoints: List[str] = []
        self.pre_clip_violation_events = 0
        self.episodes_with_violations = 0
        self.episodes_completed = 0
        self._last_episode_count = 0
        self._episode_violations = 0

    def _on_training_start(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._last_episode_count = 0
        self._episode_violations = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None and bool(np.asarray(dones).any()):
            # Episode boundary (DummyVecEnv auto-reset carries the final info
            # of the finished episode in infos[-1]); close it out once.
            infos = self.locals.get("infos") or []
            if infos:
                last_info = infos[-1]
                count = last_info.get("pre_clip_violation_count")
                if count is not None:
                    delta = int(count) - self._last_episode_count
                    if delta > 0:
                        self.pre_clip_violation_events += delta
                        self._episode_violations += delta
            self._close_episode()

        if self.model.num_timesteps % self.checkpoint_every == 0:
            filename = f"ppo_{self.model.num_timesteps:08d}_steps.zip"
            self.model.save(self.checkpoint_dir / filename)
            self.saved_checkpoints.append(filename)
            print(f"[checkpoint] step {self.model.num_timesteps} -> {filename}", flush=True)

        return True

    def _close_episode(self) -> None:
        self.episodes_completed += 1
        if self._episode_violations > 0:
            self.episodes_with_violations += 1
        self._episode_violations = 0
        self._last_episode_count = 0


def load_monitor_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, skiprows=1)
    expected = {"r", "l", "t"}
    if not expected.issubset(frame.columns):
        raise ValueError(f"monitor log lacks expected columns {sorted(expected)}")
    return frame


def plot_return_curve(frame: pd.DataFrame, figures_dir: Path, seed: int) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / f"ppo_seed{seed}_return_curve.png"
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(frame["t"].to_numpy(), frame["r"].to_numpy(), linewidth=0.7)
    axis.set_xlabel("environment steps")
    axis.set_ylabel("raw episode return")
    axis.set_title(f"PPO bring-up learning curve (seed {seed}, dev window)")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def main() -> int:
    args = parse_args()

    with args.config.open() as handle:
        config = yaml.safe_load(handle)

    ppo_block = config.get("ppo")
    if not isinstance(ppo_block, dict):
        raise KeyError("config must provide a 'ppo' hyperparameter block (frozen before training)")

    schema_path = PROJECT_ROOT / config["schema_path"]
    seed = int(args.seed) if args.seed is not None else int(ppo_block["seed"])
    device = str(ppo_block.get("device", "cpu"))
    if device != "cpu":
        raise ValueError(
            f"training device must be cpu per plan §0 (got {device!r}); "
            "MPS/GPU kernels are non-deterministic"
        )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (PROJECT_ROOT / "outputs" / "ppo" / f"seed{seed}").resolve()
    )
    checkpoints_dir = output_dir / "checkpoints"
    figures_dir = args.figures_dir.resolve()

    requested_timesteps = (
        int(args.total_timesteps)
        if args.total_timesteps is not None
        else int(ppo_block["total_timesteps"])
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    env = CityLearnRLEnv(
        str(schema_path),
        config=config,
        simulation_start_time_step=int(config["windows"]["dev"]["simulation_start_time_step"]),
        simulation_end_time_step=int(config["windows"]["dev"]["simulation_end_time_step"]),
        central_agent=bool(config.get("central_agent", True)),
    )
    monitored_env = Monitor(env, filename=str(output_dir))

    policy_layers = [int(layer) for layer in ppo_block["policy_hidden_layers"]]
    model = PPO(
        policy=str(ppo_block.get("policy", "MlpPolicy")),
        env=monitored_env,
        seed=seed,
        device=device,
        verbose=1,
        n_steps=int(ppo_block["n_steps"]),
        batch_size=int(ppo_block["batch_size"]),
        n_epochs=int(ppo_block["n_epochs"]),
        learning_rate=float(ppo_block["learning_rate"]),
        gamma=float(ppo_block["gamma"]),
        gae_lambda=float(ppo_block["gae_lambda"]),
        clip_range=float(ppo_block["clip_range"]),
        ent_coef=float(ppo_block["ent_coef"]),
        vf_coef=float(ppo_block["vf_coef"]),
        max_grad_norm=float(ppo_block["max_grad_norm"]),
        policy_kwargs={"net_arch": policy_layers},
    )

    callback = BringupCallback(checkpoints_dir, int(ppo_block["checkpoint_every"]))

    started = time.time()
    model.learn(total_timesteps=requested_timesteps, callback=callback)
    elapsed = time.time() - started

    final_path = checkpoints_dir / "final.zip"
    model.save(final_path)

    monitor_path = output_dir / "monitor.csv"
    frame = load_monitor_csv(monitor_path)
    nan_values = int(frame[["r", "l", "t"]].isna().to_numpy().sum())
    finite = bool(np.isfinite(frame[["r", "l", "t"]].to_numpy()).all())

    figure_path = plot_return_curve(frame, figures_dir, seed)

    metadata: Dict[str, Any] = {
        "purpose": "week-3 phase B single-seed PPO bring-up",
        "plan_reference": "docs/plans/week3-implementation-plan.md §B",
        "controller": "PPO",
        "seed": seed,
        "device": device,
        "schema_path": relative_to_project_root(schema_path),
        "simulation_start_time_step": int(config["windows"]["dev"]["simulation_start_time_step"]),
        "simulation_end_time_step": int(config["windows"]["dev"]["simulation_end_time_step"]),
        "config_path": relative_to_project_root(args.config),
        "config_sha256": sha256_of_file(args.config),
        "git_commit": resolve_git_commit(PROJECT_ROOT),
        "stable_baselines3_version": stable_baselines3.__version__,
        "torch_version": torch.__version__,
        "gymnasium_version": gym.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "total_timesteps_requested": requested_timesteps,
        "total_timesteps_completed": int(model.num_timesteps),
        "checkpoint_every": int(ppo_block["checkpoint_every"]),
        "checkpoints": callback.saved_checkpoints + [final_path.name],
        "final_model_path": relative_to_project_root(final_path),
        "monitor_csv_path": relative_to_project_root(monitor_path),
        "return_curve_figure": relative_to_project_root(figure_path),
        "episodes_completed": int(len(frame)),
        "episodes_completed_callback_count": int(callback.episodes_completed),
        "episodes_with_violations": int(callback.episodes_with_violations),
        "pre_clip_violation_events": int(callback.pre_clip_violation_events),
        "monitor_nan_values": nan_values,
        "monitor_all_finite": finite,
        "wall_clock_seconds": round(elapsed, 2),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_config_ppo_block": dict(ppo_block),
    }
    metadata_path = output_dir / "run_metadata.json"
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=False)
        handle.write("\n")

    print("[gate] monitor NaN values:", nan_values, "(required: 0)", flush=True)
    print(
        "[gate] pre-clip action violations:",
        callback.pre_clip_violation_events,
        "(expected 0; any clipping is logged and reported)",
        flush=True,
    )
    print(
        "[gate] numbered checkpoints:",
        len(callback.saved_checkpoints),
        "+ final.zip",
        flush=True,
    )
    print("[gate] return curve:", relative_to_project_root(figure_path), flush=True)

    gates_ok = nan_values == 0 and finite
    if not gates_ok:
        print("[gate] FAIL: monitor log contains non-finite values", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
