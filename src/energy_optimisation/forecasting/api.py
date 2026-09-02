"""Frozen Week-5-facing interface to selected Week-4 forecasters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .data import HORIZONS, QUANTILES, TARGETS, Dataset, load_dataset
from .models import ForecastModel, load_model


class ForecastProvider:
    """Load selected full-series refits and expose causal forecast feature blocks."""

    def __init__(
        self,
        selected_models_path: Path,
        dataset_dir: Optional[Path] = None,
        project_root: Optional[Path] = None,
        dataset: Optional[Dataset] = None,
    ) -> None:
        self.selected_models_path = Path(selected_models_path).expanduser().resolve()
        self.project_root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else self.selected_models_path.parents[3]
        )
        selection = json.loads(self.selected_models_path.read_text())
        entries = selection.get("targets", selection)
        missing = [target for target in TARGETS if target not in entries]
        if missing:
            raise ValueError(f"selection lacks forecast targets: {', '.join(missing)}")
        if dataset is not None:
            self.dataset = dataset
        else:
            configured_dir = dataset_dir or selection.get("dataset_dir")
            if configured_dir is None:
                raise ValueError("dataset_dir is required when the selection does not record it")
            directory = Path(configured_dir)
            if not directory.is_absolute():
                directory = self.project_root / directory
            self.dataset = load_dataset(directory)
        self.models: Dict[str, ForecastModel] = {}
        for target in TARGETS:
            artifact = Path(entries[target]["artifact_path"])
            if not artifact.is_absolute():
                artifact = self.project_root / artifact
            self.models[target] = load_model(artifact)

    def predict_quantiles(self, t: int) -> Dict[str, Dict[int, Dict[float, float]]]:
        origin = int(t)
        if origin < 0 or origin >= self.dataset.row_count:
            raise IndexError(
                f"forecast origin must be in [0, {self.dataset.row_count - 1}], got {origin}"
            )
        # Learned inputs require a 24-hour history, but Week 5 starts at row 0 and
        # there is no pre-dataset history. Use a declared causal cold start until row
        # 24: current-value persistence with degenerate intervals.
        if origin < 24:
            return {
                target: {
                    horizon: {
                        quantile: float(self.dataset.building.at[origin, target])
                        for quantile in QUANTILES
                    }
                    for horizon in HORIZONS
                }
                for target in TARGETS
            }
        causal_dataset = self.dataset.through(origin)
        return {
            target: self.models[target].predict_all(causal_dataset, origin)
            for target in TARGETS
        }

    def feature_vector(self, t: int, variant: str) -> np.ndarray:
        forecasts = self.predict_quantiles(int(t))
        values = []
        if variant == "point":
            for target in TARGETS:
                for horizon in HORIZONS:
                    values.append(forecasts[target][horizon][0.50])
        elif variant == "interval":
            for target in TARGETS:
                for horizon in HORIZONS:
                    quantiles = forecasts[target][horizon]
                    values.extend(
                        (
                            quantiles[0.05],
                            quantiles[0.50],
                            quantiles[0.95],
                            quantiles[0.95] - quantiles[0.05],
                        )
                    )
        else:
            raise ValueError("forecast feature variant must be 'point' or 'interval'")
        vector = np.asarray(values, dtype=np.float64)
        expected = 9 if variant == "point" else 36
        if vector.shape != (expected,) or not np.isfinite(vector).all():
            raise ValueError(f"{variant} forecast block must contain {expected} finite values")
        return vector
