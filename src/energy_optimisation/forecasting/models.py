"""Five-rung probabilistic forecasting model ladder.

Models share one small interface and keep all experiment choices injectable from the
frozen YAML config.  The learned models emit all horizons and quantiles jointly; this
keeps the implementation compact while preserving the plan's exact loss and inputs.
"""

from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

from .data import (
    HORIZONS,
    QUANTILES,
    Dataset,
    build_sequence_features,
    build_static_features,
    training_arrays,
)
from .metrics import enforce_quantile_monotonicity


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)


def _quantile_key(value: float) -> str:
    return f"q{int(round(float(value) * 100)):02d}"


class ForecastModel(ABC):
    """Common target-specific multi-horizon quantile forecaster."""

    model_name = "abstract"

    def __init__(
        self,
        target: str,
        horizons: Sequence[int] = HORIZONS,
        quantiles: Sequence[float] = QUANTILES,
        seed: int = 42,
    ) -> None:
        self.target = str(target)
        self.horizons = tuple(int(value) for value in horizons)
        self.quantiles = tuple(float(value) for value in quantiles)
        self.seed = int(seed)
        self.train_end: Optional[int] = None

    @abstractmethod
    def fit(self, dataset: Dataset, train_end: int) -> "ForecastModel":
        """Fit on rows strictly before ``train_end``."""

    @abstractmethod
    def _predict_matrix(self, dataset: Dataset, origin_t: int) -> np.ndarray:
        """Return raw ``[horizon, quantile]`` predictions."""

    def predict_all(self, dataset: Dataset, origin_t: int) -> Dict[int, Dict[float, float]]:
        matrix = np.asarray(self._predict_matrix(dataset, int(origin_t)), dtype=np.float64)
        expected = (len(self.horizons), len(self.quantiles))
        if matrix.shape != expected or not np.isfinite(matrix).all():
            raise ValueError(
                f"{self.model_name} returned invalid prediction shape/value: {matrix.shape}"
            )
        repaired = enforce_quantile_monotonicity(matrix)
        return {
            horizon: {
                quantile: float(repaired[h_index, q_index])
                for q_index, quantile in enumerate(self.quantiles)
            }
            for h_index, horizon in enumerate(self.horizons)
        }

    def predict_quantiles(
        self, dataset: Dataset, origin_t: int, horizon: int
    ) -> Dict[float, float]:
        return self.predict_all(dataset, origin_t)[int(horizon)]

    def state_payload(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "target": self.target,
            "horizons": self.horizons,
            "quantiles": self.quantiles,
            "seed": self.seed,
            "train_end": self.train_end,
        }


class PersistenceLast(ForecastModel):
    model_name = "persistence_last"

    def fit(self, dataset: Dataset, train_end: int) -> "PersistenceLast":
        self.train_end = int(train_end)
        return self

    def _predict_matrix(self, dataset: Dataset, origin_t: int) -> np.ndarray:
        value = float(dataset.building.at[int(origin_t), self.target])
        return np.full((len(self.horizons), len(self.quantiles)), value, dtype=np.float64)


class Persistence24h(ForecastModel):
    model_name = "persistence_24h"

    def fit(self, dataset: Dataset, train_end: int) -> "Persistence24h":
        self.train_end = int(train_end)
        return self

    def _predict_matrix(self, dataset: Dataset, origin_t: int) -> np.ndarray:
        values = [
            float(dataset.building.at[int(origin_t) + horizon - 24, self.target])
            for horizon in self.horizons
        ]
        return np.repeat(np.asarray(values)[:, None], len(self.quantiles), axis=1)


class ClimatologyHourly(ForecastModel):
    model_name = "climatology_hourly"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hourly_means: Dict[int, float] = {}

    def fit(self, dataset: Dataset, train_end: int) -> "ClimatologyHourly":
        frame = dataset.building.iloc[: int(train_end)]
        grouped = frame.groupby("hour")[self.target].mean()
        self.hourly_means = {int(hour): float(value) for hour, value in grouped.items()}
        if len(self.hourly_means) != 24:
            raise ValueError("climatology training segment does not contain all 24 hours")
        self.train_end = int(train_end)
        return self

    def _predict_matrix(self, dataset: Dataset, origin_t: int) -> np.ndarray:
        origin_hour = int(round(float(dataset.building.at[int(origin_t), "hour"])))
        values = [
            self.hourly_means[((origin_hour - 1 + horizon) % 24) + 1]
            for horizon in self.horizons
        ]
        return np.repeat(np.asarray(values)[:, None], len(self.quantiles), axis=1)

    def state_payload(self) -> Dict[str, Any]:
        payload = super().state_payload()
        payload["hourly_means"] = self.hourly_means
        return payload


class _LinearHead(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values)


def _joint_pinball_loss(
    prediction: torch.Tensor, truth: torch.Tensor, quantiles: Sequence[float]
) -> torch.Tensor:
    tau = torch.tensor(quantiles, dtype=prediction.dtype, device=prediction.device)
    residual = truth.unsqueeze(-1) - prediction
    return torch.maximum(tau * residual, (tau - 1.0) * residual).mean()


class LinearQuantile(ForecastModel):
    model_name = "linear_quantile"

    def __init__(
        self,
        *args: Any,
        learning_rate: float = 0.01,
        steps: int = 2000,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.learning_rate = float(learning_rate)
        self.steps = int(steps)
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_scale: Optional[np.ndarray] = None
        self.network: Optional[_LinearHead] = None

    def fit(self, dataset: Dataset, train_end: int) -> "LinearQuantile":
        _seed_everything(self.seed)
        _, static, _, labels = training_arrays(dataset, self.target, train_end, self.horizons)
        self.feature_mean = static.mean(axis=0)
        self.feature_scale = static.std(axis=0)
        self.feature_scale[self.feature_scale < 1e-12] = 1.0
        inputs = torch.tensor(
            (static - self.feature_mean) / self.feature_scale, dtype=torch.float32
        )
        truth = torch.tensor(labels, dtype=torch.float32)
        self.network = _LinearHead(
            static.shape[1], len(self.horizons) * len(self.quantiles)
        )
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        for _ in range(self.steps):
            optimizer.zero_grad()
            prediction = self.network(inputs).reshape(
                -1, len(self.horizons), len(self.quantiles)
            )
            loss = _joint_pinball_loss(prediction, truth, self.quantiles)
            loss.backward()
            optimizer.step()
        self.network.eval()
        self.train_end = int(train_end)
        return self

    def _predict_matrix(self, dataset: Dataset, origin_t: int) -> np.ndarray:
        if self.network is None or self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("linear_quantile must be fit before prediction")
        static = build_static_features(dataset, origin_t, self.target)
        inputs = torch.tensor(
            ((static - self.feature_mean) / self.feature_scale)[None, :], dtype=torch.float32
        )
        with torch.no_grad():
            output = self.network(inputs).reshape(len(self.horizons), len(self.quantiles))
        return output.numpy().astype(np.float64)

    def state_payload(self) -> Dict[str, Any]:
        if self.network is None:
            raise RuntimeError("cannot save an unfitted linear_quantile model")
        payload = super().state_payload()
        payload.update(
            {
                "learning_rate": self.learning_rate,
                "steps": self.steps,
                "feature_mean": self.feature_mean,
                "feature_scale": self.feature_scale,
                "network_state": self.network.state_dict(),
            }
        )
        return payload


class _GruNetwork(nn.Module):
    def __init__(self, static_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=5, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size + static_size, output_size)

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(sequence)
        return self.head(torch.cat((hidden[-1], static), dim=1))


class GruQuantile(ForecastModel):
    model_name = "gru_quantile"

    def __init__(
        self,
        *args: Any,
        hidden_size: int = 32,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 10,
        validation_fraction: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_size = int(hidden_size)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.validation_fraction = float(validation_fraction)
        self.static_mean: Optional[np.ndarray] = None
        self.static_scale: Optional[np.ndarray] = None
        self.sequence_mean: Optional[np.ndarray] = None
        self.sequence_scale: Optional[np.ndarray] = None
        self.network: Optional[_GruNetwork] = None
        self.best_epoch: Optional[int] = None

    def fit(self, dataset: Dataset, train_end: int) -> "GruQuantile":
        _seed_everything(self.seed)
        _, static, sequences, labels = training_arrays(
            dataset, self.target, train_end, self.horizons
        )
        split = int(np.floor(len(static) * (1.0 - self.validation_fraction)))
        if split <= 0 or split >= len(static):
            raise ValueError("GRU validation split leaves an empty train or validation segment")
        # Scaling is fit on the internal training portion only.
        self.static_mean = static[:split].mean(axis=0)
        self.static_scale = static[:split].std(axis=0)
        self.static_scale[self.static_scale < 1e-12] = 1.0
        self.sequence_mean = sequences[:split].reshape(-1, 5).mean(axis=0)
        self.sequence_scale = sequences[:split].reshape(-1, 5).std(axis=0)
        self.sequence_scale[self.sequence_scale < 1e-12] = 1.0

        x_static = torch.tensor(
            (static - self.static_mean) / self.static_scale, dtype=torch.float32
        )
        x_sequence = torch.tensor(
            (sequences - self.sequence_mean) / self.sequence_scale, dtype=torch.float32
        )
        y = torch.tensor(labels, dtype=torch.float32)
        self.network = _GruNetwork(
            static.shape[1], self.hidden_size, len(self.horizons) * len(self.quantiles)
        )
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        best_loss = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        stale_epochs = 0
        generator = torch.Generator().manual_seed(self.seed)

        for epoch in range(self.max_epochs):
            self.network.train()
            ordering = torch.randperm(split, generator=generator)
            for start in range(0, split, self.batch_size):
                batch = ordering[start : start + self.batch_size]
                optimizer.zero_grad()
                prediction = self.network(x_sequence[batch], x_static[batch]).reshape(
                    -1, len(self.horizons), len(self.quantiles)
                )
                loss = _joint_pinball_loss(prediction, y[batch], self.quantiles)
                loss.backward()
                optimizer.step()
            self.network.eval()
            with torch.no_grad():
                validation = self.network(x_sequence[split:], x_static[split:]).reshape(
                    -1, len(self.horizons), len(self.quantiles)
                )
                validation_loss = float(
                    _joint_pinball_loss(validation, y[split:], self.quantiles).item()
                )
            if validation_loss < best_loss - 1e-10:
                best_loss = validation_loss
                best_state = copy.deepcopy(self.network.state_dict())
                self.best_epoch = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError("GRU training did not produce a finite validation state")
        # Early stopping chooses only the epoch count. Refit from scratch on the entire
        # expanding segment so every row < fold start participates in the final fold
        # model, while fold observations remain completely untouched.
        self.static_mean = static.mean(axis=0)
        self.static_scale = static.std(axis=0)
        self.static_scale[self.static_scale < 1e-12] = 1.0
        self.sequence_mean = sequences.reshape(-1, 5).mean(axis=0)
        self.sequence_scale = sequences.reshape(-1, 5).std(axis=0)
        self.sequence_scale[self.sequence_scale < 1e-12] = 1.0
        x_static = torch.tensor(
            (static - self.static_mean) / self.static_scale, dtype=torch.float32
        )
        x_sequence = torch.tensor(
            (sequences - self.sequence_mean) / self.sequence_scale, dtype=torch.float32
        )
        _seed_everything(self.seed)
        self.network = _GruNetwork(
            static.shape[1], self.hidden_size, len(self.horizons) * len(self.quantiles)
        )
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(int(self.best_epoch)):
            ordering = torch.randperm(len(static), generator=generator)
            for start in range(0, len(static), self.batch_size):
                batch = ordering[start : start + self.batch_size]
                optimizer.zero_grad()
                prediction = self.network(x_sequence[batch], x_static[batch]).reshape(
                    -1, len(self.horizons), len(self.quantiles)
                )
                loss = _joint_pinball_loss(prediction, y[batch], self.quantiles)
                loss.backward()
                optimizer.step()
        self.network.eval()
        self.train_end = int(train_end)
        return self

    def _predict_matrix(self, dataset: Dataset, origin_t: int) -> np.ndarray:
        required = (
            self.network,
            self.static_mean,
            self.static_scale,
            self.sequence_mean,
            self.sequence_scale,
        )
        if any(value is None for value in required):
            raise RuntimeError("gru_quantile must be fit before prediction")
        static = build_static_features(dataset, origin_t, self.target)
        sequence = build_sequence_features(dataset, origin_t, self.target)
        static_tensor = torch.tensor(
            ((static - self.static_mean) / self.static_scale)[None, :], dtype=torch.float32
        )
        sequence_tensor = torch.tensor(
            ((sequence - self.sequence_mean) / self.sequence_scale)[None, :, :],
            dtype=torch.float32,
        )
        with torch.no_grad():
            output = self.network(sequence_tensor, static_tensor).reshape(
                len(self.horizons), len(self.quantiles)
            )
        return output.numpy().astype(np.float64)

    def state_payload(self) -> Dict[str, Any]:
        if self.network is None:
            raise RuntimeError("cannot save an unfitted gru_quantile model")
        payload = super().state_payload()
        payload.update(
            {
                "hidden_size": self.hidden_size,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "validation_fraction": self.validation_fraction,
                "best_epoch": self.best_epoch,
                "static_mean": self.static_mean,
                "static_scale": self.static_scale,
                "sequence_mean": self.sequence_mean,
                "sequence_scale": self.sequence_scale,
                "network_state": self.network.state_dict(),
            }
        )
        return payload


class Conformalized(ForecastModel):
    """Symmetric split-conformal interval widening around a fitted base model."""

    model_name = "conformalized"

    def __init__(self, base_model: ForecastModel) -> None:
        super().__init__(
            target=base_model.target,
            horizons=base_model.horizons,
            quantiles=base_model.quantiles,
            seed=base_model.seed,
        )
        self.base_model = base_model
        self.adjustments: Dict[int, Dict[str, float]] = {}
        self.train_end = base_model.train_end

    def fit(self, dataset: Dataset, train_end: int) -> "Conformalized":
        self.base_model.fit(dataset, train_end)
        self.train_end = int(train_end)
        return self

    @staticmethod
    def _higher_quantile(values: np.ndarray, probability: float) -> float:
        try:
            return float(np.quantile(values, probability, method="higher"))
        except TypeError:  # numpy < 1.22 compatibility
            return float(np.quantile(values, probability, interpolation="higher"))

    def calibrate(
        self,
        dataset: Dataset,
        origins: Sequence[int],
        calibration_model: Optional[ForecastModel] = None,
    ) -> "Conformalized":
        if len(origins) == 0:
            raise ValueError("conformal calibration requires at least one origin")
        scoring_model = calibration_model or self.base_model
        for horizon in self.horizons:
            residuals = []
            for origin in origins:
                prediction = scoring_model.predict_quantiles(dataset, int(origin), horizon)
                truth = float(dataset.building.at[int(origin) + horizon, self.target])
                residuals.append(abs(truth - prediction[0.50]))
            values = np.asarray(residuals, dtype=np.float64)
            self.adjustments[horizon] = {
                "delta90": self._higher_quantile(values, 0.90),
                "delta50": self._higher_quantile(values, 0.50),
            }
        return self

    def _predict_matrix(self, dataset: Dataset, origin_t: int) -> np.ndarray:
        raw = self.base_model.predict_all(dataset, origin_t)
        matrix = np.asarray(
            [[raw[horizon][quantile] for quantile in self.quantiles] for horizon in self.horizons],
            dtype=np.float64,
        )
        index = {quantile: position for position, quantile in enumerate(self.quantiles)}
        for h_index, horizon in enumerate(self.horizons):
            if horizon not in self.adjustments:
                raise RuntimeError("conformalized model must be calibrated before prediction")
            adjustment = self.adjustments[horizon]
            matrix[h_index, index[0.05]] -= adjustment["delta90"]
            matrix[h_index, index[0.95]] += adjustment["delta90"]
            matrix[h_index, index[0.25]] -= adjustment["delta50"]
            matrix[h_index, index[0.75]] += adjustment["delta50"]
        return matrix

    def state_payload(self) -> Dict[str, Any]:
        payload = super().state_payload()
        payload.update(
            {
                "base_model": self.base_model.state_payload(),
                "adjustments": self.adjustments,
            }
        )
        return payload


MODEL_TYPES = {
    PersistenceLast.model_name: PersistenceLast,
    Persistence24h.model_name: Persistence24h,
    ClimatologyHourly.model_name: ClimatologyHourly,
    LinearQuantile.model_name: LinearQuantile,
    GruQuantile.model_name: GruQuantile,
}


def create_model(
    model_name: str,
    target: str,
    config: Mapping[str, Any],
    horizons: Sequence[int] = HORIZONS,
    quantiles: Sequence[float] = QUANTILES,
    seed: int = 42,
) -> ForecastModel:
    """Construct a ladder rung exclusively from frozen configuration values."""

    common = {
        "target": target,
        "horizons": horizons,
        "quantiles": quantiles,
        "seed": seed,
    }
    if model_name == "linear_quantile":
        return LinearQuantile(
            **common,
            learning_rate=float(config["learning_rate"]),
            steps=int(config["steps"]),
        )
    if model_name == "gru_quantile":
        return GruQuantile(
            **common,
            hidden_size=int(config["hidden_size"]),
            learning_rate=float(config["learning_rate"]),
            batch_size=int(config["batch_size"]),
            max_epochs=int(config["max_epochs"]),
            patience=int(config["patience"]),
            validation_fraction=float(config["validation_fraction"]),
        )
    try:
        model_class = MODEL_TYPES[model_name]
    except KeyError as error:
        raise KeyError(f"unknown forecast model {model_name!r}") from error
    return model_class(**common)


def save_model(model: ForecastModel, path: Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_payload(), destination)


def _load_payload(payload: Mapping[str, Any]) -> ForecastModel:
    name = str(payload["model_name"])
    if name == "conformalized":
        base = _load_payload(payload["base_model"])
        model = Conformalized(base)
        model.adjustments = {
            int(horizon): {key: float(value) for key, value in adjustment.items()}
            for horizon, adjustment in payload["adjustments"].items()
        }
        return model
    model = create_model(
        name,
        target=str(payload["target"]),
        config=payload,
        horizons=payload["horizons"],
        quantiles=payload["quantiles"],
        seed=int(payload["seed"]),
    )
    model.train_end = int(payload["train_end"]) if payload.get("train_end") is not None else None
    if isinstance(model, ClimatologyHourly):
        model.hourly_means = {
            int(hour): float(value) for hour, value in payload["hourly_means"].items()
        }
    elif isinstance(model, LinearQuantile):
        model.feature_mean = np.asarray(payload["feature_mean"], dtype=np.float64)
        model.feature_scale = np.asarray(payload["feature_scale"], dtype=np.float64)
        model.network = _LinearHead(22, len(model.horizons) * len(model.quantiles))
        model.network.load_state_dict(payload["network_state"])
        model.network.eval()
    elif isinstance(model, GruQuantile):
        model.static_mean = np.asarray(payload["static_mean"], dtype=np.float64)
        model.static_scale = np.asarray(payload["static_scale"], dtype=np.float64)
        model.sequence_mean = np.asarray(payload["sequence_mean"], dtype=np.float64)
        model.sequence_scale = np.asarray(payload["sequence_scale"], dtype=np.float64)
        model.best_epoch = int(payload["best_epoch"])
        model.network = _GruNetwork(
            22, model.hidden_size, len(model.horizons) * len(model.quantiles)
        )
        model.network.load_state_dict(payload["network_state"])
        model.network.eval()
    return model


def load_model(path: Path) -> ForecastModel:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # torch versions before weights_only was accepted
        payload = torch.load(Path(path), map_location="cpu")
    return _load_payload(payload)
