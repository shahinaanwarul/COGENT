from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from torch import nn


class ARIMABaseline:
    """Target-history-only ARIMA reference, as characterized in the manuscript."""

    def __init__(self, order=(2, 1, 2)) -> None:
        self.order = order

    def forecast(self, history: Sequence[float], horizon: int) -> np.ndarray:
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(np.asarray(history, dtype=float), order=self.order).fit()
        return np.asarray(model.forecast(horizon), dtype=np.float32)


class ProphetBaseline:
    """Optional Prophet wrapper. Install `prophet` to use it."""

    def forecast(self, dates, values, future_dates) -> np.ndarray:
        try:
            from prophet import Prophet
        except ImportError as exc:
            raise ImportError("Install `prophet` to run the Prophet baseline.") from exc
        import pandas as pd

        m = Prophet()
        m.fit(pd.DataFrame({"ds": dates, "y": values}))
        pred = m.predict(pd.DataFrame({"ds": future_dates}))
        return pred["yhat"].to_numpy(dtype=np.float32)


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        return self.head(y[:, -1])


class TransformerBaseline(nn.Module):
    """Matched-input deterministic Transformer reference.

    It consumes the flattened multimodal tensor but intentionally has no graph,
    latent SDE, structural scenario, attribution, or utility modules.
    """

    def __init__(self, input_dim: int, hidden_dim: int, heads: int, layers: int, horizon: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(hidden_dim, heads, 4 * hidden_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.proj(x))
        return self.head(z[:, -1])


def flatten_modalities(modalities: Mapping[str, torch.Tensor]) -> torch.Tensor:
    names = sorted(modalities)
    return torch.cat([modalities[k] for k in names], dim=-1)


class ExternalBaselineAdapter:
    """Explicit adapter point for TFT, Informer, TimesNet, and PatchTST.

    The manuscript names these baselines and gives common search controls, but it
    does not uniquely specify library/version-specific architectures. To avoid
    silently implementing a different model under the same name, this adapter asks
    callers to provide a fitted external estimator (e.g. NeuralForecast,
    PyTorch-Forecasting, or the authors' exact baseline code).
    """

    def __init__(self, estimator, name: str) -> None:
        self.estimator = estimator
        self.name = name

    def fit(self, *args, **kwargs):
        return self.estimator.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        if hasattr(self.estimator, "predict"):
            return self.estimator.predict(*args, **kwargs)
        raise TypeError(f"External {self.name} estimator has no predict() method.")
