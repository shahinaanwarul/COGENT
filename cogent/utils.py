from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_tensor(x: Any, *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype=dtype)
    else:
        t = torch.as_tensor(x, dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def detach_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def safe_std(x: np.ndarray, eps: float = 1e-8) -> float:
    s = float(np.nanstd(x))
    return s if s > eps else 1.0


def sinusoidal_calendar_features(dates: Sequence[np.datetime64]) -> np.ndarray:
    import pandas as pd

    dt = pd.DatetimeIndex(dates)
    dow = dt.dayofweek.to_numpy(dtype=np.float32)
    doy = dt.dayofyear.to_numpy(dtype=np.float32)
    return np.stack(
        [
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
        ],
        axis=-1,
    ).astype(np.float32)


def lifecycle_coordinate(length: int, start_fraction: float = 0.0, end_fraction: float = 1.0) -> np.ndarray:
    if length <= 1:
        return np.array([end_fraction], dtype=np.float32)
    return np.linspace(start_fraction, end_fraction, length, dtype=np.float32)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), p)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def one_hot(index: int, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        out[index] = 1.0
    return out


def normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def batched(iterable: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]
