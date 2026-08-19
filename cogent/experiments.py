from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .config import DataConfig
from .dataset import COGENTSample, COGENTWindowDataset
from .graph import ablate_graph
from .metrics import rmse
from .model import COGENTModel


@dataclass
class DateFold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def rolling_date_folds(
    dates: Sequence[pd.Timestamp],
    n_folds: int = 5,
    train_fraction: float = 0.70,
    val_fraction: float = 0.10,
    test_fraction: float = 0.20,
) -> List[DateFold]:
    """Construct expanding rolling-origin folds when exact fold boundaries are absent.

    The manuscript states five rolling temporal folds and a 70/10/20 chronology but
    does not list the exact calendar boundaries. This function therefore makes the
    boundary rule explicit and reproducible rather than pretending the dates were
    specified. The final fold uses the full date range; earlier folds end earlier.
    """
    unique = pd.DatetimeIndex(sorted(pd.unique(pd.to_datetime(dates))))
    n = len(unique)
    min_end = max(10, int(np.ceil(n * 0.60)))
    endpoints = np.linspace(min_end, n, n_folds, dtype=int)
    folds: List[DateFold] = []
    for i, end in enumerate(endpoints, start=1):
        sub = unique[:end]
        nsub = len(sub)
        t1 = max(1, int(np.floor(nsub * train_fraction)))
        t2 = max(t1 + 1, int(np.floor(nsub * (train_fraction + val_fraction))))
        t2 = min(t2, nsub - 1)
        folds.append(DateFold(i, sub[0], sub[t1 - 1], sub[t1], sub[t2 - 1], sub[t2], sub[-1]))
    return folds


def earliest_training_fraction(df: pd.DataFrame, config: DataConfig, fraction: float) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby(config.hashtag_col, sort=False):
        g = g.sort_values(config.date_col)
        n = max(1, int(np.ceil(len(g) * float(fraction))))
        parts.append(g.iloc[:n])
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[:0].copy()


@torch.no_grad()
def evaluate_graph_ablation(
    model: COGENTModel,
    dataset: COGENTWindowDataset,
    condition: str,
    device: torch.device | str,
    max_samples: int | None = None,
) -> float:
    model.eval()
    true = []
    pred = []
    limit = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for i in range(limit):
        s = dataset[i].to(device)
        s = COGENTSample(
            hashtag=s.hashtag,
            origin_date=s.origin_date,
            modalities=s.modalities,
            lifecycle=s.lifecycle,
            target=s.target,
            emergence_target=s.emergence_target,
            graph=ablate_graph(s.graph, condition),
            target_node_index=s.target_node_index,
            modality_node_indices=s.modality_node_indices,
        )
        out = model(s, n_samples=min(32, model.config.model.monte_carlo_samples_eval))
        true.append(float(s.target[-1].cpu()))
        pred.append(float(out.forecast.mean_trajectory[s.target.numel() - 1].cpu()))
    return rmse(np.asarray(true), np.asarray(pred))
