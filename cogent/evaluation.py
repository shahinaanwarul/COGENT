from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

from .dataset import COGENTWindowDataset
from .metrics import auc, best_f1_threshold, empirical_crps_np, f1, interval_metrics, mae, rmse
from .model import COGENTModel


@dataclass
class EvaluationResult:
    metrics_by_horizon: Dict[int, Dict[str, float]]
    target: np.ndarray
    prediction: np.ndarray
    probability: np.ndarray
    samples: np.ndarray
    emergence_target: np.ndarray | None
    threshold: float | None


@torch.no_grad()
def collect_predictions(
    model: COGENTModel,
    dataset: COGENTWindowDataset,
    device: torch.device | str,
    n_samples: int | None = None,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    model.eval()
    ys: List[np.ndarray] = []
    means: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    draws: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    limit = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for i in range(limit):
        s = dataset[i].to(device)
        out = model(s, n_samples=n_samples)
        ys.append(s.target.detach().cpu().numpy())
        means.append(out.forecast.mean_trajectory[: s.target.numel()].detach().cpu().numpy())
        probs.append(out.forecast.mean_emergence_probability[: s.target.numel()].detach().cpu().numpy())
        draws.append(out.forecast.trajectory_samples[:, : s.target.numel()].detach().cpu().numpy())
        if s.emergence_target is not None:
            labels.append(s.emergence_target.detach().cpu().numpy())
    y = np.stack(ys)
    pred = np.stack(means)
    prob = np.stack(probs)
    sample_arr = np.stack(draws)  # [S,K,H]
    label_arr = np.stack(labels) if labels else None
    return y, pred, prob, sample_arr, label_arr


def evaluate_dataset(
    model: COGENTModel,
    dataset: COGENTWindowDataset,
    device: torch.device | str,
    horizons: Sequence[int],
    threshold: float | None = None,
    threshold_dataset: COGENTWindowDataset | None = None,
    n_samples: int | None = None,
    max_samples: int | None = None,
) -> EvaluationResult:
    y, pred, prob, draws, labels = collect_predictions(model, dataset, device, n_samples=n_samples, max_samples=max_samples)
    chosen_threshold = threshold
    if labels is not None and chosen_threshold is None:
        if threshold_dataset is not None:
            vy, vp, vprob, vdraws, vlabels = collect_predictions(model, threshold_dataset, device, n_samples=n_samples, max_samples=max_samples)
            if vlabels is not None:
                chosen_threshold = best_f1_threshold(vlabels.reshape(-1), vprob.reshape(-1))
        if chosen_threshold is None:
            chosen_threshold = 0.5

    metrics: Dict[int, Dict[str, float]] = {}
    for h in horizons:
        idx = min(int(h), y.shape[1]) - 1
        yt = y[:, idx]
        yp = pred[:, idx]
        p = prob[:, idx]
        # Pool predictive draws for sample-wise CRPS and interval metrics.
        crps_values = []
        cover_values = []
        width_values = []
        cal_values = []
        for s in range(y.shape[0]):
            d = draws[s, :, : idx + 1]
            t = y[s, : idx + 1]
            crps_values.append(empirical_crps_np(d, t))
            im = interval_metrics(d, t, nominal=0.90)
            cover_values.append(im["coverage"])
            width_values.append(im["interval_width"])
            cal_values.append(im["absolute_calibration_error"])
        m = {
            "RMSE": rmse(yt, yp),
            "MAE": mae(yt, yp),
            "CRPS": float(np.mean(crps_values)),
            "coverage_90": float(np.mean(cover_values)),
            "interval_width_90": float(np.mean(width_values)),
            "calibration_error_90": float(np.mean(cal_values)),
        }
        if labels is not None:
            lab = labels[:, idx]
            m["F1"] = f1(lab, p, threshold=float(chosen_threshold))
            m["AUC"] = auc(lab, p)
        metrics[int(h)] = m
    return EvaluationResult(metrics, y, pred, prob, draws, labels, chosen_threshold)
