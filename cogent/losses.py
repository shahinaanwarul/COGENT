from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch
import torch.nn.functional as F

from .config import LossConfig
from .dataset import COGENTSample
from .model import COGENTOutput


def empirical_crps(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Empirical CRPS for samples [K,H] and target [H]."""
    first = torch.mean(torch.abs(samples - target.unsqueeze(0)), dim=0)
    pairwise = torch.abs(samples[:, None, :] - samples[None, :, :]).mean(dim=(0, 1))
    return (first - 0.5 * pairwise).mean()


def graph_smoothness(node_states: torch.Tensor, laplacians: Mapping[str, torch.Tensor], attention: Mapping[str, torch.Tensor]) -> torch.Tensor:
    loss = torch.zeros((), device=node_states.device)
    for r, L in laplacians.items():
        if r not in attention:
            continue
        LH = L @ node_states
        loss = loss + attention[r] * torch.mean(torch.sum(node_states * LH, dim=-1))
    return loss


def interval_calibration_error(samples: torch.Tensor, target: torch.Tensor, nominal: float = 0.90) -> torch.Tensor:
    alpha = (1.0 - nominal) / 2.0
    lo = torch.quantile(samples, alpha, dim=0)
    hi = torch.quantile(samples, 1.0 - alpha, dim=0)
    covered = ((target >= lo) & (target <= hi)).float().mean()
    return torch.abs(covered - nominal)


@dataclass
class LossBreakdown:
    total: torch.Tensor
    prediction_mse: torch.Tensor
    crps: torch.Tensor
    classification: torch.Tensor
    graph_smoothness: torch.Tensor
    calibration: torch.Tensor

    def scalar_dict(self) -> Dict[str, float]:
        return {k: float(getattr(self, k).detach().cpu()) for k in self.__dataclass_fields__}


def cogent_loss(output: COGENTOutput, sample: COGENTSample, cfg: LossConfig) -> LossBreakdown:
    pred = output.forecast.mean_trajectory[: sample.target.numel()]
    samples = output.forecast.trajectory_samples[:, : sample.target.numel()]
    mse = F.mse_loss(pred, sample.target)
    crps = empirical_crps(samples, sample.target)
    if sample.emergence_target is not None:
        p = output.forecast.mean_emergence_probability[: sample.emergence_target.numel()].clamp(1e-6, 1 - 1e-6)
        cls = F.binary_cross_entropy(p, sample.emergence_target)
    else:
        cls = torch.zeros((), device=pred.device)
    smooth = graph_smoothness(output.graph.node_states, output.graph.laplacians, output.graph.relation_attention)
    cal = interval_calibration_error(samples, sample.target, nominal=0.90)
    total = (
        cfg.prediction_mse_weight * mse
        + cfg.crps_weight * crps
        + cfg.classification_weight * cls
        + cfg.graph_smoothness_weight * smooth
        + cfg.calibration_weight * cal
    )
    return LossBreakdown(total, mse, crps, cls, smooth, cal)
