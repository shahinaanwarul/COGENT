from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch

from .dataset import COGENTSample
from .model import COGENTModel


@dataclass
class EGAMExplanation:
    relation_attribution: Dict[str, torch.Tensor]
    relation_importance: Dict[str, float]
    modality_attribution: Dict[str, torch.Tensor]
    temporal_importance: torch.Tensor
    feature_importance: Dict[str, torch.Tensor]


class EGAM:
    """Integrated-gradient explanation head over relation states and modalities."""

    def __init__(self, model: COGENTModel, steps: int = 32) -> None:
        self.model = model
        self.steps = int(steps)

    def _objective(self, output, horizon_index: int | None) -> torch.Tensor:
        y = output.forecast.mean_trajectory
        if horizon_index is None:
            return y.mean()
        return y[int(horizon_index)]

    def relation_integrated_gradients(self, sample: COGENTSample, horizon_index: int | None = None) -> Dict[str, torch.Tensor]:
        self.model.eval()
        base = self.model.chgdm(sample.graph, sample.lifecycle[-1])
        actual = {r: t.detach() for r, t in base.relation_states.items()}
        accum = {r: torch.zeros_like(t) for r, t in actual.items()}
        for alpha in torch.linspace(0.0, 1.0, self.steps, device=sample.graph.node_features.device):
            scaled = {r: (alpha * t).detach().requires_grad_(True) for r, t in actual.items()}
            out = self.model.forward_with_relation_states(sample, base, scaled, n_samples=1, deterministic=True)
            objective = self._objective(out, horizon_index)
            grads = torch.autograd.grad(objective, list(scaled.values()), retain_graph=False, create_graph=False, allow_unused=False)
            for r, grad in zip(scaled, grads):
                accum[r] = accum[r] + grad.detach()
        return {r: actual[r] * accum[r] / float(self.steps) for r in actual}

    def modality_integrated_gradients(self, sample: COGENTSample, horizon_index: int | None = None) -> Dict[str, torch.Tensor]:
        self.model.eval()
        actual = {m: x.detach() for m, x in sample.modalities.items()}
        accum = {m: torch.zeros_like(x) for m, x in actual.items()}
        for alpha in torch.linspace(0.0, 1.0, self.steps, device=sample.graph.node_features.device):
            scaled = {m: (alpha * x).detach().requires_grad_(True) for m, x in actual.items()}
            s = COGENTSample(
                hashtag=sample.hashtag,
                origin_date=sample.origin_date,
                modalities=scaled,
                lifecycle=sample.lifecycle,
                target=sample.target,
                emergence_target=sample.emergence_target,
                graph=sample.graph,
                target_node_index=sample.target_node_index,
                modality_node_indices=sample.modality_node_indices,
            )
            out = self.model(s, n_samples=1, deterministic=True)
            objective = self._objective(out, horizon_index)
            grads = torch.autograd.grad(objective, list(scaled.values()), retain_graph=False, create_graph=False, allow_unused=False)
            for m, grad in zip(scaled, grads):
                accum[m] = accum[m] + grad.detach()
        return {m: actual[m] * accum[m] / float(self.steps) for m in actual}

    def explain(self, sample: COGENTSample, horizon_index: int | None = None) -> EGAMExplanation:
        rel = self.relation_integrated_gradients(sample, horizon_index)
        mod = self.modality_integrated_gradients(sample, horizon_index)
        rel_l1 = {r: float(v.abs().sum().detach().cpu()) for r, v in rel.items()}
        total = sum(rel_l1.values()) or 1.0
        relation_importance = {r: v / total for r, v in rel_l1.items()}
        temporal = torch.zeros_like(sample.lifecycle)
        feature: Dict[str, torch.Tensor] = {}
        for m, a in mod.items():
            temporal = temporal + a.abs().sum(dim=-1)
            feature[m] = a.abs().mean(dim=0)
        temporal = temporal / temporal.sum().clamp_min(1e-8)
        return EGAMExplanation(rel, relation_importance, mod, temporal, feature)
