from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence, Tuple

import torch

from .dataset import COGENTSample
from .graph import GraphSnapshot, project_adjacency
from .model import COGENTModel, COGENTOutput


@dataclass
class StructuralScenario:
    name: str
    relation: str
    strength: float
    operation: str = "multiply"  # multiply | add
    edges: Sequence[Tuple[int, int]] | None = None
    max_weight: float | None = None


@dataclass
class ScenarioResult:
    scenario: StructuralScenario
    baseline: COGENTOutput
    perturbed: COGENTOutput
    per_sample_contrast: torch.Tensor  # [K,H]
    ssc: torch.Tensor  # [H]
    asc: torch.Tensor  # [H]


def perturb_snapshot(snapshot: GraphSnapshot, scenario: StructuralScenario) -> GraphSnapshot:
    g = snapshot.clone()
    if scenario.relation not in g.adjacencies:
        raise KeyError(f"Relation {scenario.relation!r} is absent from this factual graph.")
    A = g.adjacencies[scenario.relation].clone()
    if scenario.edges is None:
        mask = A > 0
        if scenario.operation == "multiply":
            A = torch.where(mask, A * (1.0 + scenario.strength), A)
        elif scenario.operation == "add":
            A = torch.where(mask, A + scenario.strength, A)
        else:
            raise ValueError("operation must be 'multiply' or 'add'")
    else:
        for i, j in scenario.edges:
            if scenario.operation == "multiply":
                A[i, j] = A[i, j] * (1.0 + scenario.strength)
                A[j, i] = A[j, i] * (1.0 + scenario.strength)
            elif scenario.operation == "add":
                A[i, j] = A[i, j] + scenario.strength
                A[j, i] = A[j, i] + scenario.strength
            else:
                raise ValueError("operation must be 'multiply' or 'add'")
    g.adjacencies[scenario.relation] = project_adjacency(A, max_weight=scenario.max_weight)
    return g


class GSISM:
    """Generative Structural Intervention Sensitivity Model.

    Outputs are model-based structural sensitivity contrasts, not identified causal
    effects, matching the interpretation explicitly stated in the manuscript.
    """

    def __init__(self, model: COGENTModel) -> None:
        self.model = model

    def evaluate(self, sample: COGENTSample, scenario: StructuralScenario, n_samples: int | None = None, seed: int = 42) -> ScenarioResult:
        self.model.eval()
        K = int(n_samples or self.model.config.model.monte_carlo_samples_eval)
        H = self.model.config.data.max_horizon
        N = sample.graph.num_nodes
        D = self.model.config.model.latent_dim
        gen = torch.Generator(device=sample.graph.node_features.device)
        gen.manual_seed(seed)
        noise = torch.randn((K, H, N, D), generator=gen, device=sample.graph.node_features.device)
        obs_noise = torch.randn((K, H), generator=gen, device=sample.graph.node_features.device)

        with torch.no_grad():
            base = self.model(sample, n_samples=K, noise=noise, observation_noise=obs_noise)
            pert_sample = COGENTSample(
                hashtag=sample.hashtag,
                origin_date=sample.origin_date,
                modalities=sample.modalities,
                lifecycle=sample.lifecycle,
                target=sample.target,
                emergence_target=sample.emergence_target,
                graph=perturb_snapshot(sample.graph, scenario),
                target_node_index=sample.target_node_index,
                modality_node_indices=sample.modality_node_indices,
            )
            pert = self.model(pert_sample, n_samples=K, noise=noise, observation_noise=obs_noise)
        contrast = pert.forecast.trajectory_samples - base.forecast.trajectory_samples
        ssc = pert.forecast.mean_trajectory - base.forecast.mean_trajectory
        asc = contrast.mean(dim=0)
        return ScenarioResult(scenario, base, pert, contrast, ssc, asc)
