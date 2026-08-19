from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence
import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import GraphConfig
from .graph import GraphSnapshot, normalized_adjacency_and_laplacian


NODE_TYPES = ("hashtag", "country", "sentiment", "topic", "platform")


@dataclass
class CHGDMOutput:
    node_states: torch.Tensor
    initial_states: torch.Tensor
    relation_states: Dict[str, torch.Tensor]
    relation_attention: Dict[str, torch.Tensor]
    laplacians: Dict[str, torch.Tensor]
    normalized_adjacencies: Dict[str, torch.Tensor]
    heat_kernels: Dict[str, torch.Tensor]
    gamma_shape: Dict[str, torch.Tensor]
    gamma_rate: Dict[str, torch.Tensor]


def _inv_softplus(x: float) -> float:
    return math.log(math.exp(x) - 1.0) if x > 1e-6 else -20.0


class CHGDM(nn.Module):
    """Causal Heterogeneous Graph Diffusion Module.

    The paper defines a Gamma-kernel expectation of the heat semigroup. The exact
    numerical conjugate-gradient approximation is not fully specified in the
    manuscript. This reference code therefore evaluates the *same integral*
    spectrally for the symmetric normalized Laplacian:

        E_tau[exp(-tau L)] = U diag((b / (b + lambda)) ** a) U^T,

    for tau ~ Gamma(shape=a, rate=b). This is exact up to eigensolver precision and
    keeps the module differentiable. It can be replaced with a CG approximation
    without changing the public interface.
    """

    def __init__(self, input_dim: int, hidden_dim: int, graph_config: GraphConfig) -> None:
        super().__init__()
        self.cfg = graph_config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.relations = list(graph_config.relation_names)
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.type_embedding = nn.Embedding(len(NODE_TYPES), hidden_dim)
        self.shape_raw = nn.ParameterDict({
            r: nn.Parameter(torch.tensor(_inv_softplus(graph_config.gamma_shape_init - graph_config.gamma_shape_min), dtype=torch.float32))
            for r in self.relations
        })
        self.rate_raw = nn.ParameterDict({
            r: nn.Parameter(torch.tensor(_inv_softplus(graph_config.gamma_rate_init - graph_config.gamma_rate_min), dtype=torch.float32))
            for r in self.relations
        })
        self.lifecycle_gamma = nn.ModuleDict({r: nn.Linear(1, 2) for r in self.relations})
        self.relation_score = nn.ModuleDict({
            r: nn.Sequential(nn.Linear(hidden_dim + 1, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
            for r in self.relations
        })
        self.gate = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid())
        self.output_norm = nn.LayerNorm(hidden_dim)

    def _type_ids(self, snapshot: GraphSnapshot, device: torch.device) -> torch.Tensor:
        mapping = {t: i for i, t in enumerate(NODE_TYPES)}
        return torch.tensor([mapping.get(t, 0) for t in snapshot.node_types], dtype=torch.long, device=device)

    def _gamma_parameters(self, relation: str, lifecycle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.lifecycle_gamma[relation](lifecycle.reshape(1, 1)).reshape(-1)
        shape = F.softplus(self.shape_raw[relation] + delta[0]) + self.cfg.gamma_shape_min
        rate = F.softplus(self.rate_raw[relation] + delta[1]) + self.cfg.gamma_rate_min
        return shape, rate

    @staticmethod
    def gamma_heat_operator(L: torch.Tensor, shape: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
        # L is symmetric PSD for the empirical graph described in the paper.
        evals, evecs = torch.linalg.eigh(0.5 * (L + L.T))
        evals = evals.clamp_min(0.0)
        response = torch.pow(rate / (rate + evals + 1e-8), shape)
        return (evecs * response.unsqueeze(0)) @ evecs.T

    def aggregate_from_relation_states(
        self,
        initial_states: torch.Tensor,
        relation_states: Mapping[str, torch.Tensor],
        relation_attention: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if not relation_states:
            return initial_states
        D = torch.zeros_like(initial_states)
        for r, state in relation_states.items():
            D = D + relation_attention[r] * state
        gate = self.gate(torch.cat([initial_states, D], dim=-1))
        return self.output_norm(gate * D + (1.0 - gate) * initial_states)

    def forward(self, snapshot: GraphSnapshot, lifecycle: torch.Tensor | float) -> CHGDMOutput:
        device = snapshot.node_features.device
        rho = lifecycle if isinstance(lifecycle, torch.Tensor) else torch.tensor(float(lifecycle), device=device)
        rho = rho.to(device=device, dtype=torch.float32).reshape(())

        x = snapshot.node_features.float()
        h0 = self.node_encoder(x) + self.type_embedding(self._type_ids(snapshot, device))
        pooled = h0.mean(dim=0)

        relation_states: Dict[str, torch.Tensor] = {}
        laplacians: Dict[str, torch.Tensor] = {}
        normalized: Dict[str, torch.Tensor] = {}
        heat_kernels: Dict[str, torch.Tensor] = {}
        shapes: Dict[str, torch.Tensor] = {}
        rates: Dict[str, torch.Tensor] = {}
        scores: Dict[str, torch.Tensor] = {}

        for relation, A in snapshot.adjacencies.items():
            if relation not in self.relation_score:
                continue
            Atilde, L = normalized_adjacency_and_laplacian(A.float())
            shape, rate = self._gamma_parameters(relation, rho)
            K_gamma = self.gamma_heat_operator(L, shape, rate)
            relation_states[relation] = K_gamma @ h0
            laplacians[relation] = L
            normalized[relation] = Atilde
            heat_kernels[relation] = torch.matrix_exp(-float(self.cfg.heat_tau_star) * L)
            shapes[relation] = shape
            rates[relation] = rate
            scores[relation] = self.relation_score[relation](torch.cat([pooled, rho.view(1)], dim=0)).reshape(())

        if not scores:
            raise ValueError("Graph snapshot contains no supported relation adjacency matrices.")
        names = list(scores)
        alpha_vec = torch.softmax(torch.stack([scores[r] for r in names]), dim=0)
        alphas = {r: alpha_vec[i] for i, r in enumerate(names)}
        h = self.aggregate_from_relation_states(h0, relation_states, alphas)
        return CHGDMOutput(
            node_states=h,
            initial_states=h0,
            relation_states=relation_states,
            relation_attention=alphas,
            laplacians=laplacians,
            normalized_adjacencies=normalized,
            heat_kernels=heat_kernels,
            gamma_shape=shapes,
            gamma_rate=rates,
        )
