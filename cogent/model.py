from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch
from torch import nn

from .chgdm import CHGDM, CHGDMOutput
from .config import ExperimentConfig
from .dataset import COGENTSample
from .gtfm import GTFM, GTFMOutput
from .tmft import TMFT, TMFTOutput


@dataclass
class COGENTOutput:
    forecast: GTFMOutput
    graph: CHGDMOutput
    fusion: TMFTOutput


class COGENTModel(nn.Module):
    def __init__(self, config: ExperimentConfig, graph_feature_dim: int, modality_dims: Mapping[str, int]) -> None:
        super().__init__()
        self.config = config
        mc = config.model
        self.chgdm = CHGDM(graph_feature_dim, mc.graph_hidden_dim, config.graph)
        self.tmft = TMFT(
            modality_dims=modality_dims,
            graph_hidden_dim=mc.graph_hidden_dim,
            d_model=mc.hidden_dim,
            n_heads=mc.transformer_heads,
            n_layers=mc.tmft_layers,
            ff_multiplier=mc.ff_multiplier,
            dropout=mc.dropout,
            eta_init=mc.diffusion_bias_eta_init,
        )
        self.gtfm = GTFM(
            graph_hidden_dim=mc.graph_hidden_dim,
            context_dim=mc.hidden_dim,
            latent_dim=mc.latent_dim,
            horizon=config.data.max_horizon,
            observation_noise_floor=mc.observation_noise_floor,
            emergence_head=mc.emergence_head,
        )

    def _downstream(
        self,
        sample: COGENTSample,
        graph_out: CHGDMOutput,
        n_samples: int,
        noise: torch.Tensor | None = None,
        observation_noise: torch.Tensor | None = None,
    ) -> COGENTOutput:
        target_state = graph_out.node_states[sample.target_node_index]
        fusion = self.tmft(
            modalities=sample.modalities,
            lifecycle=sample.lifecycle,
            graph_target_state=target_state,
            heat_kernels=graph_out.heat_kernels,
            relation_attention=graph_out.relation_attention,
            modality_node_indices=sample.modality_node_indices,
        )
        forecast = self.gtfm(
            graph_node_states=graph_out.node_states,
            context=fusion.fused,
            laplacians=graph_out.laplacians,
            relation_attention=graph_out.relation_attention,
            target_idx=sample.target_node_index,
            n_samples=n_samples,
            noise=noise,
            observation_noise=observation_noise,
        )
        return COGENTOutput(forecast=forecast, graph=graph_out, fusion=fusion)

    def forward(
        self,
        sample: COGENTSample,
        n_samples: int | None = None,
        deterministic: bool = False,
        noise: torch.Tensor | None = None,
        observation_noise: torch.Tensor | None = None,
    ) -> COGENTOutput:
        K = int(n_samples or (self.config.model.monte_carlo_samples_train if self.training else self.config.model.monte_carlo_samples_eval))
        graph_out = self.chgdm(sample.graph, sample.lifecycle[-1])
        if deterministic:
            N = sample.graph.num_nodes
            noise = torch.zeros((K, self.config.data.max_horizon, N, self.config.model.latent_dim), device=sample.graph.node_features.device)
            observation_noise = torch.zeros((K, self.config.data.max_horizon), device=sample.graph.node_features.device)
        return self._downstream(sample, graph_out, K, noise=noise, observation_noise=observation_noise)

    def forward_with_relation_states(
        self,
        sample: COGENTSample,
        base_graph_out: CHGDMOutput,
        relation_states: Mapping[str, torch.Tensor],
        n_samples: int = 1,
        deterministic: bool = True,
    ) -> COGENTOutput:
        node_states = self.chgdm.aggregate_from_relation_states(
            base_graph_out.initial_states,
            relation_states,
            base_graph_out.relation_attention,
        )
        graph_out = CHGDMOutput(
            node_states=node_states,
            initial_states=base_graph_out.initial_states,
            relation_states=dict(relation_states),
            relation_attention=base_graph_out.relation_attention,
            laplacians=base_graph_out.laplacians,
            normalized_adjacencies=base_graph_out.normalized_adjacencies,
            heat_kernels=base_graph_out.heat_kernels,
            gamma_shape=base_graph_out.gamma_shape,
            gamma_rate=base_graph_out.gamma_rate,
        )
        if deterministic:
            N = sample.graph.num_nodes
            noise = torch.zeros((n_samples, self.config.data.max_horizon, N, self.config.model.latent_dim), device=node_states.device)
            obs_noise = torch.zeros((n_samples, self.config.data.max_horizon), device=node_states.device)
        else:
            noise = None
            obs_noise = None
        return self._downstream(sample, graph_out, n_samples, noise=noise, observation_noise=obs_noise)
