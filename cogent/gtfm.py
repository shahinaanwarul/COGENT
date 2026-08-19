from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GTFMOutput:
    trajectory_samples: torch.Tensor  # [K,H]
    emergence_probability_samples: torch.Tensor  # [K,H]
    mean_trajectory: torch.Tensor  # [H]
    mean_emergence_probability: torch.Tensor  # [H]
    latent_samples: torch.Tensor  # [K,H,N,D]
    weighted_laplacian: torch.Tensor
    observation_scale: torch.Tensor
    process_scale: torch.Tensor
    driving_noise: torch.Tensor


class GTFM(nn.Module):
    """Graph-driven latent SDE forecaster using Euler-Maruyama simulation.

    The manuscript defines the lifecycle SDE but does not specify a numerical SDE
    solver or step size. This reference implementation uses one Euler-Maruyama step
    per forecast horizon step with total lifecycle length normalized to one.
    """

    def __init__(
        self,
        graph_hidden_dim: int,
        context_dim: int,
        latent_dim: int,
        horizon: int,
        observation_noise_floor: float = 1e-4,
        emergence_head: bool = True,
    ) -> None:
        super().__init__()
        self.graph_hidden_dim = graph_hidden_dim
        self.context_dim = context_dim
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.observation_noise_floor = observation_noise_floor
        self.emergence_head_enabled = emergence_head

        self.node_init = nn.Linear(graph_hidden_dim, latent_dim)
        self.context_init = nn.Linear(context_dim, latent_dim)
        self.node_control = nn.Linear(graph_hidden_dim, latent_dim)
        self.context_control = nn.Sequential(nn.Linear(context_dim, latent_dim), nn.Tanh(), nn.Linear(latent_dim, latent_dim))
        self.process_scale_head = nn.Sequential(nn.Linear(context_dim, latent_dim), nn.Softplus())
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + context_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 1),
        )
        self.observation_scale_head = nn.Sequential(nn.Linear(context_dim, latent_dim // 2), nn.GELU(), nn.Linear(latent_dim // 2, 1), nn.Softplus())
        if emergence_head:
            self.emergence_head = nn.Sequential(
                nn.Linear(latent_dim + context_dim, latent_dim // 2),
                nn.GELU(),
                nn.Linear(latent_dim // 2, 1),
            )

    @staticmethod
    def weighted_laplacian(laplacians: Mapping[str, torch.Tensor], relation_attention: Mapping[str, torch.Tensor]) -> torch.Tensor:
        names = [r for r in laplacians if r in relation_attention]
        if not names:
            raise ValueError("No laplacian/attention pairs were supplied.")
        L = torch.zeros_like(laplacians[names[0]])
        denom = torch.zeros((), device=L.device)
        for r in names:
            L = L + relation_attention[r] * laplacians[r]
            denom = denom + relation_attention[r]
        return L / denom.clamp_min(1e-8)

    def _decode(self, latent: torch.Tensor, context: torch.Tensor, target_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # latent [K,N,D]
        target = latent[:, target_idx, :]
        context_k = context.unsqueeze(0).expand(target.shape[0], -1)
        z = torch.cat([target, context_k], dim=-1)
        mean = self.decoder(z).squeeze(-1)
        if self.emergence_head_enabled:
            prob = torch.sigmoid(self.emergence_head(z).squeeze(-1))
        else:
            prob = torch.sigmoid(mean)
        return mean, prob

    def forward(
        self,
        graph_node_states: torch.Tensor,
        context: torch.Tensor,
        laplacians: Mapping[str, torch.Tensor],
        relation_attention: Mapping[str, torch.Tensor],
        target_idx: int,
        n_samples: int,
        noise: torch.Tensor | None = None,
        observation_noise: torch.Tensor | None = None,
    ) -> GTFMOutput:
        device = graph_node_states.device
        N = graph_node_states.shape[0]
        H = self.horizon
        K = int(n_samples)
        dt = 1.0 / max(H, 1)
        sqrt_dt = math.sqrt(dt)

        Lbar = self.weighted_laplacian(laplacians, relation_attention)
        base_state = self.node_init(graph_node_states) + self.context_init(context).unsqueeze(0)
        latent = base_state.unsqueeze(0).expand(K, -1, -1).clone()
        node_control = self.node_control(graph_node_states)
        context_control = self.context_control(context).unsqueeze(0)
        control = node_control + context_control
        process_scale = self.process_scale_head(context).clamp_min(1e-5)
        obs_scale = self.observation_scale_head(context).reshape(()) + self.observation_noise_floor

        if noise is None:
            noise = torch.randn((K, H, N, self.latent_dim), dtype=latent.dtype, device=device)
        else:
            if noise.shape != (K, H, N, self.latent_dim):
                raise ValueError(f"noise must have shape {(K, H, N, self.latent_dim)}, got {tuple(noise.shape)}")
            noise = noise.to(device=device, dtype=latent.dtype)

        if observation_noise is None:
            observation_noise = torch.randn((K, H), dtype=latent.dtype, device=device)
        else:
            if observation_noise.shape != (K, H):
                raise ValueError(f"observation_noise must have shape {(K, H)}, got {tuple(observation_noise.shape)}")
            observation_noise = observation_noise.to(device=device, dtype=latent.dtype)

        ys = []
        ps = []
        states = []
        for step in range(H):
            graph_drift = -torch.einsum("nm,kmd->knd", Lbar, latent)
            drift = graph_drift + control.unsqueeze(0)
            diffusion = process_scale.view(1, 1, -1) * noise[:, step]
            latent = latent + dt * drift + sqrt_dt * diffusion
            mean, prob = self._decode(latent, context, target_idx)
            # Observation noise makes the predictive output distribution explicit.
            y = mean + obs_scale * observation_noise[:, step]
            ys.append(y)
            ps.append(prob)
            states.append(latent)

        trajectory = torch.stack(ys, dim=1)
        probs = torch.stack(ps, dim=1)
        latent_samples = torch.stack(states, dim=1)
        return GTFMOutput(
            trajectory_samples=trajectory,
            emergence_probability_samples=probs,
            mean_trajectory=trajectory.mean(dim=0),
            mean_emergence_probability=probs.mean(dim=0),
            latent_samples=latent_samples,
            weighted_laplacian=Lbar,
            observation_scale=obs_scale,
            process_scale=process_scale,
            driving_noise=noise,
        )
