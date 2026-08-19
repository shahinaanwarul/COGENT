from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TMFTOutput:
    fused: torch.Tensor
    tokens: torch.Tensor
    diffusion_bias: torch.Tensor
    temporal_weights: torch.Tensor
    attention_maps: List[torch.Tensor]
    token_modalities: List[str]
    token_times: torch.Tensor


class DiffusionBiasedMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [T,D], bias: [T,T]
        T, D = x.shape
        qkv = self.qkv(x).reshape(T, 3, self.n_heads, self.head_dim).permute(1, 2, 0, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [H,T,d]
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        logits = logits + bias.unsqueeze(0)
        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(0, 1).reshape(T, D)
        return self.out(out), attn


class DiffusionTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = DiffusionBiasedMultiheadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_multiplier * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_multiplier * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y, attn = self.attn(self.norm1(x), bias)
        x = x + self.drop(y)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, attn


class TMFT(nn.Module):
    """Temporal Multimodal Fusion Transformer with diffusion-biased attention."""

    def __init__(
        self,
        modality_dims: Mapping[str, int],
        graph_hidden_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_multiplier: int = 4,
        dropout: float = 0.2,
        eta_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.modality_names = list(modality_dims)
        self.projectors = nn.ModuleDict({m: nn.Linear(int(dim), d_model) for m, dim in modality_dims.items()})
        self.modality_embedding = nn.Embedding(len(self.modality_names), d_model)
        self.phase_encoder = nn.Sequential(nn.Linear(1, d_model), nn.Tanh(), nn.Linear(d_model, d_model))
        self.graph_projector = nn.Linear(graph_hidden_dim, d_model)
        self.layers = nn.ModuleList(
            [DiffusionTransformerLayer(d_model, n_heads, ff_multiplier, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.pool_score = nn.Sequential(nn.Linear(2 * d_model + 1, d_model), nn.Tanh(), nn.Linear(d_model, 1))
        self.pool_gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        self.eta_raw = nn.Parameter(torch.tensor(float(eta_init)))
        self.d_model = d_model

    def _build_bias(
        self,
        heat_kernels: Mapping[str, torch.Tensor],
        relation_attention: Mapping[str, torch.Tensor],
        node_indices: torch.Tensor,
        token_rho: torch.Tensor,
    ) -> torch.Tensor:
        T = node_indices.shape[0]
        device = token_rho.device
        B = torch.zeros((T, T), dtype=torch.float32, device=device)
        for r, Phi in heat_kernels.items():
            if r not in relation_attention:
                continue
            idx_i = node_indices[:, None].expand(T, T)
            idx_j = node_indices[None, :].expand(T, T)
            B = B + relation_attention[r] * Phi[idx_i, idx_j]
        phase = torch.exp(-torch.abs(token_rho[:, None] - token_rho[None, :]))
        B = B * phase
        # Proposition II assumes a bounded diffusion bias; normalize if necessary.
        scale = B.abs().amax().clamp_min(1.0)
        return B / scale

    def forward(
        self,
        modalities: Mapping[str, torch.Tensor],
        lifecycle: torch.Tensor,
        graph_target_state: torch.Tensor,
        heat_kernels: Mapping[str, torch.Tensor],
        relation_attention: Mapping[str, torch.Tensor],
        modality_node_indices: Mapping[str, int],
    ) -> TMFTOutput:
        L = int(lifecycle.shape[0])
        tokens: List[torch.Tensor] = []
        token_modalities: List[str] = []
        token_times: List[int] = []
        node_indices: List[int] = []
        rho_tokens: List[torch.Tensor] = []
        graph_token = self.graph_projector(graph_target_state)

        for m_idx, name in enumerate(self.modality_names):
            if name not in modalities:
                continue
            x = modalities[name]
            if x.ndim != 2 or x.shape[0] != L:
                raise ValueError(f"Modality {name!r} must have shape [lookback, features].")
            p = self.projectors[name](x.float())
            p = p + self.modality_embedding.weight[m_idx] + self.phase_encoder(lifecycle[:, None].float()) + graph_token
            for t in range(L):
                tokens.append(p[t])
                token_modalities.append(name)
                token_times.append(t)
                node_indices.append(int(modality_node_indices.get(name, modality_node_indices.get("default", 0))))
                rho_tokens.append(lifecycle[t])

        z = torch.stack(tokens, dim=0)
        node_idx_t = torch.tensor(node_indices, dtype=torch.long, device=z.device)
        rho_t = torch.stack(rho_tokens).float()
        base_bias = self._build_bias(heat_kernels, relation_attention, node_idx_t, rho_t)
        eta = F.softplus(self.eta_raw)
        bias = eta * base_bias

        attention_maps: List[torch.Tensor] = []
        x = z
        for layer in self.layers:
            x, attn = layer(x, bias)
            attention_maps.append(attn)
        x = self.final_norm(x)

        # Convert token stack to one vector per time step before gated temporal pooling.
        time_repr: List[torch.Tensor] = []
        for t in range(L):
            ids = [i for i, tt in enumerate(token_times) if tt == t]
            time_repr.append(x[ids].mean(dim=0))
        time_repr_t = torch.stack(time_repr, dim=0)
        graph_repeated = graph_token.unsqueeze(0).expand(L, -1)
        scores = self.pool_score(torch.cat([time_repr_t, graph_repeated, lifecycle[:, None].float()], dim=-1)).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        pooled = torch.sum(weights[:, None] * time_repr_t, dim=0)
        gate = self.pool_gate(torch.cat([pooled, graph_token], dim=-1))
        fused = gate * pooled + (1.0 - gate) * graph_token
        return TMFTOutput(
            fused=fused,
            tokens=x,
            diffusion_bias=base_bias,
            temporal_weights=weights,
            attention_maps=attention_maps,
            token_modalities=token_modalities,
            token_times=torch.tensor(token_times, dtype=torch.long, device=x.device),
        )
