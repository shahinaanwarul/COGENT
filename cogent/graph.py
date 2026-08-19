from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import copy

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from .config import DataConfig, GraphConfig


@dataclass
class GraphSnapshot:
    date: pd.Timestamp
    node_names: List[str]
    node_types: List[str]
    node_features: torch.Tensor
    adjacencies: Dict[str, torch.Tensor]
    hashtag_to_index: Dict[str, int]
    country_to_index: Dict[str, int]
    sentiment_to_index: Dict[str, int]
    topic_to_index: Dict[int, int]
    platform_to_index: Dict[str, int]

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)

    def clone(self) -> "GraphSnapshot":
        return GraphSnapshot(
            date=self.date,
            node_names=list(self.node_names),
            node_types=list(self.node_types),
            node_features=self.node_features.clone(),
            adjacencies={k: v.clone() for k, v in self.adjacencies.items()},
            hashtag_to_index=dict(self.hashtag_to_index),
            country_to_index=dict(self.country_to_index),
            sentiment_to_index=dict(self.sentiment_to_index),
            topic_to_index=dict(self.topic_to_index),
            platform_to_index=dict(self.platform_to_index),
        )

    def to(self, device: torch.device | str) -> "GraphSnapshot":
        return GraphSnapshot(
            date=self.date,
            node_names=self.node_names,
            node_types=self.node_types,
            node_features=self.node_features.to(device),
            adjacencies={k: v.to(device) for k, v in self.adjacencies.items()},
            hashtag_to_index=self.hashtag_to_index,
            country_to_index=self.country_to_index,
            sentiment_to_index=self.sentiment_to_index,
            topic_to_index=self.topic_to_index,
            platform_to_index=self.platform_to_index,
        )


def normalized_adjacency_and_laplacian(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = A.shape[0]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    S = 0.5 * (A + A.transpose(-1, -2)) + I
    deg = S.sum(dim=-1).clamp_min(1e-8)
    inv = deg.rsqrt()
    Atilde = inv[:, None] * S * inv[None, :]
    L = I - Atilde
    return Atilde, L


def project_adjacency(A: torch.Tensor, max_weight: float | None = None) -> torch.Tensor:
    """Projection Pi used for bounded structural scenarios.

    The manuscript requires projection back to an admissible normalized graph but
    does not uniquely define Pi. This reference projection enforces non-negativity,
    symmetry, zero raw self-edges, and an optional weight cap. Normalization is then
    performed by normalized_adjacency_and_laplacian().
    """
    out = 0.5 * (A + A.T)
    out = out.clamp_min(0.0)
    if max_weight is not None:
        out = out.clamp_max(float(max_weight))
    out = out.clone()
    out.fill_diagonal_(0.0)
    return out


class EmpiricalGraphBuilder:
    def __init__(
        self,
        data_config: DataConfig,
        graph_config: GraphConfig,
        embedding_map: Mapping[str, np.ndarray],
        random_state: int = 42,
    ) -> None:
        self.d = data_config
        self.g = graph_config
        self.embedding_map = {str(k): np.asarray(v, dtype=np.float32) for k, v in embedding_map.items()}
        self.random_state = random_state
        self.topic_model: KMeans | None = None
        self.topic_assignment: Dict[str, int] = {}
        self.feature_dim = self.g.lookback_days * 8

    def fit_topics(self, training_hashtags: Sequence[str]) -> None:
        tags = [h for h in sorted(set(map(str, training_hashtags))) if h in self.embedding_map]
        if not tags:
            raise ValueError("No training hashtags have semantic embeddings.")
        X = np.stack([self.embedding_map[h] for h in tags])
        n_clusters = min(self.g.topic_clusters, len(tags))
        self.topic_model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10).fit(X)
        self.topic_assignment = {h: int(z) for h, z in zip(tags, self.topic_model.labels_)}

    def assign_topic(self, hashtag: str) -> int:
        if self.topic_model is None:
            raise RuntimeError("Call fit_topics() using training hashtags before building graphs.")
        if hashtag in self.topic_assignment:
            return self.topic_assignment[hashtag]
        if hashtag not in self.embedding_map:
            return 0
        z = int(self.topic_model.predict(self.embedding_map[hashtag][None, :])[0])
        self.topic_assignment[hashtag] = z
        return z

    def sentiment_bin(self, value: float) -> str:
        if value < self.g.sentiment_negative_cut:
            return "negative"
        if value > self.g.sentiment_positive_cut:
            return "positive"
        return "neutral"

    def _history_vector(self, h: pd.DataFrame, origin: pd.Timestamp) -> np.ndarray:
        c = self.d
        L = self.g.lookback_days
        hist = h[h[c.date_col] <= origin].sort_values(c.date_col).tail(L)
        rows: List[np.ndarray] = []
        for _, row in hist.iterrows():
            vals = np.array(
                [
                    row.get(f"scaled_{c.mentions_col}", 0.0),
                    row.get(f"scaled_{c.reach_col}", 0.0),
                    row.get(f"scaled_{c.sentiment_col}", 0.0),
                    row.get("delta_mentions", 0.0),
                    row.get("delta_reach", 0.0),
                    row.get(f"mask_{c.mentions_col}", 0.0),
                    row.get(f"mask_{c.reach_col}", 0.0),
                    row.get(f"mask_{c.sentiment_col}", 0.0),
                ],
                dtype=np.float32,
            )
            rows.append(np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0))
        if len(rows) < L:
            pad = [np.zeros(8, dtype=np.float32) for _ in range(L - len(rows))]
            # Missingness masks in padded history are marked unavailable.
            for p in pad:
                p[5:] = 1.0
            rows = pad + rows
        return np.concatenate(rows, axis=0).astype(np.float32)

    def build(self, frame: pd.DataFrame, origin_date: pd.Timestamp | str) -> GraphSnapshot:
        c, g = self.d, self.g
        origin = pd.Timestamp(origin_date)
        if self.topic_model is None:
            raise RuntimeError("fit_topics() must be called before build().")

        current = frame[frame[c.date_col] == origin].copy()
        if current.empty:
            raise ValueError(f"No records exist at graph origin {origin.date()}.")
        if "long_gap" in current:
            current = current[current["long_gap"] == 0]
        current = current.drop_duplicates(subset=[c.hashtag_col], keep="last")
        hashtags = sorted(current[c.hashtag_col].astype(str).tolist())
        if not hashtags:
            raise ValueError("No valid active hashtags at this graph origin.")

        histories = {h: frame[frame[c.hashtag_col].astype(str) == h] for h in hashtags}
        hashtag_features = {h: self._history_vector(histories[h], origin) for h in hashtags}
        row_by_hash = {str(r[c.hashtag_col]): r for _, r in current.iterrows()}

        countries = sorted({str(row_by_hash[h].get(c.country_col, "Unknown")) for h in hashtags})
        sentiments = ["negative", "neutral", "positive"]
        topics = sorted({self.assign_topic(h) for h in hashtags})
        platforms: List[str] = []
        if g.add_platform_relation_if_available and c.platform_col in current:
            platforms = sorted({str(row_by_hash[h].get(c.platform_col, "Unknown")) for h in hashtags if str(row_by_hash[h].get(c.platform_col, "Unknown")) not in {"Unknown", "nan", "None", ""}})

        node_names: List[str] = []
        node_types: List[str] = []
        hashtag_to_index: Dict[str, int] = {}
        country_to_index: Dict[str, int] = {}
        sentiment_to_index: Dict[str, int] = {}
        topic_to_index: Dict[int, int] = {}
        platform_to_index: Dict[str, int] = {}

        def add(name: str, typ: str) -> int:
            idx = len(node_names)
            node_names.append(name)
            node_types.append(typ)
            return idx

        for h in hashtags:
            hashtag_to_index[h] = add(f"hashtag:{h}", "hashtag")
        for x in countries:
            country_to_index[x] = add(f"country:{x}", "country")
        for x in sentiments:
            sentiment_to_index[x] = add(f"sentiment:{x}", "sentiment")
        for x in topics:
            topic_to_index[x] = add(f"topic:{x}", "topic")
        for x in platforms:
            platform_to_index[x] = add(f"platform:{x}", "platform")

        n = len(node_names)
        node_features = np.zeros((n, self.feature_dim), dtype=np.float32)
        for h, idx in hashtag_to_index.items():
            node_features[idx] = hashtag_features[h]

        adjs: Dict[str, np.ndarray] = {r: np.zeros((n, n), dtype=np.float32) for r in ("sim", "geo", "sent", "topic")}
        if platforms:
            adjs["plat"] = np.zeros((n, n), dtype=np.float32)

        # Hashtag similarity: positive cosine, symmetrized kNN union.
        X = np.stack([hashtag_features[h] for h in hashtags])
        sim = cosine_similarity(X)
        np.fill_diagonal(sim, -np.inf)
        directed = np.zeros_like(sim, dtype=bool)
        for i in range(len(hashtags)):
            k = min(g.knn_k, max(0, len(hashtags) - 1))
            if k:
                nn = np.argpartition(-sim[i], kth=k - 1)[:k]
                directed[i, nn] = True
        union = directed | directed.T
        for i, hi in enumerate(hashtags):
            for j, hj in enumerate(hashtags):
                if i != j and union[i, j]:
                    w = max(float(sim[i, j]), 0.0)
                    adjs["sim"][hashtag_to_index[hi], hashtag_to_index[hj]] = w
        adjs["sim"] = 0.5 * (adjs["sim"] + adjs["sim"].T)

        for h in hashtags:
            row = row_by_hash[h]
            hi = hashtag_to_index[h]
            country = str(row.get(c.country_col, "Unknown"))
            ci = country_to_index[country]
            m = max(float(row.get(f"scaled_{c.mentions_col}", 0.0)), 0.0)
            q = max(float(row.get(f"scaled_{c.reach_col}", 0.0)), 0.0)
            # The manuscript uses sqrt(m*q) with normalized non-negative values.
            # Standardized features may be negative, so the reference builder clips
            # them at zero for this non-negative adjacency weight.
            w_geo = float(np.sqrt(m * q))
            adjs["geo"][hi, ci] = adjs["geo"][ci, hi] = w_geo

            s_raw = float(row.get(c.sentiment_col, 0.0))
            si = sentiment_to_index[self.sentiment_bin(s_raw)]
            adjs["sent"][hi, si] = adjs["sent"][si, hi] = 1.0

            topic = self.assign_topic(h)
            ti = topic_to_index[topic]
            adjs["topic"][hi, ti] = adjs["topic"][ti, hi] = 1.0

            if "plat" in adjs:
                p = str(row.get(c.platform_col, "Unknown"))
                if p in platform_to_index:
                    pi = platform_to_index[p]
                    adjs["plat"][hi, pi] = adjs["plat"][pi, hi] = 1.0

        # Attribute-node features are masked means of incident hashtag features.
        for relation, A in adjs.items():
            for j, typ in enumerate(node_types):
                if typ == "hashtag":
                    continue
                incident = [i for i, t in enumerate(node_types) if t == "hashtag" and A[i, j] > 0]
                if incident:
                    node_features[j] = node_features[incident].mean(axis=0)

        return GraphSnapshot(
            date=origin,
            node_names=node_names,
            node_types=node_types,
            node_features=torch.from_numpy(node_features),
            adjacencies={k: torch.from_numpy(v) for k, v in adjs.items()},
            hashtag_to_index=hashtag_to_index,
            country_to_index=country_to_index,
            sentiment_to_index=sentiment_to_index,
            topic_to_index=topic_to_index,
            platform_to_index=platform_to_index,
        )


def ablate_graph(snapshot: GraphSnapshot, condition: str, seed: int = 42) -> GraphSnapshot:
    """Graph-topology controls described in the manuscript."""
    g = snapshot.clone()
    if condition in {"identity", "no_topology"}:
        for r in g.adjacencies:
            g.adjacencies[r].zero_()
        return g
    if condition.startswith("without_"):
        relation = condition.replace("without_", "")
        aliases = {"hashtag_similarity": "sim", "country": "geo", "sentiment": "sent", "topic": "topic", "platform": "plat"}
        relation = aliases.get(relation, relation)
        if relation in g.adjacencies:
            g.adjacencies[relation].zero_()
        return g
    if condition == "endpoint_shuffle":
        rng = np.random.default_rng(seed)
        for r, A_t in list(g.adjacencies.items()):
            A = A_t.cpu().numpy().copy()
            upper = np.argwhere(np.triu(A, 1) > 0)
            weights = np.array([A[i, j] for i, j in upper], dtype=np.float32)
            out = np.zeros_like(A)
            if len(upper):
                endpoints = upper.copy()
                rng.shuffle(endpoints[:, 1])
                for (i, j), w in zip(endpoints, weights):
                    if i != j:
                        out[i, j] = out[j, i] = w
            g.adjacencies[r] = torch.from_numpy(out).to(A_t.device)
        return g
    raise ValueError(f"Unknown graph ablation condition: {condition}")
