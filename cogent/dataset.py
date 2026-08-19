from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import DataConfig, GraphConfig
from .data import WindowRecord, build_modalities, iter_windows
from .graph import EmpiricalGraphBuilder, GraphSnapshot


@dataclass
class COGENTSample:
    hashtag: str
    origin_date: pd.Timestamp
    modalities: Dict[str, torch.Tensor]
    lifecycle: torch.Tensor
    target: torch.Tensor
    emergence_target: torch.Tensor | None
    graph: GraphSnapshot
    target_node_index: int
    modality_node_indices: Dict[str, int]

    def to(self, device: torch.device | str) -> "COGENTSample":
        return COGENTSample(
            hashtag=self.hashtag,
            origin_date=self.origin_date,
            modalities={k: v.to(device) for k, v in self.modalities.items()},
            lifecycle=self.lifecycle.to(device),
            target=self.target.to(device),
            emergence_target=None if self.emergence_target is None else self.emergence_target.to(device),
            graph=self.graph.to(device),
            target_node_index=self.target_node_index,
            modality_node_indices=self.modality_node_indices,
        )


class COGENTWindowDataset(Dataset):
    """Sample-wise dataset for variable graph snapshots.

    Graph snapshots are cached by forecast origin. Training operates sample-wise
    (or with gradient accumulation) because the typed graph can vary in size across
    origins. This keeps the implementation dependency-free from graph batching
    libraries while preserving the manuscript's daily dynamic graph semantics.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        data_config: DataConfig,
        graph_config: GraphConfig,
        graph_builder: EmpiricalGraphBuilder,
        embedding_map: Mapping[str, np.ndarray],
        country_vocab: Mapping[str, int],
        platform_vocab: Mapping[str, int],
        horizon: int | None = None,
    ) -> None:
        self.frame = frame.copy()
        self.d = data_config
        self.g = graph_config
        self.builder = graph_builder
        self.embedding_map = embedding_map
        self.country_vocab = country_vocab
        self.platform_vocab = platform_vocab
        self.horizon = int(horizon or data_config.max_horizon)
        self.records: List[WindowRecord] = list(iter_windows(self.frame, self.d, self.g.lookback_days, self.horizon))
        self.graph_cache: Dict[pd.Timestamp, GraphSnapshot] = {}
        self.rank_lookup: Dict[tuple[str, pd.Timestamp], tuple[int, int]] = {}
        for hashtag, group in self.frame.groupby(self.d.hashtag_col, sort=False):
            g = group.sort_values(self.d.date_col).reset_index(drop=True)
            n = len(g)
            for i, row in g.iterrows():
                self.rank_lookup[(str(hashtag), pd.Timestamp(row[self.d.date_col]))] = (i, n)

    def __len__(self) -> int:
        return len(self.records)

    def modality_dims(self) -> Dict[str, int]:
        if not self.records:
            raise ValueError("Dataset has no valid windows.")
        sample = self[0]
        return {k: int(v.shape[-1]) for k, v in sample.modalities.items()}

    def _target_column(self) -> str:
        c = self.d
        scaled = f"scaled_{c.target_col}"
        return scaled if scaled in self.frame.columns else c.target_col

    def _graph(self, date: pd.Timestamp) -> GraphSnapshot:
        date = pd.Timestamp(date)
        if date not in self.graph_cache:
            self.graph_cache[date] = self.builder.build(self.frame, date)
        return self.graph_cache[date]

    def __getitem__(self, idx: int) -> COGENTSample:
        record = self.records[idx]
        if record.hashtag not in self.embedding_map:
            raise KeyError(f"Missing semantic embedding for hashtag {record.hashtag!r}")
        graph = self._graph(record.origin_date)
        if record.hashtag not in graph.hashtag_to_index:
            raise KeyError(f"Hashtag {record.hashtag!r} is not active in graph at {record.origin_date}")
        target_idx = graph.hashtag_to_index[record.hashtag]
        modalities_np = build_modalities(
            record,
            self.d,
            self.country_vocab,
            self.platform_vocab,
            self.embedding_map[record.hashtag],
        )
        modalities = {k: torch.from_numpy(v).float() for k, v in modalities_np.items()}

        rank, n = self.rank_lookup[(record.hashtag, record.origin_date)]
        start = max(0, rank - self.g.lookback_days + 1)
        denom = max(n - 1, 1)
        lifecycle = torch.linspace(start / denom, rank / denom, self.g.lookback_days, dtype=torch.float32)

        target_col = self._target_column()
        target = torch.tensor(record.future[target_col].to_numpy(dtype=np.float32), dtype=torch.float32)
        emergence = None
        if "Emerging" in record.future:
            emergence = torch.tensor(record.future["Emerging"].to_numpy(dtype=np.float32), dtype=torch.float32)

        last = record.history.iloc[-1]
        c = self.d
        country = str(last.get(c.country_col, "Unknown"))
        platform = str(last.get(c.platform_col, "Unknown"))
        sentiment_value = float(last.get(c.sentiment_col, 0.0))
        sentiment_bin = self.builder.sentiment_bin(sentiment_value)
        topic = self.builder.assign_topic(record.hashtag)
        modality_nodes = {
            "default": target_idx,
            "engagement": target_idx,
            "missingness": target_idx,
            "calendar": target_idx,
            "semantic": target_idx,
            "geographic": graph.country_to_index.get(country, target_idx),
            "sentiment": graph.sentiment_to_index.get(sentiment_bin, target_idx),
            "platform": graph.platform_to_index.get(platform, target_idx),
            "topic": graph.topic_to_index.get(topic, target_idx),
        }
        return COGENTSample(
            hashtag=record.hashtag,
            origin_date=record.origin_date,
            modalities=modalities,
            lifecycle=lifecycle,
            target=target,
            emergence_target=emergence,
            graph=graph,
            target_node_index=target_idx,
            modality_node_indices=modality_nodes,
        )
