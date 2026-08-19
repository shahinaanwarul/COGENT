#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from cogent.config import ExperimentConfig
from cogent.dataset import COGENTWindowDataset
from cogent.experiments import evaluate_graph_ablation
from cogent.graph import EmpiricalGraphBuilder
from cogent.io import load_prepared_directory
from cogent.model import COGENTModel
from cogent.semantic import load_embedding_map
from cogent.utils import resolve_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ExperimentConfig.from_dict(ckpt["config"])
    train, _, test, state = load_prepared_directory(args.prepared)
    emb = load_embedding_map(args.embeddings)
    builder = EmpiricalGraphBuilder(cfg.data, cfg.graph, emb, random_state=cfg.training.seed)
    builder.fit_topics(train[cfg.data.hashtag_col].astype(str).unique())
    ds = COGENTWindowDataset(test, cfg.data, cfg.graph, builder, emb, state["country_vocab"], state["platform_vocab"])
    model = COGENTModel(cfg, ckpt["graph_feature_dim"], ckpt["modality_dims"])
    model.load_state_dict(ckpt["model_state"])
    device = resolve_device(cfg.training.device)
    model.to(device)
    conditions = ["identity", "endpoint_shuffle", "without_hashtag_similarity", "without_country", "without_sentiment", "without_topic"]
    results = {c: evaluate_graph_ablation(model, ds, c, device, max_samples=args.max_samples) for c in conditions}
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
