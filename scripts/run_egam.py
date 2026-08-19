#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from cogent.config import ExperimentConfig
from cogent.dataset import COGENTWindowDataset
from cogent.egam import EGAM
from cogent.graph import EmpiricalGraphBuilder
from cogent.io import load_prepared_directory
from cogent.model import COGENTModel
from cogent.semantic import load_embedding_map
from cogent.utils import resolve_device


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate EGAM relation/modality/time attributions.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--horizon-index", type=int, default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ExperimentConfig.from_dict(ckpt["config"])
    train, _, test, state = load_prepared_directory(args.prepared)
    emb = load_embedding_map(args.embeddings)
    builder = EmpiricalGraphBuilder(cfg.data, cfg.graph, emb, random_state=cfg.training.seed)
    builder.fit_topics(train[cfg.data.hashtag_col].unique())
    ds = COGENTWindowDataset(test, cfg.data, cfg.graph, builder, emb, state["country_vocab"], state["platform_vocab"])
    model = COGENTModel(cfg, ckpt["graph_feature_dim"], ckpt["modality_dims"])
    model.load_state_dict(ckpt["model_state"])
    device = resolve_device(cfg.training.device)
    model.to(device)
    sample = ds[args.sample_index].to(device)
    exp = EGAM(model, steps=args.steps).explain(sample, args.horizon_index)
    payload = {
        "hashtag": sample.hashtag,
        "origin_date": str(sample.origin_date),
        "relation_importance": exp.relation_importance,
        "temporal_importance": exp.temporal_importance.detach().cpu().tolist(),
        "feature_importance": {k: v.detach().cpu().tolist() for k, v in exp.feature_importance.items()},
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
