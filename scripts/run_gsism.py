#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from cogent.config import ExperimentConfig
from cogent.dataset import COGENTWindowDataset
from cogent.graph import EmpiricalGraphBuilder
from cogent.gsism import GSISM, StructuralScenario
from cogent.io import load_prepared_directory
from cogent.model import COGENTModel
from cogent.semantic import load_embedding_map
from cogent.utils import resolve_device


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one bounded GSISM graph perturbation.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--relation", required=True, choices=["sim", "geo", "sent", "topic", "plat"])
    ap.add_argument("--strength", type=float, default=0.20)
    ap.add_argument("--sample-index", type=int, default=0)
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
    scenario = StructuralScenario(f"{args.relation}_{args.strength:+.3f}", args.relation, args.strength)
    result = GSISM(model).evaluate(sample, scenario)
    payload = {
        "hashtag": sample.hashtag,
        "origin_date": str(sample.origin_date),
        "scenario": scenario.__dict__,
        "ssc": result.ssc.cpu().tolist(),
        "asc": result.asc.cpu().tolist(),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
