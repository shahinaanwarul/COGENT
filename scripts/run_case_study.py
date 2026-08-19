#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import torch

from cogent.case_study import campaign_forecasts
from cogent.config import ExperimentConfig
from cogent.dataset import COGENTWindowDataset
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
    rows = campaign_forecasts(model, ds, device)
    payload = [{"hashtag": r.hashtag, "origin_date": r.origin_date, "forecast": r.forecast.tolist(), "emergence_probability": r.emergence_probability.tolist()} for r in rows]
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(payload)} campaign forecasts")


if __name__ == "__main__":
    main()
