#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cogent.config import ExperimentConfig
from cogent.dataset import COGENTWindowDataset
from cogent.evaluation import evaluate_dataset
from cogent.graph import EmpiricalGraphBuilder
from cogent.io import load_prepared_directory
from cogent.model import COGENTModel
from cogent.semantic import load_embedding_map
from cogent.utils import resolve_device


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained COGENT checkpoint.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ExperimentConfig.from_dict(ckpt["config"])
    train, val, test, state = load_prepared_directory(args.prepared)
    embeddings = load_embedding_map(args.embeddings)
    builder = EmpiricalGraphBuilder(cfg.data, cfg.graph, embeddings, random_state=cfg.training.seed)
    builder.fit_topics(train[cfg.data.hashtag_col].astype(str).unique())
    val_ds = COGENTWindowDataset(val, cfg.data, cfg.graph, builder, embeddings, state["country_vocab"], state["platform_vocab"])
    test_ds = COGENTWindowDataset(test, cfg.data, cfg.graph, builder, embeddings, state["country_vocab"], state["platform_vocab"])
    model = COGENTModel(cfg, ckpt["graph_feature_dim"], ckpt["modality_dims"])
    model.load_state_dict(ckpt["model_state"])
    device = resolve_device(cfg.training.device)
    model.to(device)
    result = evaluate_dataset(
        model,
        test_ds,
        device,
        cfg.data.forecast_horizons,
        threshold_dataset=val_ds,
        n_samples=cfg.model.monte_carlo_samples_eval,
        max_samples=args.max_samples,
    )
    payload = {"threshold": result.threshold, "metrics_by_horizon": result.metrics_by_horizon}
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
