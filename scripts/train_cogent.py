#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from cogent.config import ExperimentConfig
from cogent.dataset import COGENTWindowDataset
from cogent.graph import EmpiricalGraphBuilder
from cogent.io import load_prepared_directory
from cogent.model import COGENTModel
from cogent.semantic import load_embedding_map
from cogent.training import COGENTTrainer


def build_datasets(prepared_dir, embeddings_path, cfg):
    train, val, test, state = load_prepared_directory(prepared_dir)
    embeddings = load_embedding_map(embeddings_path)
    builder = EmpiricalGraphBuilder(cfg.data, cfg.graph, embeddings, random_state=cfg.training.seed)
    builder.fit_topics(train[cfg.data.hashtag_col].astype(str).unique())
    country_vocab = state["country_vocab"]
    platform_vocab = state["platform_vocab"]
    train_ds = COGENTWindowDataset(train, cfg.data, cfg.graph, builder, embeddings, country_vocab, platform_vocab)
    val_ds = COGENTWindowDataset(val, cfg.data, cfg.graph, builder, embeddings, country_vocab, platform_vocab)
    test_ds = COGENTWindowDataset(test, cfg.data, cfg.graph, builder, embeddings, country_vocab, platform_vocab)
    return train_ds, val_ds, test_ds


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the COGENT predictive core.")
    ap.add_argument("--prepared", required=True, help="Directory created by prepare_data.py")
    ap.add_argument("--embeddings", required=True, help="NPZ created by precompute_embeddings.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true", help="Small smoke-training run")
    args = ap.parse_args()

    cfg_path = args.config or str(Path(args.prepared) / "config.json")
    cfg = ExperimentConfig.load(cfg_path)
    if args.quick:
        cfg.training.max_epochs = 2
        cfg.training.patience = 2
        cfg.model.monte_carlo_samples_train = 2
        cfg.model.monte_carlo_samples_eval = 4
        cfg.model.hidden_dim = 64
        cfg.model.graph_hidden_dim = 64
        cfg.model.latent_dim = 32
        cfg.model.transformer_heads = 4
        cfg.model.tmft_layers = 2

    train_ds, val_ds, _ = build_datasets(args.prepared, args.embeddings, cfg)
    if not len(train_ds) or not len(val_ds):
        raise RuntimeError("Prepared train/validation splits do not contain enough within-split history for lookback+horizon windows.")
    model = COGENTModel(cfg, train_ds.builder.feature_dim, train_ds.modality_dims())
    trainer = COGENTTrainer(model, cfg, args.output)
    hist = trainer.fit(train_ds, val_ds, max_train_samples=16 if args.quick else None, max_val_samples=8 if args.quick else None)
    print(f"best_epoch={hist.best_epoch+1} best_val_nrmse={hist.best_val_rmse:.6f}")


if __name__ == "__main__":
    main()
