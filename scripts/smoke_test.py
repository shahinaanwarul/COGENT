#!/usr/bin/env python
from __future__ import annotations

import copy
import torch

from cogent.config import manuscript_defaults
from cogent.data import AggregatePreprocessor
from cogent.dataset import COGENTWindowDataset
from cogent.graph import EmpiricalGraphBuilder
from cogent.losses import cogent_loss
from cogent.model import COGENTModel
from cogent.synthetic import make_synthetic_aggregate, synthetic_embeddings
from cogent.utils import set_seed


def main() -> None:
    set_seed(7)
    cfg = manuscript_defaults()
    # Smoke-test-only reductions. They do not alter manuscript_defaults().
    cfg.data.max_horizon = 5
    cfg.data.forecast_horizons = (1, 3, 5)
    cfg.model.hidden_dim = 32
    cfg.model.graph_hidden_dim = 32
    cfg.model.latent_dim = 16
    cfg.model.transformer_heads = 4
    cfg.model.tmft_layers = 1
    cfg.model.monte_carlo_samples_train = 2
    cfg.model.monte_carlo_samples_eval = 4
    cfg.model.semantic_dim = 32

    df = make_synthetic_aggregate(n_hashtags=6, n_days=120, seed=7)
    prep = AggregatePreprocessor(cfg.data, cfg.labels)
    split = prep.prepare(df)
    hashtags = split.train[cfg.data.hashtag_col].astype(str).unique()
    emb = synthetic_embeddings(hashtags, dim=32, seed=7)
    # Ensure held-out hashtags use the same static mapping.
    all_tags = df[cfg.data.hashtag_col].astype(str).str.lower().str.lstrip("#").unique()
    emb.update(synthetic_embeddings([h for h in all_tags if h not in emb], dim=32, seed=8))
    builder = EmpiricalGraphBuilder(cfg.data, cfg.graph, emb, random_state=7)
    builder.fit_topics(hashtags)
    ds = COGENTWindowDataset(split.train, cfg.data, cfg.graph, builder, emb, prep.country_vocab, prep.platform_vocab, horizon=cfg.data.max_horizon)
    sample = ds[0]
    model = COGENTModel(cfg, builder.feature_dim, ds.modality_dims())
    out = model(sample, n_samples=2)
    loss = cogent_loss(out, sample, cfg.loss)
    loss.total.backward()
    print("COGENT smoke test passed")
    print("forecast shape:", tuple(out.forecast.trajectory_samples.shape))
    print("relations:", list(out.graph.relation_states))
    print("loss:", float(loss.total.detach()))


if __name__ == "__main__":
    main()
