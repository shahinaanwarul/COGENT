from cogent.config import manuscript_defaults
from cogent.data import AggregatePreprocessor
from cogent.dataset import COGENTWindowDataset
from cogent.graph import EmpiricalGraphBuilder
from cogent.model import COGENTModel
from cogent.synthetic import make_synthetic_aggregate, synthetic_embeddings


def test_forward_shapes():
    cfg = manuscript_defaults()
    cfg.data.max_horizon = 3
    cfg.model.hidden_dim = 16
    cfg.model.graph_hidden_dim = 16
    cfg.model.latent_dim = 8
    cfg.model.transformer_heads = 4
    cfg.model.tmft_layers = 1
    cfg.model.monte_carlo_samples_train = 2
    df = make_synthetic_aggregate(n_hashtags=4, n_days=90, seed=1)
    prep = AggregatePreprocessor(cfg.data, cfg.labels)
    split = prep.prepare(df)
    tags = df[cfg.data.hashtag_col].astype(str).str.lower().str.lstrip("#").unique()
    emb = synthetic_embeddings(tags, dim=16, seed=1)
    builder = EmpiricalGraphBuilder(cfg.data, cfg.graph, emb, random_state=1)
    builder.fit_topics(split.train[cfg.data.hashtag_col].unique())
    ds = COGENTWindowDataset(split.train, cfg.data, cfg.graph, builder, emb, prep.country_vocab, prep.platform_vocab, horizon=3)
    sample = ds[0]
    model = COGENTModel(cfg, builder.feature_dim, ds.modality_dims())
    out = model(sample, n_samples=2)
    assert tuple(out.forecast.trajectory_samples.shape) == (2, 3)
    assert out.forecast.mean_trajectory.shape[0] == 3
