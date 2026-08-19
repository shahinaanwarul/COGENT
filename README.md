# COGENT Python implementation

This package is a source-grounded reference implementation of the manuscript “COGENT: An Explainable Generative AI Framework for Early Trend Forecasting and Structural Sensitivity Analysis in Social Media Networks.

## What is implemented

The code follows the manuscript's computational order:

1. Data preparation and source fusion (`cogent/data.py`)  
   - Unicode/lower-case hashtag normalization.
   - Separation of measurement-source identity from explicit social-platform labels.
   - At least two valid sources for a fused hashtag-date record.
   - Chronological 70%/10%/20% train/validation/test partitioning.
   - Causal forward fill for gaps of at most two days within a split and missingness masks.
   - Train-only numerical scaling and categorical vocabularies.
   - Operational emerging-trend threshold support when a `Semantic_Novelty` column is supplied.

2. Empirical heterogeneous graph (`cogent/graph.py`)  
   - Causal look-back `L=14` days.
   - `k=5` hashtag nearest neighbours.
   - Positive cosine hashtag-similarity relation.
   - Hashtag-country weight `sqrt(m*q)`.
   - Sentiment bins using `-0.05` and `+0.05` cut points.
   - Eight training-only hashtag-topic clusters.
   - Optional hashtag-platform incidence only when an explicit platform label exists.
   - Symmetric normalized adjacency and Laplacian.
   - Identity, endpoint-shuffle, and relation-removal graph controls.

3. CH-GDM (`cogent/chgdm.py`)  
   - Typed node encoding.
   - Lifecycle-conditioned Gamma heat-diffusion parameters.
   - Relation attention and residual gating.
   - Relation-specific heat kernels and normalized Laplacians.

4. TMFT (`cogent/tmft.py`)  
   - Multimodal projections for engagement, sentiment, semantic, geographic, platform, calendar, and missingness inputs.
   - Continuous lifecycle encoding.
   - Diffusion-biased multi-head attention implementing the manuscript's additive heat-kernel bias.
   - Gated temporal pooling.

5. GTFM (`cogent/gtfm.py`)  
   - Graph-driven latent stochastic differential dynamics.
   - Monte Carlo trajectory generation.
   - Probabilistic emergence head.
   - Shared baseline/perturbed decoder for structural scenarios.

6. GSISM (`cogent/gsism.py`)  
   - Bounded relation-matrix perturbations.
   - Paired baseline/perturbed simulations using common random noise.
   - SSC and ASC trajectory contrasts.
   - These are **structural model sensitivities, not identified causal effects**, consistent with the manuscript.

7. EGAM (`cogent/egam.py`)  
   - Integrated gradients over CH-GDM relation-diffused states.
   - Integrated gradients over multimodal inputs.
   - Relation, temporal, and feature importance summaries.

8. DAUOM (`cogent/dauom.py`)  
   - Discounted SSC reward.
   - Action cost.
   - CVaR-based downside risk.
   - Finite-action utility ranking.

9. Training/evaluation/statistics  
   - AdamW, early stopping (maximum 120 epochs, patience 12).
   - Optuna search ranges from the manuscript: learning rate `[1e-5,1e-3]`, hidden dimension `{128,256,512,768}`, heads `{4,8,12}`, dropout `[0.1,0.5]`, batch candidates `{16,32,64,128}`, weight decay `[1e-6,1e-2]`, up to 150 trials.
   - RMSE, MAE, F1, AUC, CRPS, 90% coverage, interval width, calibration error.
   - Paired t-test, exact Wilcoxon, Holm correction, Cohen's `d_z`, rank-biserial correlation, Student-t confidence intervals.


### Gamma diffusion numerical implementation

The manuscript states that the Gamma-kernel heat diffusion uses a conjugate-gradient approximation but does not give the actual approximation order/tolerance. `CHGDM.gamma_heat_operator` therefore evaluates the same Gamma heat integral exactly in the Laplacian eigenbasis:

`E[exp(-tau L)] = U diag((rate/(rate+lambda))**shape) U^T`.

This is a transparent reference implementation of the stated mathematical operator, not a claim to reproduce an unpublished CG routine.

### Semantic encoder

The manuscript specifies frozen `meta-llama/Meta-Llama-3-8B-Instruct` final-layer embeddings with masked mean pooling. `LlamaSemanticEncoder` implements that path and requires Hugging Face `transformers` plus checkpoint access. `HashingSemanticEncoder` exists only for local smoke tests and is not manuscript-equivalent.

## Installation

```bash
cd COGENT_python_implementation
pip install -e .
```

For the manuscript LLaMA encoder and Prophet baseline:

```bash
pip install -r requirements-full.txt
```

## Data schema

The aggregate CSV should contain at least:

```text
Date,Hashtag,Mentions,Estimated_Reach,Sentiment_Score,Top_Country
```

An explicit `Platform` column is optional. A source-disaggregated panel can additionally contain `Source`; use `scripts/fuse_sources.py` to create the aggregate panel. Provider/source names are never treated as social-platform nodes.

## Typical workflow

Prepare aggregate data:

```bash
python scripts/prepare_data.py --input trending.csv --output prepared
```

Create manuscript LLaMA embeddings:

```bash
python scripts/precompute_embeddings.py --input trending.csv --output hashtag_embeddings.npz --encoder llama3
```

Train:

```bash
python scripts/train_cogent.py \
  --prepared prepared \
  --embeddings hashtag_embeddings.npz \
  --output runs/cogent
```

Evaluate the 1/3/7/14/30-day horizons:

```bash
python scripts/evaluate_cogent.py \
  --checkpoint runs/cogent/best.pt \
  --prepared prepared \
  --embeddings hashtag_embeddings.npz \
  --output runs/cogent/test_metrics.json
```

Run graph controls:

```bash
python scripts/run_graph_ablations.py \
  --checkpoint runs/cogent/best.pt \
  --prepared prepared \
  --embeddings hashtag_embeddings.npz \
  --output runs/cogent/graph_ablations.json
```

Run a structural sensitivity scenario:

```bash
python scripts/run_gsism.py \
  --checkpoint runs/cogent/best.pt \
  --prepared prepared \
  --embeddings hashtag_embeddings.npz \
  --relation sim --strength 0.20 \
  --output runs/cogent/gsism.json
```

Generate EGAM explanations:

```bash
python scripts/run_egam.py \
  --checkpoint runs/cogent/best.pt \
  --prepared prepared \
  --embeddings hashtag_embeddings.npz \
  --output runs/cogent/egam.json
```

Rank actions with DAUOM:

```bash
python scripts/run_dauom.py \
  --checkpoint runs/cogent/best.pt \
  --prepared prepared \
  --embeddings hashtag_embeddings.npz \
  --output runs/cogent/actions.json
```

## Smoke test

The smoke test uses a synthetic dataset and deliberately reduced dimensions/horizon only to verify code execution:

```bash
PYTHONPATH=. python scripts/smoke_test.py
pytest -q
```

## File map

- `cogent/config.py` — all manuscript and implementation configuration.
- `cogent/data.py` — source fusion, partitioning, missing data, scaling and labels.
- `cogent/semantic.py` — LLaMA and smoke-test semantic encoders.
- `cogent/graph.py` — empirical CH-GDM graph builder and topology controls.
- `cogent/chgdm.py` — continuous relation-specific diffusion.
- `cogent/tmft.py` — diffusion-biased temporal multimodal fusion.
- `cogent/gtfm.py` — graph-driven probabilistic trajectory generation.
- `cogent/gsism.py` — paired structural sensitivity analysis.
- `cogent/egam.py` — integrated-gradient explanations.
- `cogent/dauom.py` — cost/risk-aware action ranking.
- `cogent/model.py` — end-to-end COGENT composition.
- `cogent/losses.py` — predictive, CRPS, classification, calibration and graph smoothness losses.
- `cogent/metrics.py` — forecasting, calibration, attribution and statistical metrics.
- `cogent/training.py` — AdamW training and Optuna objective.
- `cogent/evaluation.py` — horizon-wise model evaluation.
- `cogent/experiments.py` — rolling-fold helper, data-efficiency and graph-ablation utilities.
- `cogent/case_study.py` — five manuscript campaign helpers.
- `cogent/baselines.py` — ARIMA, Prophet, LSTM, matched Transformer and explicit external adapters.
- `scripts/*.py` — command-line workflows.
- `tests/*.py` — graph and model smoke tests.
