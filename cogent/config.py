from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import json


@dataclass
class DataConfig:
    date_col: str = "Date"
    hashtag_col: str = "Hashtag"
    mentions_col: str = "Mentions"
    reach_col: str = "Estimated_Reach"
    sentiment_col: str = "Sentiment_Score"
    country_col: str = "Top_Country"
    platform_col: str = "Platform"
    source_col: str = "Source"
    target_col: str = "Mentions"
    engagement_col: str = "Estimated_Reach"
    train_fraction: float = 0.70
    val_fraction: float = 0.10
    test_fraction: float = 0.20
    min_sources_per_record: int = 2
    max_forward_fill_days: int = 2
    min_consecutive_days: int = 30
    min_nonzero_days: int = 10
    min_cross_source_fraction: float = 0.80
    language: str = "en"
    forecast_horizons: Tuple[int, ...] = (1, 3, 7, 14, 30)
    max_horizon: int = 30


@dataclass
class GraphConfig:
    lookback_days: int = 14
    knn_k: int = 5
    sentiment_negative_cut: float = -0.05
    sentiment_positive_cut: float = 0.05
    topic_clusters: int = 8
    add_platform_relation_if_available: bool = True
    relation_names: Tuple[str, ...] = ("sim", "geo", "sent", "topic", "plat")
    heat_tau_star: float = 1.0
    gamma_shape_init: float = 2.0
    gamma_rate_init: float = 2.0
    gamma_shape_min: float = 0.10
    gamma_rate_min: float = 0.10
    graph_smoothness_weight: float = 1e-4


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    graph_hidden_dim: int = 256
    tmft_layers: int = 4
    transformer_heads: int = 8
    ff_multiplier: int = 4
    dropout: float = 0.20
    latent_dim: int = 128
    semantic_dim: int = 4096
    semantic_encoder_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    diffusion_bias_eta_init: float = 1.0
    monte_carlo_samples_train: int = 16
    monte_carlo_samples_eval: int = 128
    observation_noise_floor: float = 1e-4
    emergence_head: bool = True


@dataclass
class LossConfig:
    prediction_mse_weight: float = 1.0
    crps_weight: float = 1.0
    classification_weight: float = 0.25
    graph_smoothness_weight: float = 1e-4
    calibration_weight: float = 0.10
    attribution_sparsity_weight: float = 1e-3
    attribution_stability_weight: float = 1e-3
    structural_consistency_weight: float = 0.0
    decision_regularization_weight: float = 0.0
    calibration_levels: Tuple[float, ...] = (0.50, 0.80, 0.90)


@dataclass
class TrainingConfig:
    seed: int = 42
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 120
    patience: int = 12
    grad_clip_norm: float = 1.0
    grad_accum_steps: int = 1
    batch_size_nominal: int = 16
    device: str = "auto"
    num_workers: int = 0
    n_rolling_folds: int = 5
    optuna_trials: int = 150
    lr_range: Tuple[float, float] = (1e-5, 1e-3)
    hidden_dim_choices: Tuple[int, ...] = (128, 256, 512, 768)
    head_choices: Tuple[int, ...] = (4, 8, 12)
    dropout_range: Tuple[float, float] = (0.1, 0.5)
    batch_choices: Tuple[int, ...] = (16, 32, 64, 128)
    weight_decay_range: Tuple[float, float] = (1e-6, 1e-2)


@dataclass
class LabelConfig:
    mention_growth_sigma_multiplier: float = 1.5
    engagement_growth_sigma_multiplier: float = 1.2
    semantic_novelty_threshold: float = 0.35
    novelty_history_days: int = 14


@dataclass
class DecisionConfig:
    discount_gamma: float = 0.95
    risk_lambda: float = 0.25
    cvar_beta: float = 0.90
    default_action_cost: float = 0.0
    policy_smoothness_weight: float = 0.0
    attribution_coherence_weight: float = 0.0


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        return cls(
            data=DataConfig(**data.get("data", {})),
            graph=GraphConfig(**data.get("graph", {})),
            model=ModelConfig(**data.get("model", {})),
            loss=LossConfig(**data.get("loss", {})),
            training=TrainingConfig(**data.get("training", {})),
            labels=LabelConfig(**data.get("labels", {})),
            decision=DecisionConfig(**data.get("decision", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def manuscript_defaults() -> ExperimentConfig:
    """Return manuscript-grounded defaults plus explicit implementation defaults.

    Manuscript-specified values include L=14, k=5, sentiment cuts ±0.05,
    eight topic clusters, 70/10/20 chronological partitions, AdamW training,
    120 maximum epochs, patience 12, and the stated Optuna search ranges.

    Neural widths, Monte Carlo sample counts, Gamma-kernel numerical details,
    loss weights, and decision cost/risk constants are not uniquely fixed by the
    manuscript. They are therefore exposed here as editable implementation defaults.
    """
    return ExperimentConfig()
