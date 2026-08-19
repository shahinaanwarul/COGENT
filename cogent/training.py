from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence
import copy
import math
import random

import numpy as np
import torch
from torch.optim import AdamW

from .config import ExperimentConfig
from .dataset import COGENTWindowDataset
from .evaluation import evaluate_dataset
from .losses import cogent_loss
from .model import COGENTModel
from .utils import resolve_device, save_checkpoint, set_seed


@dataclass
class TrainingHistory:
    train_loss: List[float]
    val_rmse: List[float]
    best_epoch: int
    best_val_rmse: float


class COGENTTrainer:
    def __init__(self, model: COGENTModel, config: ExperimentConfig, output_dir: str | Path) -> None:
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(config.training.device)
        self.model.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

    def _train_epoch(self, dataset: COGENTWindowDataset, max_samples: int | None = None) -> float:
        self.model.train()
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        if max_samples is not None:
            indices = indices[:max_samples]
        total = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        accum = max(1, self.config.training.grad_accum_steps)
        for step, idx in enumerate(indices):
            sample = dataset[idx].to(self.device)
            out = self.model(sample, n_samples=self.config.model.monte_carlo_samples_train)
            breakdown = cogent_loss(out, sample, self.config.loss)
            (breakdown.total / accum).backward()
            total += float(breakdown.total.detach().cpu())
            if (step + 1) % accum == 0 or step + 1 == len(indices):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
        return total / max(len(indices), 1)

    def _validate(self, dataset: COGENTWindowDataset, max_samples: int | None = None) -> float:
        result = evaluate_dataset(
            self.model,
            dataset,
            self.device,
            horizons=(self.config.data.max_horizon,),
            n_samples=min(32, self.config.model.monte_carlo_samples_eval),
            max_samples=max_samples,
        )
        return float(result.metrics_by_horizon[self.config.data.max_horizon]["RMSE"])

    def fit(
        self,
        train_dataset: COGENTWindowDataset,
        val_dataset: COGENTWindowDataset,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
    ) -> TrainingHistory:
        set_seed(self.config.training.seed)
        best_state = copy.deepcopy(self.model.state_dict())
        best_val = float("inf")
        best_epoch = -1
        train_losses: List[float] = []
        val_rmses: List[float] = []
        stale = 0
        for epoch in range(self.config.training.max_epochs):
            tr = self._train_epoch(train_dataset, max_samples=max_train_samples)
            va = self._validate(val_dataset, max_samples=max_val_samples)
            train_losses.append(tr)
            val_rmses.append(va)
            print(f"epoch={epoch+1:03d} train_loss={tr:.6f} val_nrmse={va:.6f}")
            if va < best_val:
                best_val = va
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                stale = 0
                save_checkpoint(
                    self.output_dir / "best.pt",
                    {
                        "model_state": best_state,
                        "config": self.config.to_dict(),
                        "graph_feature_dim": train_dataset.builder.feature_dim,
                        "modality_dims": train_dataset.modality_dims(),
                        "best_epoch": best_epoch,
                        "best_val_rmse": best_val,
                    },
                )
            else:
                stale += 1
                if stale >= self.config.training.patience:
                    break
        self.model.load_state_dict(best_state)
        return TrainingHistory(train_losses, val_rmses, best_epoch, best_val)


def optuna_objective_factory(
    base_config: ExperimentConfig,
    dataset_factory,
    output_root: str | Path,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
):
    """Build an Optuna objective using exactly the search ranges stated in the manuscript."""
    def objective(trial):
        cfg = copy.deepcopy(base_config)
        cfg.training.learning_rate = trial.suggest_float("learning_rate", *cfg.training.lr_range, log=True)
        cfg.model.hidden_dim = trial.suggest_categorical("hidden_dim", list(cfg.training.hidden_dim_choices))
        valid_heads = [h for h in cfg.training.head_choices if cfg.model.hidden_dim % h == 0]
        cfg.model.transformer_heads = trial.suggest_categorical("transformer_heads", valid_heads)
        cfg.model.dropout = trial.suggest_float("dropout", *cfg.training.dropout_range)
        cfg.training.batch_size_nominal = trial.suggest_categorical("batch_size", list(cfg.training.batch_choices))
        cfg.training.weight_decay = trial.suggest_float("weight_decay", *cfg.training.weight_decay_range, log=True)
        train_ds, val_ds = dataset_factory(cfg)
        model = COGENTModel(cfg, train_ds.builder.feature_dim, train_ds.modality_dims())
        trainer = COGENTTrainer(model, cfg, Path(output_root) / f"trial_{trial.number:03d}")
        hist = trainer.fit(train_ds, val_ds, max_train_samples=max_train_samples, max_val_samples=max_val_samples)
        return hist.best_val_rmse
    return objective
