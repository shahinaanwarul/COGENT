#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import optuna

from cogent.config import ExperimentConfig
from cogent.dataset import COGENTWindowDataset
from cogent.graph import EmpiricalGraphBuilder
from cogent.io import load_prepared_directory
from cogent.semantic import load_embedding_map
from cogent.training import optuna_objective_factory


def main() -> None:
    ap = argparse.ArgumentParser(description="Optuna tuning using manuscript search ranges.")
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--trials", type=int, default=None)
    args = ap.parse_args()
    cfg = ExperimentConfig.load(Path(args.prepared) / "config.json")
    train, val, _, state = load_prepared_directory(args.prepared)
    embeddings = load_embedding_map(args.embeddings)

    def dataset_factory(trial_cfg):
        builder = EmpiricalGraphBuilder(trial_cfg.data, trial_cfg.graph, embeddings, random_state=trial_cfg.training.seed)
        builder.fit_topics(train[trial_cfg.data.hashtag_col].astype(str).unique())
        train_ds = COGENTWindowDataset(train, trial_cfg.data, trial_cfg.graph, builder, embeddings, state["country_vocab"], state["platform_vocab"])
        val_ds = COGENTWindowDataset(val, trial_cfg.data, trial_cfg.graph, builder, embeddings, state["country_vocab"], state["platform_vocab"])
        return train_ds, val_ds

    objective = optuna_objective_factory(cfg, dataset_factory, args.output)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials or cfg.training.optuna_trials)
    print("Best parameters:", study.best_params)
    print("Best validation NRMSE:", study.best_value)


if __name__ == "__main__":
    main()
