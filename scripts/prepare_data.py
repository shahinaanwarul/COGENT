#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cogent.config import ExperimentConfig, manuscript_defaults
from cogent.data import AggregatePreprocessor
from cogent.io import save_preprocessor_state


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare aggregate Trending#-style data for COGENT.")
    ap.add_argument("--input", required=True, help="Aggregate CSV containing Date, Hashtag, Mentions, Estimated_Reach, Sentiment_Score, Top_Country.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--apply-eligibility", action="store_true")
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.config) if args.config else manuscript_defaults()
    df = pd.read_csv(args.input)
    prep = AggregatePreprocessor(cfg.data, cfg.labels)
    split = prep.prepare(df, apply_eligibility=args.apply_eligibility)
    if "Semantic_Novelty" in split.train.columns:
        prep.fit_label_thresholds(split.train)
        split.train = prep.apply_emergence_labels(split.train)
        split.val = prep.apply_emergence_labels(split.val)
        split.test = prep.apply_emergence_labels(split.test)
    else:
        print("Semantic_Novelty is absent: no Emerging label is generated. Add a manuscript-consistent novelty column to train the detection head.")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    split.train.to_csv(out / "train.csv", index=False)
    split.val.to_csv(out / "val.csv", index=False)
    split.test.to_csv(out / "test.csv", index=False)
    save_preprocessor_state(out / "preprocessor_state.json", prep)
    cfg.save(out / "config.json")
    print(f"Wrote prepared splits to {out}")


if __name__ == "__main__":
    main()
