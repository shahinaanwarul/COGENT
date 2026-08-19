#!/usr/bin/env python
from __future__ import annotations

import argparse
import pandas as pd

from cogent.config import ExperimentConfig, manuscript_defaults
from cogent.data import AggregatePreprocessor


def main() -> None:
    ap = argparse.ArgumentParser(description="Fuse source-disaggregated exports into D_agg without treating sources as platforms.")
    ap.add_argument("--input", required=True, help="CSV with Source plus manuscript data fields")
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = ExperimentConfig.load(args.config) if args.config else manuscript_defaults()
    df = pd.read_csv(args.input)
    prep = AggregatePreprocessor(cfg.data, cfg.labels)
    agg = prep.fuse_sources(df)
    agg.to_csv(args.output, index=False)
    print(f"Wrote {len(agg)} fused hashtag-date rows to {args.output}")


if __name__ == "__main__":
    main()
