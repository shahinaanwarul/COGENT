#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from cogent.metrics import holm_adjust, paired_tests


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired fold tests and Holm correction used in the manuscript.")
    ap.add_argument("--input", required=True, help="CSV columns: metric,horizon,model,fold,value; include model=COGENT")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    rows = []
    for (metric, horizon), fam in df.groupby(["metric", "horizon"]):
        base = fam[fam["model"] == "COGENT"].sort_values("fold")
        family_rows = []
        for model, g in fam[fam["model"] != "COGENT"].groupby("model"):
            g = g.sort_values("fold")
            merged = base[["fold", "value"]].merge(g[["fold", "value"]], on="fold", suffixes=("_cogent", "_baseline"))
            stat = paired_tests(merged["value_cogent"], merged["value_baseline"])
            family_rows.append({"metric": metric, "horizon": horizon, "baseline": model, **stat})
        p_t = holm_adjust([r["paired_t_p"] for r in family_rows])
        p_w = holm_adjust([r["wilcoxon_exact_p"] for r in family_rows])
        for r, a, b in zip(family_rows, p_t, p_w):
            r["paired_t_p_holm"] = float(a)
            r["wilcoxon_p_holm"] = float(b)
            rows.append(r)
    Path(args.output).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
