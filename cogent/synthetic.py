from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd


def make_synthetic_aggregate(
    n_hashtags: int = 10,
    n_days: int = 80,
    seed: int = 42,
    with_platform: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    countries = ["India", "United States", "United Kingdom", "Canada"]
    platforms = ["X", "Instagram", "TikTok", "YouTube"]
    rows = []
    for h in range(n_hashtags):
        phase = rng.uniform(0, 2 * np.pi)
        level = rng.uniform(20, 100)
        for t, date in enumerate(dates):
            trend = level + 15 * np.sin(t / 8 + phase) + 0.3 * t + rng.normal(0, 3)
            reach = max(1.0, 8 * trend + rng.normal(0, 15))
            sentiment = np.clip(0.4 * np.sin(t / 13 + phase) + rng.normal(0, 0.08), -1, 1)
            row = {
                "Date": date,
                "Hashtag": f"trend{h}",
                "Mentions": max(0.0, trend),
                "Estimated_Reach": reach,
                "Sentiment_Score": sentiment,
                "Top_Country": countries[(h + t // 15) % len(countries)],
            }
            if with_platform:
                row["Platform"] = platforms[(h + t // 10) % len(platforms)]
            rows.append(row)
    return pd.DataFrame(rows)


def synthetic_embeddings(hashtags: Sequence[str], dim: int = 64, seed: int = 42) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for h in hashtags:
        x = rng.normal(size=dim).astype(np.float32)
        x /= np.linalg.norm(x) + 1e-8
        out[str(h)] = x
    return out
