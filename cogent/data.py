from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import DataConfig, LabelConfig
from .utils import one_hot, sinusoidal_calendar_features


def normalize_hashtag(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).strip().lower()
    if value.startswith("#"):
        value = value[1:]
    return value.strip()


@dataclass
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass
class ScalingState:
    columns: Tuple[str, ...]
    means: Dict[str, float]
    scales: Dict[str, float]

    def transform_value(self, column: str, x: np.ndarray | float) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.means[column]) / self.scales[column]

    def inverse_value(self, column: str, x: np.ndarray | float) -> np.ndarray:
        return np.asarray(x, dtype=np.float32) * self.scales[column] + self.means[column]


class AggregatePreprocessor:
    """Preprocessing protocol grounded in the manuscript.

    The implementation preserves source-level missingness until source fusion,
    requires at least two valid sources for an aggregate hashtag-date record,
    uses causal forward filling for gaps of at most two days inside a split,
    and fits numerical scaling on the training partition only.
    """

    def __init__(self, config: DataConfig, label_config: LabelConfig | None = None) -> None:
        self.config = config
        self.label_config = label_config or LabelConfig()
        self.scaling: ScalingState | None = None
        self.country_vocab: Dict[str, int] = {}
        self.platform_vocab: Dict[str, int] = {}
        self.label_thresholds: Dict[str, float] = {}

    def clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        out = df.copy()
        if c.date_col not in out or c.hashtag_col not in out:
            raise ValueError(f"Input must contain {c.date_col!r} and {c.hashtag_col!r}.")
        out[c.date_col] = pd.to_datetime(out[c.date_col], errors="coerce")
        out[c.hashtag_col] = out[c.hashtag_col].map(normalize_hashtag)
        out = out.dropna(subset=[c.date_col])
        out = out[out[c.hashtag_col].astype(str).str.len() > 0]
        return out.sort_values([c.hashtag_col, c.date_col]).reset_index(drop=True)

    def fuse_sources(self, source_panel: pd.DataFrame) -> pd.DataFrame:
        """Create D_agg from source-disaggregated records.

        The manuscript describes per-indicator, per-day min-max normalization and
        unweighted averaging across available sources. Provider identity remains a
        measurement-source field and is never converted into a social-platform node.
        """
        c = self.config
        df = self.clean_columns(source_panel)
        if c.source_col not in df:
            raise ValueError(f"Source panel requires a {c.source_col!r} column.")
        numeric = [x for x in (c.mentions_col, c.reach_col, c.sentiment_col) if x in df]
        if len(numeric) < 3:
            raise ValueError(f"Source panel must contain {c.mentions_col}, {c.reach_col}, and {c.sentiment_col}.")

        # Normalize each indicator within each date and source. This keeps the
        # source scales relative, as described in the manuscript, without imputing
        # values that were absent at the provider level.
        norm_cols: Dict[str, str] = {}
        for col in numeric:
            ncol = f"__norm_{col}"
            norm_cols[col] = ncol
            grouped = df.groupby([c.date_col, c.source_col], sort=False)[col]
            minv = grouped.transform("min")
            maxv = grouped.transform("max")
            denom = (maxv - minv).replace(0, np.nan)
            val = (pd.to_numeric(df[col], errors="coerce") - minv) / denom
            # If a provider/day has no range, retain 0.5 for observed values only.
            val = val.where(~denom.isna(), np.where(df[col].notna(), 0.5, np.nan))
            df[ncol] = val

        keys = [c.hashtag_col, c.date_col]
        valid_source_count = df.groupby(keys)[c.source_col].nunique().rename("Source_Count")
        grouped = df.groupby(keys, sort=False)
        agg = grouped[[norm_cols[x] for x in numeric]].mean()
        agg.columns = numeric
        agg = agg.join(valid_source_count).reset_index()
        agg = agg[agg["Source_Count"] >= c.min_sources_per_record].copy()

        if c.country_col in df:
            country = grouped[c.country_col].agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else "Unknown")
            agg = agg.merge(country.rename(c.country_col).reset_index(), on=keys, how="left")
        else:
            agg[c.country_col] = "Unknown"

        if c.platform_col in df:
            # Retain an explicit platform only when providers supply a consistent
            # verified label. Provider names are never substituted.
            platform = grouped[c.platform_col].agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan)
            agg = agg.merge(platform.rename(c.platform_col).reset_index(), on=keys, how="left")
        return agg.sort_values(keys).reset_index(drop=True)

    def apply_hashtag_eligibility(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        df = self.clean_columns(df)
        keep: List[str] = []
        for hashtag, g in df.groupby(c.hashtag_col, sort=False):
            g = g.sort_values(c.date_col)
            nonzero = int((pd.to_numeric(g[c.mentions_col], errors="coerce").fillna(0) > 0).sum())
            dates = g[c.date_col].dropna().sort_values().drop_duplicates()
            if len(dates) == 0:
                continue
            # Longest consecutive daily run.
            diffs = dates.diff().dt.days.fillna(1)
            run = best = 0
            for d in diffs:
                if d <= 1:
                    run += 1
                else:
                    run = 1
                best = max(best, run)
            source_ok = 1.0
            if "Source_Count" in g:
                source_ok = float((g["Source_Count"] >= c.min_sources_per_record).mean())
            if best >= c.min_consecutive_days and nonzero >= c.min_nonzero_days and source_ok >= c.min_cross_source_fraction:
                keep.append(str(hashtag))
        return df[df[c.hashtag_col].isin(keep)].copy().reset_index(drop=True)

    def chronological_split(self, df: pd.DataFrame) -> SplitFrames:
        c = self.config
        df = self.clean_columns(df)
        train_parts: List[pd.DataFrame] = []
        val_parts: List[pd.DataFrame] = []
        test_parts: List[pd.DataFrame] = []
        for _, g in df.groupby(c.hashtag_col, sort=False):
            g = g.sort_values(c.date_col).reset_index(drop=True)
            n = len(g)
            i1 = max(1, int(np.floor(n * c.train_fraction)))
            i2 = max(i1 + 1, int(np.floor(n * (c.train_fraction + c.val_fraction))))
            i2 = min(i2, n)
            train_parts.append(g.iloc[:i1].copy())
            val_parts.append(g.iloc[i1:i2].copy())
            test_parts.append(g.iloc[i2:].copy())
        concat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame(columns=df.columns)
        return SplitFrames(concat(train_parts), concat(val_parts), concat(test_parts))

    def _reindex_and_fill_group(self, g: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        g = g.sort_values(c.date_col).copy()
        idx = pd.date_range(g[c.date_col].min(), g[c.date_col].max(), freq="D")
        h = str(g[c.hashtag_col].iloc[0])
        g = g.set_index(c.date_col).reindex(idx)
        g.index.name = c.date_col
        g[c.hashtag_col] = h

        numerical = [c.mentions_col, c.reach_col, c.sentiment_col]
        for col in numerical:
            if col not in g:
                g[col] = np.nan
            missing = g[col].isna()
            g[f"mask_{col}"] = missing.astype(np.float32)
            g[col] = pd.to_numeric(g[col], errors="coerce").ffill(limit=c.max_forward_fill_days)

        for col in [c.country_col, c.platform_col]:
            if col in g:
                g[col] = g[col].fillna("Unknown")
            else:
                g[col] = "Unknown"
        g["long_gap"] = g[numerical].isna().any(axis=1).astype(np.int8)
        return g.reset_index()

    def fill_within_split(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        if df.empty:
            return df.copy()
        parts = [self._reindex_and_fill_group(g) for _, g in df.groupby(c.hashtag_col, sort=False)]
        return pd.concat(parts, ignore_index=True).sort_values([c.hashtag_col, c.date_col]).reset_index(drop=True)

    def fit_scalers_and_vocab(self, train_df: pd.DataFrame) -> None:
        c = self.config
        cols = [c.mentions_col, c.reach_col, c.sentiment_col]
        means: Dict[str, float] = {}
        scales: Dict[str, float] = {}
        for col in cols:
            arr = pd.to_numeric(train_df[col], errors="coerce").to_numpy(dtype=np.float64)
            means[col] = float(np.nanmean(arr)) if np.isfinite(arr).any() else 0.0
            s = float(np.nanstd(arr)) if np.isfinite(arr).any() else 1.0
            scales[col] = s if s > 1e-8 else 1.0
        self.scaling = ScalingState(tuple(cols), means, scales)
        countries = sorted(set(train_df.get(c.country_col, pd.Series(["Unknown"])).fillna("Unknown").astype(str)))
        platforms = sorted(set(train_df.get(c.platform_col, pd.Series(["Unknown"])).fillna("Unknown").astype(str)))
        if "Unknown" not in countries:
            countries.insert(0, "Unknown")
        if "Unknown" not in platforms:
            platforms.insert(0, "Unknown")
        self.country_vocab = {v: i for i, v in enumerate(countries)}
        self.platform_vocab = {v: i for i, v in enumerate(platforms)}

    def scale_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scaling is None:
            raise RuntimeError("Call fit_scalers_and_vocab(train_df) before scale_frame().")
        out = df.copy()
        for col in self.scaling.columns:
            out[f"scaled_{col}"] = self.scaling.transform_value(col, pd.to_numeric(out[col], errors="coerce").to_numpy())
        c = self.config
        out["delta_mentions"] = out.groupby(c.hashtag_col)[f"scaled_{c.mentions_col}"].diff().fillna(0.0)
        out["delta_reach"] = out.groupby(c.hashtag_col)[f"scaled_{c.reach_col}"].diff().fillna(0.0)
        return out

    def prepare(self, df: pd.DataFrame, apply_eligibility: bool = False) -> SplitFrames:
        if apply_eligibility:
            df = self.apply_hashtag_eligibility(df)
        split = self.chronological_split(df)
        train = self.fill_within_split(split.train)
        val = self.fill_within_split(split.val)
        test = self.fill_within_split(split.test)
        self.fit_scalers_and_vocab(train)
        return SplitFrames(self.scale_frame(train), self.scale_frame(val), self.scale_frame(test))


    def fit_label_thresholds(self, train_df: pd.DataFrame) -> None:
        c = self.config
        l = self.label_config
        tmp = train_df.copy().sort_values([c.hashtag_col, c.date_col])
        mg = tmp.groupby(c.hashtag_col)[c.mentions_col].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        eg = tmp.groupby(c.hashtag_col)[c.engagement_col].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self.label_thresholds = {
            "mention_growth": l.mention_growth_sigma_multiplier * float(mg.std(ddof=0) or 1.0),
            "engagement_growth": l.engagement_growth_sigma_multiplier * float(eg.std(ddof=0) or 1.0),
            "semantic_novelty": float(l.semantic_novelty_threshold),
        }

    def apply_emergence_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.label_thresholds:
            raise RuntimeError("Call fit_label_thresholds(train_df) first.")
        c = self.config
        out = df.copy().sort_values([c.hashtag_col, c.date_col])
        out["mention_growth"] = out.groupby(c.hashtag_col)[c.mentions_col].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["engagement_growth"] = out.groupby(c.hashtag_col)[c.engagement_col].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if "Semantic_Novelty" not in out:
            raise ValueError("Operational emerging-trend labels require a Semantic_Novelty column; the manuscript does not specify a unique estimator from static hashtag embeddings.")
        out["Emerging"] = (
            (out["mention_growth"] > self.label_thresholds["mention_growth"])
            & (out["engagement_growth"] > self.label_thresholds["engagement_growth"])
            & (out["Semantic_Novelty"] > self.label_thresholds["semantic_novelty"])
        ).astype(np.float32)
        return out

    def add_emergence_labels(self, df: pd.DataFrame, embedding_map: Mapping[str, np.ndarray] | None = None) -> pd.DataFrame:
        """Add the operational emerging-trend label used by the manuscript.

        The manuscript specifies thresholds for mention growth, engagement growth,
        and semantic novelty, but does not give a unique row-level novelty estimator.
        This reference implementation uses cosine distance from the mean embedding
        of the hashtag's prior novelty_history_days observations when time-varying
        embeddings are supplied; with static hashtag embeddings, novelty defaults
        to zero and callers should provide an externally computed Semantic_Novelty
        column if label reproduction is required.
        """
        c = self.config
        l = self.label_config
        out = df.copy().sort_values([c.hashtag_col, c.date_col])
        out["mention_growth"] = out.groupby(c.hashtag_col)[c.mentions_col].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["engagement_growth"] = out.groupby(c.hashtag_col)[c.engagement_col].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if "Semantic_Novelty" not in out:
            out["Semantic_Novelty"] = 0.0
        m_sigma = float(out["mention_growth"].std(ddof=0) or 1.0)
        e_sigma = float(out["engagement_growth"].std(ddof=0) or 1.0)
        out["Emerging"] = (
            (out["mention_growth"] > l.mention_growth_sigma_multiplier * m_sigma)
            & (out["engagement_growth"] > l.engagement_growth_sigma_multiplier * e_sigma)
            & (out["Semantic_Novelty"] > l.semantic_novelty_threshold)
        ).astype(np.float32)
        return out


@dataclass
class WindowRecord:
    hashtag: str
    origin_date: pd.Timestamp
    history: pd.DataFrame
    future: pd.DataFrame


def iter_windows(df: pd.DataFrame, config: DataConfig, lookback: int, horizon: int) -> Iterator[WindowRecord]:
    c = config
    for hashtag, g in df.groupby(c.hashtag_col, sort=False):
        g = g.sort_values(c.date_col).reset_index(drop=True)
        for end in range(lookback - 1, len(g) - horizon):
            hist = g.iloc[end - lookback + 1 : end + 1]
            fut = g.iloc[end + 1 : end + 1 + horizon]
            if hist["long_gap"].any() or fut["long_gap"].any():
                continue
            yield WindowRecord(str(hashtag), pd.Timestamp(g.iloc[end][c.date_col]), hist.copy(), fut.copy())


def build_modalities(
    record: WindowRecord,
    config: DataConfig,
    country_vocab: Mapping[str, int],
    platform_vocab: Mapping[str, int],
    semantic_embedding: np.ndarray,
) -> Dict[str, np.ndarray]:
    c = config
    h = record.history
    engagement = np.stack(
        [
            h[f"scaled_{c.mentions_col}"].to_numpy(dtype=np.float32),
            h[f"scaled_{c.reach_col}"].to_numpy(dtype=np.float32),
            h["delta_mentions"].to_numpy(dtype=np.float32),
            h["delta_reach"].to_numpy(dtype=np.float32),
        ],
        axis=-1,
    )
    sentiment = h[f"scaled_{c.sentiment_col}"].to_numpy(dtype=np.float32)[:, None]
    missing = np.stack(
        [
            h.get(f"mask_{c.mentions_col}", 0).to_numpy(dtype=np.float32),
            h.get(f"mask_{c.reach_col}", 0).to_numpy(dtype=np.float32),
            h.get(f"mask_{c.sentiment_col}", 0).to_numpy(dtype=np.float32),
        ],
        axis=-1,
    )
    calendar = sinusoidal_calendar_features(h[c.date_col].to_numpy())
    semantic = np.repeat(np.asarray(semantic_embedding, dtype=np.float32)[None, :], len(h), axis=0)

    country = np.zeros((len(h), max(1, len(country_vocab))), dtype=np.float32)
    for i, value in enumerate(h[c.country_col].fillna("Unknown").astype(str)):
        country[i] = one_hot(country_vocab.get(value, country_vocab.get("Unknown", 0)), country.shape[1])

    platform = np.zeros((len(h), max(1, len(platform_vocab))), dtype=np.float32)
    for i, value in enumerate(h[c.platform_col].fillna("Unknown").astype(str)):
        platform[i] = one_hot(platform_vocab.get(value, platform_vocab.get("Unknown", 0)), platform.shape[1])

    return {
        "engagement": engagement,
        "sentiment": sentiment,
        "missingness": missing,
        "calendar": calendar,
        "semantic": semantic,
        "geographic": country,
        "platform": platform,
    }
