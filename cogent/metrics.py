from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score, mean_absolute_error, mean_squared_error
from statsmodels.stats.multitest import multipletests


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def f1(y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> float:
    return float(f1_score(y_true.astype(int), (probability >= threshold).astype(int), zero_division=0))


def auc(y_true: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probability))


def best_f1_threshold(y_true: np.ndarray, probability: np.ndarray, grid: int = 201) -> float:
    thresholds = np.linspace(0.0, 1.0, grid)
    scores = [f1(y_true, probability, t) for t in thresholds]
    return float(thresholds[int(np.nanargmax(scores))])


def empirical_crps_np(samples: np.ndarray, target: np.ndarray) -> float:
    first = np.mean(np.abs(samples - target[None, :]), axis=0)
    pairwise = np.mean(np.abs(samples[:, None, :] - samples[None, :, :]), axis=(0, 1))
    return float(np.mean(first - 0.5 * pairwise))


def interval_metrics(samples: np.ndarray, target: np.ndarray, nominal: float = 0.90) -> Dict[str, float]:
    alpha = (1.0 - nominal) / 2.0
    lo = np.quantile(samples, alpha, axis=0)
    hi = np.quantile(samples, 1 - alpha, axis=0)
    coverage = float(np.mean((target >= lo) & (target <= hi)))
    width = float(np.mean(hi - lo))
    return {
        "coverage": coverage,
        "interval_width": width,
        "absolute_calibration_error": abs(coverage - nominal),
    }


def lead_time_gain(y_true: np.ndarray, probabilities: np.ndarray, threshold: float, peak_index: int | None = None) -> float:
    """Days between first threshold crossing and observed peak.

    The manuscript's campaign timing advantage is defined in these terms. This
    implementation assumes one sample per day unless the caller rescales it.
    """
    if peak_index is None:
        peak_index = int(np.argmax(y_true))
    crossings = np.where(probabilities >= threshold)[0]
    if len(crossings) == 0:
        return 0.0
    first = int(crossings[0])
    return float(max(0, peak_index - first))


def student_t_ci(values: Sequence[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    mean = float(np.mean(x))
    if len(x) < 2:
        return mean, mean, mean
    sem = stats.sem(x)
    q = stats.t.ppf((1 + confidence) / 2.0, df=len(x) - 1)
    return mean, float(mean - q * sem), float(mean + q * sem)


def paired_cohens_dz(a: Sequence[float], b: Sequence[float]) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = np.std(d, ddof=1)
    return float(np.mean(d) / sd) if sd > 1e-12 else float("inf") * np.sign(np.mean(d))


def rank_biserial_paired(a: Sequence[float], b: Sequence[float]) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    pos = ranks[d > 0].sum()
    neg = ranks[d < 0].sum()
    return float((pos - neg) / (pos + neg))


def paired_tests(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    t = stats.ttest_rel(a, b, nan_policy="omit")
    try:
        w = stats.wilcoxon(a, b, alternative="two-sided", method="exact")
        wp = float(w.pvalue)
    except ValueError:
        wp = float("nan")
    return {
        "paired_t_p": float(t.pvalue),
        "wilcoxon_exact_p": wp,
        "cohens_dz": paired_cohens_dz(a, b),
        "rank_biserial": rank_biserial_paired(a, b),
    }


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    valid = np.isfinite(p)
    out = np.full_like(p, np.nan)
    if valid.any():
        out[valid] = multipletests(p[valid], method="holm")[1]
    return out


def attribution_sparsity(attribution: np.ndarray, eps: float = 1e-12) -> float:
    x = np.abs(np.asarray(attribution, dtype=float)).reshape(-1)
    if x.sum() <= eps or len(x) <= 1:
        return 1.0
    p = x / x.sum()
    entropy = -np.sum(p * np.log(p + eps)) / np.log(len(p))
    return float(1.0 - entropy)


def attribution_stability(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cvar(losses: np.ndarray, beta: float = 0.90) -> float:
    x = np.asarray(losses, dtype=float).reshape(-1)
    q = np.quantile(x, beta)
    tail = x[x >= q]
    return float(np.mean(tail)) if len(tail) else float(q)
