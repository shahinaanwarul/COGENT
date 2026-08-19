from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

from .dataset import COGENTWindowDataset
from .gsism import GSISM, StructuralScenario
from .model import COGENTModel


MANUSCRIPT_CAMPAIGNS = (
    "amazongreatindianfestival",
    "kuchmeethahojaaye",
    "shotoniphone",
    "spotifywrapped",
    "sharetheload",
)


@dataclass
class CampaignForecast:
    hashtag: str
    origin_date: str
    forecast: np.ndarray
    emergence_probability: np.ndarray


@torch.no_grad()
def campaign_forecasts(
    model: COGENTModel,
    dataset: COGENTWindowDataset,
    device: torch.device | str,
    hashtags: Sequence[str] = MANUSCRIPT_CAMPAIGNS,
) -> List[CampaignForecast]:
    model.eval()
    wanted = {h.lower().lstrip("#") for h in hashtags}
    latest = {}
    for i, rec in enumerate(dataset.records):
        if rec.hashtag.lower().lstrip("#") in wanted:
            latest[rec.hashtag] = i
    out: List[CampaignForecast] = []
    for hashtag, idx in latest.items():
        sample = dataset[idx].to(device)
        result = model(sample, n_samples=model.config.model.monte_carlo_samples_eval)
        out.append(
            CampaignForecast(
                hashtag=hashtag,
                origin_date=str(sample.origin_date.date()),
                forecast=result.forecast.mean_trajectory.cpu().numpy(),
                emergence_probability=result.forecast.mean_emergence_probability.cpu().numpy(),
            )
        )
    return out


def manuscript_style_scenarios() -> Dict[str, StructuralScenario]:
    """Generic bounded scenarios matching the manuscript's structural categories.

    Exact action-to-edge mappings are campaign-specific and are not specified in
    the manuscript, so callers should refine the relation and edge list for their
    study design before interpreting a scenario.
    """
    return {
        "influencer_style_amplification": StructuralScenario("influencer_style_amplification", "sim", 0.20, "multiply"),
        "cross_platform_seeding": StructuralScenario("cross_platform_seeding", "plat", 0.20, "multiply"),
        "sentiment_structure_perturbation": StructuralScenario("sentiment_structure_perturbation", "sent", 0.20, "multiply"),
        "reduced_visibility": StructuralScenario("reduced_visibility", "sim", -0.20, "multiply"),
        "suppression_style_attenuation": StructuralScenario("suppression_style_attenuation", "geo", -0.20, "multiply"),
    }
