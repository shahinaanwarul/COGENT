from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np

from .config import DecisionConfig
from .dataset import COGENTSample
from .gsism import GSISM, StructuralScenario, ScenarioResult
from .metrics import cvar


@dataclass
class CandidateAction:
    name: str
    scenario: StructuralScenario
    cost: float = 0.0


@dataclass
class ActionScore:
    action: CandidateAction
    utility: float
    discounted_reward: float
    discounted_cost: float
    discounted_risk: float
    scenario_result: ScenarioResult


class DAUOM:
    """Decision-aware utility ranking for a finite action set.

    The manuscript defines U(a) but does not specify a universal campaign reward,
    cost schedule, or the sign convention of CVaR. The defaults below use SSC as
    reward, caller-supplied action cost, and CVaR of negative forecast intensity as
    downside risk. All three can be replaced by callables.
    """

    def __init__(self, gsism: GSISM, config: DecisionConfig) -> None:
        self.gsism = gsism
        self.config = config

    def score_action(
        self,
        sample: COGENTSample,
        action: CandidateAction,
        reward_fn: Callable[[float], float] | None = None,
        risk_fn: Callable[[np.ndarray, float], float] | None = None,
        n_samples: int | None = None,
        seed: int = 42,
    ) -> ActionScore:
        reward_fn = reward_fn or (lambda x: x)
        risk_fn = risk_fn or (lambda ys, beta: cvar(-ys, beta))
        result = self.gsism.evaluate(sample, action.scenario, n_samples=n_samples, seed=seed)
        contrast = result.ssc.detach().cpu().numpy()
        trajectories = result.perturbed.forecast.trajectory_samples.detach().cpu().numpy()
        gamma = self.config.discount_gamma
        rewards = 0.0
        costs = 0.0
        risks = 0.0
        for t in range(contrast.shape[0]):
            discount = gamma ** t
            rewards += discount * float(reward_fn(float(contrast[t])))
            costs += discount * float(action.cost)
            risks += discount * float(risk_fn(trajectories[:, t], self.config.cvar_beta))
        utility = rewards - costs - self.config.risk_lambda * risks
        return ActionScore(action, utility, rewards, costs, risks, result)

    def rank(self, sample: COGENTSample, actions: Sequence[CandidateAction], **kwargs) -> List[ActionScore]:
        scored = [self.score_action(sample, action, **kwargs) for action in actions]
        return sorted(scored, key=lambda x: x.utility, reverse=True)
