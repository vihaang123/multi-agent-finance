"""
experiments/engine.py

A single, reproducible, UI-agnostic entry point for running the four
experiments of the brief (A Random, B Fixed, C Single-agent Q-learning,
D 4-agent Independent Q-learning) with:

  * TRAINING on one seed (config.train_seed) -- learning agents only;
  * OUT-OF-SAMPLE EVALUATION on a disjoint block of fresh seeds with
    epsilon=0, Q-tables frozen and no update() calls (reusing
    experiments.run_experiments.evaluate_out_of_sample unchanged).

Training and evaluation results are returned in separate fields and never
mixed. Everything is deterministic given the RunConfig (same config ->
same numbers). The Q-learning update, reward and environment maths are the
Baseline V1 ones -- this module only orchestrates them.

All data produced here comes from the SYNTHETIC market simulator in
environment/market.py. It is not real market data.
"""

from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import config as cfg
from agents.q_learning_agent import QLearningAgent
from environment.financial_env import FinancialEnv
from evaluation import analytics, metrics
from experiments.baseline import FixedAgent, RandomAgent
from experiments.run_experiments import AGENT_IDS, AGENT_ROLES, evaluate_out_of_sample
from training.trainer import Trainer

EXPERIMENT_NAMES = {
    "A": "A - Random allocation",
    "B": "B - Fixed basket",
    "C": "C - Single-agent Q-learning",
    "D": "D - 4-agent Independent Q-learning",
}
LEARNING_EXPERIMENTS = ("C", "D")
MAX_TRAIN_EPISODES = 20000   # hard safety cap for any caller


@dataclass
class QHyperParams:
    alpha: float = cfg.ALPHA_LR
    gamma: float = cfg.GAMMA
    epsilon_initial: float = cfg.EPSILON_INITIAL
    epsilon_min: float = cfg.EPSILON_MIN
    epsilon_decay: float = cfg.EPSILON_DECAY
    q_init: float = cfg.Q_INIT_VALUE

    def validate(self) -> List[str]:
        errs = []
        if not 0 < self.alpha <= 1:
            errs.append("Learning rate alpha must be in (0, 1].")
        if not 0 <= self.gamma <= 1:
            errs.append("Discount factor gamma must be in [0, 1].")
        if not 0 <= self.epsilon_min <= self.epsilon_initial <= 1:
            errs.append("Need 0 <= epsilon_min <= epsilon_initial <= 1.")
        if not 0 < self.epsilon_decay <= 1:
            errs.append("epsilon_decay must be in (0, 1].")
        if not np.isfinite(self.q_init):
            errs.append("Initial Q-value must be a finite number.")
        return errs


@dataclass
class RunConfig:
    experiment: str = "D"
    n_train_episodes: int = 1000
    train_seed: int = cfg.TRAIN_SEED
    eval_seed_start: int = cfg.EVAL_SEED_START
    n_eval_episodes: int = cfg.N_EVAL_EPISODES
    episode_length: int = cfg.EPISODE_LENGTH
    fixed_action: int = 2
    enable_market_impact: bool = True
    reward_mode: str = cfg.REWARD_MODE
    hp: QHyperParams = field(default_factory=QHyperParams)

    @property
    def eval_seeds(self) -> List[int]:
        return list(range(self.eval_seed_start, self.eval_seed_start + self.n_eval_episodes))

    def validate(self) -> List[str]:
        errs = []
        if self.experiment not in EXPERIMENT_NAMES:
            errs.append(f"Unknown experiment '{self.experiment}'.")
        if not 1 <= self.episode_length <= 1000:
            errs.append("Episode length must be between 1 and 1000 steps.")
        if self.experiment in LEARNING_EXPERIMENTS and not 1 <= self.n_train_episodes <= MAX_TRAIN_EPISODES:
            errs.append(f"Training episodes must be between 1 and {MAX_TRAIN_EPISODES}.")
        if not 1 <= self.n_eval_episodes <= 500:
            errs.append("Number of evaluation episodes must be between 1 and 500.")
        if self.train_seed in self.eval_seeds and self.experiment in LEARNING_EXPERIMENTS:
            errs.append(f"Evaluation seeds {self.eval_seeds[0]}-{self.eval_seeds[-1]} include the training seed "
                        f"{self.train_seed}: that would leak training data into evaluation. Choose a disjoint block.")
        if not 0 <= self.fixed_action < cfg.N_ACTIONS:
            errs.append(f"Fixed basket index must be in [0, {cfg.N_ACTIONS}).")
        if self.reward_mode not in ("individual", "shared", "hybrid"):
            errs.append("reward_mode must be individual, shared or hybrid.")
        errs += self.hp.validate()
        return errs

    def check(self):
        errs = self.validate()
        if errs:
            raise ValueError(" ".join(errs))

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# --------------------------------------------------------------------------- #
# Construction helpers (also used by the RL Lab page)
# --------------------------------------------------------------------------- #
def layout_for(experiment: str):
    """(agent_ids, roles, initial_capitals) for an experiment."""
    if experiment == "C":
        return [0], {0: "Growth"}, {0: cfg.TOTAL_INITIAL_CAPITAL}
    return (list(AGENT_IDS), dict(AGENT_ROLES),
            {aid: cfg.INITIAL_CAPITAL_PER_AGENT for aid in AGENT_IDS})


def build_q_agents(agent_ids, roles, hp: QHyperParams, seed: int) -> Dict[int, QLearningAgent]:
    return {aid: QLearningAgent(aid, roles[aid], alpha=hp.alpha, gamma=hp.gamma,
                                epsilon_initial=hp.epsilon_initial, epsilon_min=hp.epsilon_min,
                                epsilon_decay=hp.epsilon_decay, q_init=hp.q_init, seed=seed + aid)
            for aid in agent_ids}


def build_agents(rc: RunConfig):
    agent_ids, roles, caps = layout_for(rc.experiment)
    if rc.experiment == "A":
        agents = {aid: RandomAgent(aid, roles[aid], seed=rc.train_seed + aid) for aid in agent_ids}
    elif rc.experiment == "B":
        agents = {aid: FixedAgent(aid, roles[aid], fixed_action=rc.fixed_action) for aid in agent_ids}
    else:
        agents = build_q_agents(agent_ids, roles, rc.hp, rc.train_seed)
    return agent_ids, roles, caps, agents


def make_env(agent_ids, roles, caps, seed, episode_length, reward_mode=cfg.REWARD_MODE, enable_market_impact=True):
    return FinancialEnv(agent_ids=agent_ids, agent_roles=roles, initial_capitals=caps,
                        episode_length=episode_length, reward_mode=reward_mode,
                        enable_market_impact=enable_market_impact, seed=seed)


# --------------------------------------------------------------------------- #
# Evaluation summary (out-of-sample only)
# --------------------------------------------------------------------------- #
def summarize_evaluation(trajectories: List[List[dict]], agent_ids, caps) -> dict:
    total_initial = float(sum(caps.values()))
    per_agent_metrics = {aid: [] for aid in agent_ids}
    combined_metrics = []
    curves = {aid: [] for aid in agent_ids}
    combined_curves = []
    action_counts = {aid: np.zeros(cfg.N_ACTIONS) for aid in agent_ids}

    for traj in trajectories:
        for aid in agent_ids:
            vs = analytics.value_series(traj, [aid], caps[aid])
            per_agent_metrics[aid].append(analytics.portfolio_metrics(vs, analytics.txcost_series(traj, [aid])))
            curves[aid].append(vs)
            for step in traj:
                action_counts[aid][step["actions"][aid]] += 1
        cv = analytics.value_series(traj, agent_ids, total_initial)
        combined_metrics.append(analytics.portfolio_metrics(cv, analytics.txcost_series(traj, agent_ids)))
        combined_curves.append(cv)

    hists = {aid: (action_counts[aid] / action_counts[aid].sum()) if action_counts[aid].sum() > 0
             else action_counts[aid] for aid in agent_ids}
    summary = {
        "per_agent": {int(aid): analytics.summarize_over_seeds(per_agent_metrics[aid]) for aid in agent_ids},
        "combined": analytics.summarize_over_seeds(combined_metrics),
        "combined_returns_per_seed": [float(m["total_return"]) for m in combined_metrics],
        "per_agent_returns_per_seed": {int(aid): [float(m["total_return"]) for m in per_agent_metrics[aid]]
                                       for aid in agent_ids},
        "mean_equity_curve": np.mean(combined_curves, axis=0).tolist(),
        "std_equity_curve": np.std(combined_curves, axis=0).tolist(),
        "per_agent_mean_curve": {int(aid): np.mean(curves[aid], axis=0).tolist() for aid in agent_ids},
        "action_hist": {int(aid): hists[aid].tolist() for aid in agent_ids},
        "modal_action": {int(aid): int(np.argmax(hists[aid])) for aid in agent_ids},
        "specialization": metrics.specialization_score(hists),
        "utilisation": float(np.mean([c[-1] for c in combined_curves]) / total_initial),
        "total_initial_capital": total_initial,
    }
    return summary


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_experiment(rc: RunConfig, progress_callback: Optional[Callable] = None) -> dict:
    """Train (learning experiments only) then evaluate out-of-sample.

    Returns a dict with: config, agents, training_df (None for A/B),
    eval_trajectories, summary (JSON-serialisable evaluation stats).
    """
    rc.check()
    agent_ids, roles, caps, agents = build_agents(rc)

    training_df = None
    if rc.experiment in LEARNING_EXPERIMENTS:
        env = make_env(agent_ids, roles, caps, rc.train_seed, rc.episode_length,
                       rc.reward_mode, rc.enable_market_impact)
        training_df = Trainer(env, agents).train(
            num_episodes=rc.n_train_episodes, verbose=False, progress_callback=progress_callback)

    trajectories = evaluate_out_of_sample(
        agent_ids, roles, caps, agents, eval_seeds=rc.eval_seeds, episode_length=rc.episode_length,
        reward_mode=rc.reward_mode, enable_market_impact=rc.enable_market_impact)
    summary = summarize_evaluation(trajectories, agent_ids, caps)

    return {"config": rc, "agent_ids": agent_ids, "roles": roles, "initial_capitals": caps,
            "agents": agents, "training_df": training_df,
            "eval_trajectories": trajectories, "summary": summary}


def evaluate_agents(agents, agent_ids, roles, caps, eval_seeds, episode_length=cfg.EPISODE_LENGTH,
                    reward_mode=cfg.REWARD_MODE, enable_market_impact=True) -> dict:
    """Out-of-sample evaluation of ALREADY-TRAINED agents (RL Lab / Analytics)."""
    trajectories = evaluate_out_of_sample(agent_ids, roles, caps, agents, eval_seeds=eval_seeds,
                                          episode_length=episode_length, reward_mode=reward_mode,
                                          enable_market_impact=enable_market_impact)
    return {"eval_trajectories": trajectories, "summary": summarize_evaluation(trajectories, agent_ids, caps)}


def training_curves(df: pd.DataFrame, agent_ids, window: int = 50, every: int = 25) -> dict:
    """Down-sampled (moving-average) training curves as plain lists -> JSON friendly."""
    def curve(col, w):
        sm = df[col].rolling(w, min_periods=1).mean()
        idx = list(range(0, len(sm), every))
        if idx[-1] != len(sm) - 1:
            idx.append(len(sm) - 1)
        return {"episode": [int(df["episode"].iloc[i]) for i in idx], "value": [float(sm.iloc[i]) for i in idx]}
    return {
        "reward": {str(a): curve(f"reward_{a}", window) for a in agent_ids},
        "epsilon": {str(a): curve(f"epsilon_{a}", 1) for a in agent_ids},
        "td": {str(a): curve(f"td_{a}", window) for a in agent_ids},
        "capital": {str(a): curve(f"capital_{a}", window) for a in agent_ids},
        "total_capital": curve("total_capital", window),
    }
