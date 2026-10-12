"""
evaluation/metrics.py

Reusable metric functions across three categories:
  - Financial: final value, total/avg return, volatility, max drawdown, Sharpe
  - RL: average reward, convergence behaviour, exploration rate
  - MAS: allocation diversity across agents, resource utilisation, specialization
"""

from typing import Dict, List

import numpy as np
import pandas as pd

import config as cfg


# --------------------------------------------------------------------------- #
# Financial metrics
# --------------------------------------------------------------------------- #
def total_return(initial_value: float, final_value: float) -> float:
    if initial_value <= 0:
        return 0.0
    return (final_value - initial_value) / initial_value


def max_drawdown(value_series: List[float]) -> float:
    values = np.asarray(value_series, dtype=float)
    if len(values) == 0:
        return 0.0
    running_peak = np.maximum.accumulate(values)
    drawdowns = (running_peak - values) / np.where(running_peak == 0, 1, running_peak)
    return float(np.max(drawdowns))


def volatility_of_returns(value_series: List[float]) -> float:
    values = np.asarray(value_series, dtype=float)
    if len(values) < 2:
        return 0.0
    step_returns = np.diff(values) / values[:-1]
    return float(np.std(step_returns))


def sharpe_ratio(value_series: List[float], risk_free_per_step: float = None) -> float:
    """
    EPISODE-SCALE Sharpe ratio -- NOT an annualized Sharpe ratio, and should
    never be reported or labeled as one. Full assumptions, explicit:

      - Return frequency: per-TIMESTEP (one value per environment step, i.e.
        one per day-equivalent in this simulation -- NOT daily/monthly/yearly
        in any calendar sense).
      - Risk-free assumption: defaults to this project's own modeled CASH
        return (`cfg.ASSET_RETURN_PARAMS[NEUTRAL][CASH][0]`, currently
        0.0002/step), read from config rather than duplicated here, so it
        can never silently drift out of sync with the market model.
      - Scaling factor: multiplied by sqrt(number of steps in the series),
        e.g. sqrt(100) for one full episode. This rescales a PER-STEP Sharpe
        up to an EPISODE-SCALE Sharpe -- structurally the same maths as
        calendar annualization (which multiplies by sqrt(periods/year)), but
        the multiplier here is sqrt(steps/episode), which has no calendar
        meaning. Do not call the result "annualized".
      - Interpretation: valid for comparing runs that share the same
        EPISODE_LENGTH (all experiments in this project do). NOT comparable
        to Sharpe ratios reported elsewhere in finance without renormalizing
        by the actual number of real-world periods per year.
    """
    if risk_free_per_step is None:
        risk_free_per_step = cfg.ASSET_RETURN_PARAMS[cfg.NEUTRAL][cfg.CASH][0]
    values = np.asarray(value_series, dtype=float)
    if len(values) < 2:
        return 0.0
    step_returns = np.diff(values) / values[:-1]
    excess = step_returns - risk_free_per_step
    std = np.std(excess)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(len(step_returns)))


SHARPE_METHODOLOGY_NOTE = (
    "Sharpe figures in this project are EPISODE-SCALE, not annualized: "
    "computed from per-step returns against a per-step risk-free rate "
    f"({cfg.ASSET_RETURN_PARAMS[cfg.NEUTRAL][cfg.CASH][0]}/step, this "
    "project's own modeled cash return), scaled by sqrt(episode_length) "
    f"(sqrt({cfg.EPISODE_LENGTH}) here). Comparable across experiments in "
    "this project (all share EPISODE_LENGTH); not comparable to a "
    "conventional calendar-annualized Sharpe ratio."
)


def financial_summary(value_series: List[float]) -> dict:
    values = np.asarray(value_series, dtype=float)
    return {
        "final_value": float(values[-1]) if len(values) else 0.0,
        "total_return": total_return(values[0], values[-1]) if len(values) else 0.0,
        "volatility": volatility_of_returns(values),
        "max_drawdown": max_drawdown(values),
        "sharpe": sharpe_ratio(values),
    }


# --------------------------------------------------------------------------- #
# RL metrics
# --------------------------------------------------------------------------- #
def average_reward(training_df: pd.DataFrame, agent_id: int, window: int = 100) -> pd.Series:
    return training_df[f"reward_{agent_id}"].rolling(window, min_periods=1).mean()


def convergence_gap(training_df: pd.DataFrame, agent_id: int, tail_frac: float = 0.1) -> float:
    """
    Rough convergence diagnostic: difference between the mean reward in the
    final tail_frac of training vs the tail_frac window right before it.
    Values near 0 suggest the policy has stabilised.
    """
    n = len(training_df)
    tail = max(1, int(n * tail_frac))
    if n < 2 * tail:
        return float("nan")
    col = f"reward_{agent_id}"
    last = training_df[col].iloc[-tail:].mean()
    prev = training_df[col].iloc[-2 * tail:-tail].mean()
    return float(last - prev)


def exploration_rate_series(training_df: pd.DataFrame, agent_id: int) -> pd.Series:
    return training_df[f"epsilon_{agent_id}"]


def rl_training_summary(training_df: pd.DataFrame, agent_id: int, window: int = 100) -> dict:
    """
    Aggregates the three RL metrics (average_reward, convergence_gap,
    exploration_rate_series) into one clearly-defined summary for reporting.
    Definitions (so these numbers mean something specific, not just "more
    numbers"):

      - avg_reward_final_window: rolling mean of episode reward with the
        given window, read at the LAST training episode. "How good was the
        policy by the end of training, smoothed over the last `window`
        episodes."
      - reward_trend: mean reward in the final 10% of training minus mean
        reward in the 10% immediately before that. Near 0 = training reward
        has plateaued (stabilised); still clearly positive/negative = still
        improving/degrading when training stopped.
      - final_epsilon: the exploration rate (epsilon) at the last training
        episode -- 0 for non-learning baselines.
      - episode_epsilon_floor_reached: the first training episode at which
        epsilon reached its configured floor (EPSILON_MIN), or None if it
        never did within this run.
    """
    col_eps = f"epsilon_{agent_id}"
    avg_reward_final_window = float(average_reward(training_df, agent_id, window).iloc[-1])
    reward_trend = convergence_gap(training_df, agent_id)
    final_epsilon = float(training_df[col_eps].iloc[-1]) if col_eps in training_df.columns else 0.0

    floor_episode = None
    if col_eps in training_df.columns:
        at_floor = training_df[col_eps] <= (training_df[col_eps].min() + 1e-9)
        if at_floor.any():
            floor_episode = int(training_df.loc[at_floor, "episode"].iloc[0])

    return {
        "avg_reward_final_window": avg_reward_final_window,
        "reward_trend": reward_trend,
        "final_epsilon": final_epsilon,
        "episode_epsilon_floor_reached": floor_episode,
    }


# --------------------------------------------------------------------------- #
# MAS metrics
# --------------------------------------------------------------------------- #
def allocation_distribution(action_log: List[Dict[int, int]], agent_id: int) -> np.ndarray:
    """Given a list of {agent_id: action} dicts across steps, return the
    normalised histogram of basket choices for one agent."""
    counts = np.zeros(cfg.N_ACTIONS)
    for step_actions in action_log:
        counts[step_actions[agent_id]] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def specialization_score(agent_action_histograms: Dict[int, np.ndarray]) -> float:
    """
    Mean pairwise total-variation distance between agents' basket-choice
    histograms. 0 = every agent behaves identically (no specialization).
    Higher = agents have visibly different, role-driven policies.
    """
    ids = list(agent_action_histograms.keys())
    if len(ids) < 2:
        return 0.0
    dists = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            hi, hj = agent_action_histograms[ids[i]], agent_action_histograms[ids[j]]
            dists.append(0.5 * np.sum(np.abs(hi - hj)))
    return float(np.mean(dists))


def resource_utilisation(final_capitals: Dict[int, float], initial_capitals: Dict[int, float]) -> float:
    """Fraction of total starting capital retained across all agents combined."""
    total_initial = sum(initial_capitals.values())
    total_final = sum(final_capitals.values())
    if total_initial <= 0:
        return 0.0
    return total_final / total_initial
