"""
evaluation/analytics.py

Pure-numpy portfolio analytics computed from ACTUAL simulation output
(trajectories / value series). No Streamlit, no plotting -- shared by the
experiment engine and the dashboard so both report identical numbers.

Conventions (kept consistent with evaluation/metrics.py):
  * a "value series" includes the initial capital at t=0, so it has
    episode_length + 1 points and total_return measures the WHOLE episode
    (including the first rebalance's transaction cost);
  * volatility = std of per-step simple returns (per-step, not annualised);
  * Sharpe  = EPISODE-SCALE Sharpe from metrics.sharpe_ratio (not annualised);
  * max drawdown = largest peak-to-trough fall of the value series.

NOTE: Baseline V1's experiment printout (experiments/run_experiments.py)
computed its return from the first post-step capital onward; the series
here start from the true initial capital, so figures can differ from that
printout by a small amount (the first step's cost / return).
"""

from typing import Dict, Iterable, List, Optional

import numpy as np

import config as cfg
from evaluation import metrics


# --------------------------------------------------------------------------- #
# Series builders
# --------------------------------------------------------------------------- #
def value_series(trajectory: List[dict], agent_ids: Iterable[int], initial_capital: float) -> np.ndarray:
    """Portfolio value of the SUM of the given agents' sleeves, t=0..T."""
    ids = list(agent_ids)
    vals = [float(initial_capital)]
    for step in trajectory:
        vals.append(float(sum(step["capital"][a] for a in ids)))
    return np.asarray(vals)


def txcost_series(trajectory: List[dict], agent_ids: Iterable[int]) -> np.ndarray:
    """Rupee transaction cost paid at each step (summed over the given agents)."""
    ids = list(agent_ids)
    return np.asarray([float(sum(step["txcost"][a] for a in ids)) for step in trajectory])


def cumulative_returns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values / values[0] - 1.0 if len(values) and values[0] > 0 else np.zeros_like(values)


def step_returns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.zeros(0)
    return np.diff(values) / values[:-1]


def drawdown_series(values: np.ndarray) -> np.ndarray:
    """Drawdown from running peak at each t, as a positive fraction."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    peak = np.maximum.accumulate(values)
    return (peak - values) / np.where(peak == 0, 1, peak)


def rolling_volatility(values: np.ndarray, window: int = 10) -> np.ndarray:
    r = step_returns(values)
    out = np.full(len(r), np.nan)
    for i in range(window - 1, len(r)):
        out[i] = np.std(r[i - window + 1:i + 1])
    return out


# --------------------------------------------------------------------------- #
# Scalar metrics
# --------------------------------------------------------------------------- #
def portfolio_metrics(values: np.ndarray, txcosts: Optional[np.ndarray] = None) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return {"initial_value": float(values[0]) if len(values) else 0.0,
                "final_value": float(values[-1]) if len(values) else 0.0,
                "total_return": 0.0, "volatility": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
                "total_txcost": 0.0, "txcost_pct_of_initial": 0.0, "n_steps": 0}
    total_tx = float(np.sum(txcosts)) if txcosts is not None else 0.0
    return {
        "initial_value": float(values[0]),
        "final_value": float(values[-1]),
        "total_return": metrics.total_return(values[0], values[-1]),
        "volatility": metrics.volatility_of_returns(values),
        "max_drawdown": metrics.max_drawdown(values),
        "sharpe": metrics.sharpe_ratio(values),
        "total_txcost": total_tx,
        "txcost_pct_of_initial": total_tx / values[0] if values[0] > 0 else 0.0,
        "n_steps": int(len(values) - 1),
    }


def action_histogram(trajectory: List[dict], agent_id: int) -> np.ndarray:
    counts = np.zeros(cfg.N_ACTIONS)
    for step in trajectory:
        counts[step["actions"][agent_id]] += 1
    return counts / counts.sum() if counts.sum() > 0 else counts


def regime_allocation(trajectory: List[dict], agent_id: int) -> np.ndarray:
    """Mean TARGET weights chosen per regime (rows: regimes, cols: assets).
    Keyed on current_regime -- the regime that generated the step's return."""
    from collections import defaultdict
    acc = defaultdict(list)
    for step in trajectory:
        acc[step["current_regime"]].append(cfg.ACTION_BASKETS[step["actions"][agent_id]])
    out = np.full((len(cfg.REGIME_NAMES), cfg.N_ASSETS), np.nan)
    for r, rows in acc.items():
        out[r] = np.mean(rows, axis=0)
    return out


# --------------------------------------------------------------------------- #
# Aggregation over many held-out episodes
# --------------------------------------------------------------------------- #
def summarize_over_seeds(per_seed_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    keys = ["final_value", "total_return", "volatility", "max_drawdown", "sharpe", "total_txcost"]
    out = {}
    for k in keys:
        arr = np.asarray([m[k] for m in per_seed_metrics], dtype=float)
        out[f"{k}_mean"] = float(arr.mean())
        out[f"{k}_std"] = float(arr.std())
    arr = np.asarray([m["total_return"] for m in per_seed_metrics], dtype=float)
    out["total_return_min"] = float(arr.min())
    out["total_return_max"] = float(arr.max())
    out["prob_loss"] = float(np.mean(arr < 0))
    out["n_episodes"] = len(per_seed_metrics)
    return out
