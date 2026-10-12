"""
visualization/plots.py

All plotting lives here, built on plain matplotlib. Every function saves
a PNG to save_path and also returns the Figure (so it can be reused in
notebooks or shown inline) but does NOT call plt.show() -- caller decides.
"""

import os
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg

_COLORS = {0: "#1f77b4", 1: "#2ca02c", 2: "#ff7f0e", 3: "#d62728"}


def _color(agent_id):
    return _COLORS.get(agent_id, None)


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


# --------------------------------------------------------------------------- #
# Plot 1: Average episode reward over training
# --------------------------------------------------------------------------- #
def plot_training_rewards(training_df, agent_ids: List[int], agent_roles: Dict[int, str], save_path: str, window: int = 100):
    fig, ax = plt.subplots(figsize=(9, 5))
    for aid in agent_ids:
        col = f"reward_{aid}"
        if col not in training_df.columns:
            continue
        rolling = training_df[col].rolling(window, min_periods=1).mean()
        ax.plot(training_df["episode"], rolling, label=f"Agent {aid} ({agent_roles.get(aid, '')})", color=_color(aid))
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward (rolling mean, window={window})")
    ax.set_title("Average Episode Reward over Training")
    ax.legend()
    ax.grid(alpha=0.3)
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# Plot 2: Portfolio value over time (within a single evaluation episode)
# --------------------------------------------------------------------------- #
def plot_portfolio_value(trajectory: list, agent_ids: List[int], agent_roles: Dict[int, str], save_path: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    for aid in agent_ids:
        values = [step["capital"][aid] for step in trajectory]
        ax.plot(range(len(values)), values, label=f"Agent {aid} ({agent_roles.get(aid, '')})", color=_color(aid))
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Portfolio Value (Rs)")
    ax.set_title("Portfolio Value over Time (evaluation episode)")
    ax.legend()
    ax.grid(alpha=0.3)
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# Plot 3: Asset allocation over time
# --------------------------------------------------------------------------- #
def plot_allocation_over_time(trajectory: list, agent_id: int, role: str, save_path: str):
    weights = np.array([step["actual_weights"][agent_id] for step in trajectory])  # (T, 4)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(
        range(len(weights)),
        weights[:, cfg.EQUITY], weights[:, cfg.BONDS], weights[:, cfg.GOLD], weights[:, cfg.CASH],
        labels=cfg.ASSET_NAMES,
        colors=["#4c72b0", "#55a868", "#c44e52", "#8172b2"],
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Allocation Weight")
    ax.set_title(f"Asset Allocation over Time -- Agent {agent_id} ({role})")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# Plot 4: Agent comparison (final performance bar chart)
# --------------------------------------------------------------------------- #
def plot_agent_comparison(final_capitals: Dict[int, float], initial_capitals: Dict[int, float], agent_roles: Dict[int, str], save_path: str):
    agent_ids = list(final_capitals.keys())
    returns_pct = [100.0 * (final_capitals[aid] - initial_capitals[aid]) / initial_capitals[aid] for aid in agent_ids]
    labels = [f"Agent {aid}\n({agent_roles.get(aid, '')})" for aid in agent_ids]
    colors = [_color(aid) for aid in agent_ids]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, returns_pct, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Total Return (%)")
    ax.set_title("Agent Comparison -- Final Return by Role")
    for bar, val in zip(bars, returns_pct):
        ax.annotate(f"{val:.1f}%", (bar.get_x() + bar.get_width() / 2, val),
                     textcoords="offset points", xytext=(0, 4 if val >= 0 else -14), ha="center")
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# Plot 5: Market regime vs allocation
# --------------------------------------------------------------------------- #
def plot_regime_vs_allocation(trajectory: list, agent_ids: List[int], agent_roles: Dict[int, str], save_path: str):
    """Buckets each step's allocation by the regime that was ACTUALLY active
    when that allocation earned its return (trajectory["current_regime"]),
    not the regime the market transitioned into afterward. An earlier
    version used a single "regime" field that was actually the POST-
    transition regime, mislabeling this plot by one step. See README,
    Results & Analysis."""
    regime_names = cfg.REGIME_NAMES
    n_regimes = cfg.N_REGIMES
    avg_equity_by_regime = {aid: np.zeros(n_regimes) for aid in agent_ids}
    counts_by_regime = np.zeros(n_regimes)

    for step in trajectory:
        r = step["current_regime"]
        counts_by_regime[r] += 1
        for aid in agent_ids:
            avg_equity_by_regime[aid][r] += step["actual_weights"][aid][cfg.EQUITY]

    for aid in agent_ids:
        with np.errstate(invalid="ignore", divide="ignore"):
            avg_equity_by_regime[aid] = np.divide(
                avg_equity_by_regime[aid], counts_by_regime,
                out=np.zeros(n_regimes), where=counts_by_regime > 0,
            )

    x = np.arange(n_regimes)
    width = 0.8 / max(1, len(agent_ids))
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, aid in enumerate(agent_ids):
        ax.bar(x + i * width, avg_equity_by_regime[aid], width=width,
               label=f"Agent {aid} ({agent_roles.get(aid, '')})", color=_color(aid))
    ax.set_xticks(x + width * (len(agent_ids) - 1) / 2)
    ax.set_xticklabels(regime_names)
    ax.set_ylabel("Average Equity Allocation")
    ax.set_title("Market Regime vs Equity Allocation")
    ax.legend()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# Plot 6: Drawdown
# --------------------------------------------------------------------------- #
def plot_drawdown(trajectory: list, agent_ids: List[int], agent_roles: Dict[int, str], save_path: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    for aid in agent_ids:
        dd = [step["drawdown"][aid] * 100 for step in trajectory]
        ax.plot(range(len(dd)), dd, label=f"Agent {aid} ({agent_roles.get(aid, '')})", color=_color(aid))
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown over Time (evaluation episode)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.invert_yaxis()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# Plot 7: Risk vs return
# --------------------------------------------------------------------------- #
def plot_risk_vs_return(training_df, agent_ids: List[int], agent_roles: Dict[int, str], save_path: str, window: int = 100):
    fig, ax = plt.subplots(figsize=(7, 6))
    for aid in agent_ids:
        col = f"reward_{aid}"
        cap_col = f"capital_{aid}"
        if col not in training_df.columns or cap_col not in training_df.columns:
            continue
        ep_returns = training_df[cap_col].pct_change().fillna(0.0)
        rolling_mean_return = ep_returns.rolling(window, min_periods=1).mean()
        rolling_vol = ep_returns.rolling(window, min_periods=1).std().fillna(0.0)
        ax.scatter(rolling_vol, rolling_mean_return, s=8, alpha=0.5,
                    label=f"Agent {aid} ({agent_roles.get(aid, '')})", color=_color(aid))
    ax.set_xlabel("Rolling Volatility of Episode Return")
    ax.set_ylabel("Rolling Mean Episode Return")
    ax.set_title("Risk vs Return (across training)")
    ax.legend()
    ax.grid(alpha=0.3)
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
