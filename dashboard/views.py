"""dashboard/views.py - turn trajectories / sessions into tables and series the pages display."""

from typing import Dict, List

import numpy as np
import pandas as pd

import config as cfg
from dashboard import theme
from evaluation import analytics

TOTAL = "Total"


def color_map(roles: Dict[int, str]) -> Dict[str, str]:
    m = {role: theme.agent_color(role) for role in roles.values()}
    m[TOTAL] = theme.TOTAL_COLOR
    return m


def series_by_name(traj: List[dict], agent_ids, roles, caps, include_total=True) -> Dict[str, np.ndarray]:
    out = {roles[a]: analytics.value_series(traj, [a], caps[a]) for a in agent_ids}
    if include_total and len(agent_ids) > 1:
        out[TOTAL] = analytics.value_series(traj, agent_ids, sum(caps[a] for a in agent_ids))
    return out


def txcosts_by_name(traj, agent_ids, roles, include_total=True) -> Dict[str, np.ndarray]:
    out = {roles[a]: analytics.txcost_series(traj, [a]) for a in agent_ids}
    if include_total and len(agent_ids) > 1:
        out[TOTAL] = analytics.txcost_series(traj, agent_ids)
    return out


def metrics_frame(traj: List[dict], agent_ids, roles, caps) -> pd.DataFrame:
    """One row per agent (+ Total). Numeric fractions; format at display time."""
    vals = series_by_name(traj, agent_ids, roles, caps)
    costs = txcosts_by_name(traj, agent_ids, roles)
    rows = {}
    for name, v in vals.items():
        c = costs.get(name) if len(traj) else None
        m = analytics.portfolio_metrics(v, c)
        rows[name] = {
            "Final value": m["final_value"], "Total return": m["total_return"], "Volatility": m["volatility"],
            "Max drawdown": m["max_drawdown"], "Sharpe": m["sharpe"], "Transaction costs": m["total_txcost"],
            "Costs % of start": m["txcost_pct_of_initial"],
        }
    return pd.DataFrame(rows).T


def display_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Human-formatted copy of metrics_frame for st.dataframe."""
    out = pd.DataFrame(index=df.index)
    out["Final value"] = [theme.inr(x) for x in df["Final value"]]
    out["Total return"] = [theme.pct(x, 2, signed=True) for x in df["Total return"]]
    out["Volatility / step"] = [theme.pct(x, 3) for x in df["Volatility"]]
    out["Max drawdown"] = [theme.pct(x, 2) for x in df["Max drawdown"]]
    out["Sharpe (episode-scale)"] = [f"{x:.2f}" for x in df["Sharpe"]]
    out["Transaction costs"] = [theme.inr(x) for x in df["Transaction costs"]]
    return out


def weights_array(traj: List[dict], aid: int) -> np.ndarray:
    return np.array([step["actual_weights"][aid] for step in traj]) if traj else np.zeros((0, cfg.N_ASSETS))


def asset_pnl_total(traj: List[dict], agent_ids) -> np.ndarray:
    tot = np.zeros(cfg.N_ASSETS)
    for step in traj:
        for a in agent_ids:
            tot += step["asset_pnl"][a]
    return tot


def current_rupees(sim) -> Dict[str, np.ndarray]:
    """Rupee value held in each asset per agent right now (after the last step's returns)."""
    out = {}
    for a in sim.agent_ids:
        if sim.last is None:
            v = np.zeros(cfg.N_ASSETS)
            v[cfg.CASH] = sim.capitals[a]
        else:
            v = sim.last["asset_values"][a]
        out[sim.roles[a]] = np.asarray(v, dtype=float)
    return out


def split_series(series: Dict[str, np.ndarray]):
    """(per-agent sleeves, total-only) - sleeves start at Rs 2.5L, the total at Rs 10L, so they
    need separate y-axes (never a dual axis)."""
    sleeves = {k: v for k, v in series.items() if k != TOTAL}
    total = {k: v for k, v in series.items() if k == TOTAL}
    return sleeves, total


def sleeves_identical(series: Dict[str, np.ndarray], tol: float = 1e-6) -> bool:
    """True when every agent sleeve follows (numerically) the same path - their lines would overlap."""
    arrs = [v for k, v in series.items() if k != TOTAL]
    return len(arrs) > 1 and all(np.allclose(arrs[0], a, rtol=tol, atol=1e-6) for a in arrs[1:])
