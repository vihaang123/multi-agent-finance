"""
dashboard/charts.py - Plotly figure builders. Pure functions: data in -> go.Figure out.
No Streamlit calls here, so they are unit-testable. Every figure is drawn from
actual simulation / training output passed in by the caller.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as cfg
from dashboard import theme
from environment.state_utils import BUCKET_SHORT, decode_state
from evaluation import analytics

FONT = dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif", size=12, color=theme.INK_2)


def _base(fig: go.Figure, height: int = 340, title: str = "", legend: bool = True, ytitle: str = "",
          xtitle: str = "") -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=46 if title else 14, b=(92 if legend else 8)), font=FONT,
        title=dict(text=title, x=0, xanchor="left", y=0.985, yanchor="top", yref="container",
                   font=dict(size=14, color=theme.INK)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
        showlegend=legend, legend=dict(orientation="h", yref="container", y=0.0, yanchor="bottom", x=0, xanchor="left", font=dict(size=11)),
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor=theme.GRID, zeroline=False, title=xtitle, ticks="outside",
                     tickcolor=theme.GRID)
    fig.update_yaxes(gridcolor=theme.GRID, zeroline=False, title=ytitle, automargin=True)
    return fig


def _empty(msg: str, height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=13, color=theme.INK_3))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base(fig, height=height, legend=False)


def add_regime_bands(fig: go.Figure, regimes: Sequence[int], x0: int = 1, row=None, col=None, legend=True):
    """Shade contiguous runs of the same market regime; x of regimes[i] is x0+i.

    The rectangles are added in ONE update (not one add_vrect call per run): validating dozens of shapes one at a
    time dominated the API's response time."""
    if len(regimes) == 0:
        return fig
    start, shapes = 0, []
    kw = {} if row is None else dict(row=row, col=col)
    xref, yref = ("x", "y domain") if row is None else (None, None)
    if row is not None:
        sp = fig.get_subplot(row, col)
        xref = sp.xaxis.plotly_name.replace("axis", "")
        yref = sp.yaxis.plotly_name.replace("axis", "") + " domain"
    for i in range(1, len(regimes) + 1):
        if i == len(regimes) or regimes[i] != regimes[start]:
            name = cfg.REGIME_NAMES[regimes[start]]
            shapes.append(dict(type="rect", xref=xref, yref=yref, x0=x0 + start - 0.5, x1=x0 + i - 0.5, y0=0, y1=1,
                               fillcolor=theme.REGIME_COLORS[name], opacity=0.10, line=dict(width=0), layer="below"))
            start = i
    fig.update_layout(shapes=list(fig.layout.shapes) + shapes)
    if legend:
        for name in cfg.REGIME_NAMES:
            if cfg.REGIME_NAMES.index(name) in set(regimes):
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                         marker=dict(size=10, symbol="square", color=theme.REGIME_COLORS[name], opacity=0.45),
                                         hoverinfo="skip"), **kw)
    return fig


# --------------------------------------------------------------------------- #
# Portfolio value / returns / risk
# --------------------------------------------------------------------------- #
def portfolio_value(series: Dict[str, np.ndarray], colors: Dict[str, str], regimes: Sequence[int] = (),
                    initial: Optional[float] = None, title: str = "Portfolio value (Rs)", height: int = 380,
                    total_name: str = "Total") -> go.Figure:
    fig = go.Figure()
    if not series or all(len(v) < 2 for v in series.values()):
        return _empty("Run the simulation to see portfolio value over time.", height)
    add_regime_bands(fig, regimes, x0=1)
    for name, vals in series.items():
        is_total = name == total_name
        fig.add_trace(go.Scatter(x=list(range(len(vals))), y=vals, name=name, mode="lines",
                                 line=dict(color=colors.get(name, theme.ACCENT), width=3 if is_total else 2,
                                           dash="solid"), hovertemplate="%{y:,.0f}"))
    if initial is not None:
        fig.add_hline(y=initial, line=dict(color=theme.INK_3, width=1, dash="dot"))
    return _base(fig, height, title, ytitle="Rs (sleeve value)", xtitle="Step")


def cumulative_return_chart(values: Dict[str, np.ndarray], colors: Dict[str, str], height: int = 340) -> go.Figure:
    fig = go.Figure()
    for name, v in values.items():
        cr = analytics.cumulative_returns(v) * 100
        fig.add_trace(go.Scatter(x=list(range(len(cr))), y=cr, name=name, mode="lines",
                                 line=dict(color=colors.get(name, theme.ACCENT), width=3 if name == "Total" else 2),
                                 hovertemplate="%{y:.2f}%"))
    fig.add_hline(y=0, line=dict(color=theme.INK_3, width=1, dash="dot"))
    return _base(fig, height, "Cumulative return (%)", ytitle="%", xtitle="Step")


def drawdown_chart(values: Dict[str, np.ndarray], colors: Dict[str, str], height: int = 340) -> go.Figure:
    fig = go.Figure()
    for name, v in values.items():
        dd = -analytics.drawdown_series(v) * 100
        fig.add_trace(go.Scatter(x=list(range(len(dd))), y=dd, name=name, mode="lines",
                                 line=dict(color=colors.get(name, theme.ACCENT), width=3 if name == "Total" else 2),
                                 hovertemplate="%{y:.2f}%"))
    return _base(fig, height, "Drawdown from running peak (%)", ytitle="%", xtitle="Step")


def rolling_vol_chart(values: Dict[str, np.ndarray], colors: Dict[str, str], window: int = 10, height: int = 340):
    fig = go.Figure()
    for name, v in values.items():
        rv = analytics.rolling_volatility(v, window) * 100
        fig.add_trace(go.Scatter(x=list(range(1, len(rv) + 1)), y=rv, name=name, mode="lines",
                                 line=dict(color=colors.get(name, theme.ACCENT), width=3 if name == "Total" else 2),
                                 hovertemplate="%{y:.3f}%"))
    return _base(fig, height, f"Rolling volatility ({window}-step std of step returns, %)", ytitle="% per step", xtitle="Step")


def step_return_hist(values: Dict[str, np.ndarray], colors: Dict[str, str], height: int = 320) -> go.Figure:
    fig = go.Figure()
    for name, v in values.items():
        r = analytics.step_returns(v) * 100
        if len(r):
            fig.add_trace(go.Histogram(x=r, name=name, opacity=0.65, marker_color=colors.get(name, theme.ACCENT),
                                       nbinsx=25, hovertemplate="%{x:.2f}%: %{y} steps"))
    fig.update_layout(barmode="overlay")
    return _base(fig, height, "Distribution of per-step returns (%)", ytitle="Steps", xtitle="Step return (%)")


def txcost_chart(cum_costs: Dict[str, np.ndarray], colors: Dict[str, str], height: int = 320) -> go.Figure:
    fig = go.Figure()
    for name, c in cum_costs.items():
        fig.add_trace(go.Scatter(x=list(range(1, len(c) + 1)), y=np.cumsum(c), name=name, mode="lines",
                                 line=dict(color=colors.get(name, theme.ACCENT), width=3 if name == "Total" else 2),
                                 hovertemplate="%{y:,.0f}"))
    return _base(fig, height, "Cumulative transaction costs (Rs)", ytitle="Rs", xtitle="Step")


# --------------------------------------------------------------------------- #
# Allocation
# --------------------------------------------------------------------------- #
def allocation_stack(weights: np.ndarray, regimes: Sequence[int] = (), title: str = "Allocation over time",
                     height: int = 320) -> go.Figure:
    """weights: (T, 4) ACTUAL portfolio weights after each step."""
    if weights is None or len(weights) == 0:
        return _empty("No steps yet.", height)
    fig = go.Figure()
    x = list(range(1, len(weights) + 1))
    for j, asset in enumerate(cfg.ASSET_NAMES):
        fig.add_trace(go.Scatter(x=x, y=weights[:, j] * 100, name=asset, mode="lines", stackgroup="w",
                                 line=dict(width=0.5, color="#fcfcfb"), fillcolor=theme.ASSET_COLORS[asset],
                                 hovertemplate="%{y:.1f}%"))
    fig.update_yaxes(range=[0, 100])
    return _base(fig, height, title, ytitle="% of sleeve", xtitle="Step")


def allocation_bars(rupees_by_agent: Dict[str, np.ndarray], title: str = "Current allocation by agent (Rs)",
                    height: int = 320) -> go.Figure:
    fig = go.Figure()
    names = list(rupees_by_agent)
    for j, asset in enumerate(cfg.ASSET_NAMES):
        vals = [rupees_by_agent[n][j] for n in names]
        fig.add_trace(go.Bar(x=names, y=vals, name=asset, marker=dict(color=theme.ASSET_COLORS[asset],
                             line=dict(color="#fcfcfb", width=2)), hovertemplate="%{y:,.0f}"))
    fig.update_layout(barmode="stack")
    return _base(fig, height, title, ytitle="Rs")


def allocation_donut(rupees: np.ndarray, title: str = "Total allocation", height: int = 320) -> go.Figure:
    fig = go.Figure(go.Pie(labels=cfg.ASSET_NAMES, values=np.maximum(rupees, 0), hole=0.58, sort=False,
                           marker=dict(colors=[theme.ASSET_COLORS[a] for a in cfg.ASSET_NAMES],
                                       line=dict(color="#fcfcfb", width=2)),
                           text=[(f"{n}<br>{v / max(rupees.sum(), 1e-9):.0%}" if v / max(rupees.sum(), 1e-9) >= 0.04 else "")
                                 for n, v in zip(cfg.ASSET_NAMES, np.maximum(rupees, 0))],
                           textinfo="text", textposition="inside", insidetextorientation="horizontal",
                           hovertemplate="%{label}: Rs %{value:,.0f}"))
    fig.update_layout(annotations=[dict(text=f"{theme.inr(float(np.sum(rupees)))}", x=0.5, y=0.5,
                                        showarrow=False, font=dict(size=13, color=theme.INK))])
    return _base(fig, height, title, legend=False)


def weight_waterfall(pnl: np.ndarray, title: str = "Where this step's profit / loss came from (Rs)", height: int = 300):
    fig = go.Figure(go.Waterfall(
        x=list(cfg.ASSET_NAMES) + ["Net"], y=list(pnl) + [float(np.sum(pnl))],
        measure=["relative"] * len(pnl) + ["total"], connector=dict(line=dict(color=theme.GRID)),
        increasing=dict(marker=dict(color=theme.GOOD)), decreasing=dict(marker=dict(color=theme.BAD)),
        totals=dict(marker=dict(color=theme.INK)), hovertemplate="Rs %{y:,.0f}"))
    return _base(fig, height, title, legend=False, ytitle="Rs")


def asset_pnl_contribution(pnl_by_asset: np.ndarray, title="Cumulative profit / loss by asset (Rs)", height=300):
    fig = go.Figure(go.Bar(x=list(cfg.ASSET_NAMES), y=pnl_by_asset,
                           marker=dict(color=[theme.ASSET_COLORS[a] for a in cfg.ASSET_NAMES]),
                           hovertemplate="Rs %{y:,.0f}"))
    return _base(fig, height, title, legend=False, ytitle="Rs")


def reward_components_bar(components: Dict[str, float], title="Reward breakdown (this step)", height=300):
    names = ["return", "risk", "transaction", "drawdown", "liquidity"]
    labels = ["Return", "Risk penalty", "Trading cost", "Drawdown penalty", "Liquidity bonus"]
    vals = [components[n] for n in names]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker=dict(color=[theme.GOOD if v >= 0 else theme.BAD for v in vals],
                           line=dict(color="#fcfcfb", width=1)), hovertemplate="%{y:+.3f}"))
    fig.add_hline(y=0, line=dict(color=theme.INK_3, width=1))
    return _base(fig, height, title, legend=False, ytitle="reward points")


def asset_returns_bar(base: np.ndarray, adjusted: np.ndarray, title="Asset returns this step (%)", height=280):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(cfg.ASSET_NAMES), y=base * 100, name="Market draw",
                         marker=dict(color=[theme.ASSET_COLORS[a] for a in cfg.ASSET_NAMES]),
                         hovertemplate="%{y:+.3f}%"))
    if not np.allclose(base, adjusted):
        fig.add_trace(go.Bar(x=list(cfg.ASSET_NAMES), y=adjusted * 100, name="After agents' market impact",
                             marker=dict(color="#0b0b0b", opacity=0.55), hovertemplate="%{y:+.3f}%"))
    fig.update_layout(barmode="group")
    fig.add_hline(y=0, line=dict(color=theme.INK_3, width=1))
    return _base(fig, height, title, ytitle="%")


# --------------------------------------------------------------------------- #
# Agent comparison
# --------------------------------------------------------------------------- #
def metric_bars(df, metric: str, title: str, colors: Dict[str, str], pct_axis=True, height=300):
    """df: rows indexed by agent name with the metric column (fraction or raw)."""
    vals = df[metric].astype(float).values * (100 if pct_axis else 1)
    fig = go.Figure(go.Bar(x=list(df.index), y=vals, marker=dict(color=[colors.get(n, theme.ACCENT) for n in df.index]),
                           text=[f"{v:.2f}{'%' if pct_axis else ''}" for v in vals], textposition="outside",
                           hovertemplate="%{y:.3f}"))
    fig.add_hline(y=0, line=dict(color=theme.INK_3, width=1))
    return _base(fig, height, title, legend=False)


def risk_return_scatter(points: Dict[str, tuple], colors: Dict[str, str], height=320):
    fig = go.Figure()
    for name, (vol, ret) in points.items():
        fig.add_trace(go.Scatter(x=[vol * 100], y=[ret * 100], mode="markers+text", name=name, text=[name],
                                 textposition="top center", marker=dict(size=14, color=colors.get(name, theme.ACCENT),
                                 line=dict(color="#fcfcfb", width=2)), hovertemplate="vol %{x:.3f}% / return %{y:.2f}%"))
    return _base(fig, height, "Risk vs return", ytitle="Total return (%)", xtitle="Volatility (std of step return, %)")


def returns_box(returns_by_name: Dict[str, Sequence[float]], colors: Dict[str, str], title="Distribution of episode returns across held-out seeds (%)", height=340):
    fig = go.Figure()
    for name, r in returns_by_name.items():
        fig.add_trace(go.Box(y=np.asarray(r) * 100, name=name, marker_color=colors.get(name, theme.ACCENT),
                             boxmean=True, boxpoints="all", jitter=0.4, pointpos=0, marker_size=4,
                             hovertemplate="%{y:.2f}%"))
    fig.add_hline(y=0, line=dict(color=theme.INK_3, width=1, dash="dot"))
    fig.update_layout(hovermode="closest")
    return _base(fig, height, title, legend=False, ytitle="Total return (%)")


def mean_curve_band(curves: Dict[str, tuple], colors: Dict[str, str], title="Mean portfolio value across held-out seeds (Rs, +/- 1 std)", height=360):
    """curves: name -> (mean array, std array)."""
    fig = go.Figure()
    for name, (m, s) in curves.items():
        m, s = np.asarray(m), np.asarray(s)
        x = list(range(len(m)))
        col = colors.get(name, theme.ACCENT)
        fig.add_trace(go.Scatter(x=x + x[::-1], y=list(m + s) + list((m - s)[::-1]), fill="toself", mode="lines",
                                 line=dict(width=0), fillcolor=_rgba(col, 0.12), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=m, name=name, mode="lines", line=dict(color=col, width=2.5), hovertemplate="%{y:,.0f}"))
    return _base(fig, height, title, ytitle="Rs", xtitle="Step")


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def action_hist_heatmap(hists: Dict[str, Sequence[float]], title="How often each agent chose each basket (%)", height=300):
    names = list(hists)
    z = np.array([hists[n] for n in names]) * 100
    fig = go.Figure(go.Heatmap(z=z, x=[f"{i}: {n}" for i, n in enumerate(cfg.ACTION_BASKET_NAMES)], y=names,
                               colorscale=[[0, "#f4f8fe"], [1, "#2a78d6"]], zmin=0, zmax=100,
                               text=np.round(z, 0).astype(int).astype(str), texttemplate="%{text}",
                               hovertemplate="%{y} chose %{x}: %{z:.1f}%<extra></extra>", colorbar=dict(title="%")))
    fig.update_xaxes(tickangle=-30)
    fig.update_layout(hovermode="closest")
    return _base(fig, height, title, legend=False)


# --------------------------------------------------------------------------- #
# RL charts
# --------------------------------------------------------------------------- #
def learning_curves(df, agent_ids, roles, window: int = 50, height=360):
    if df is None or len(df) == 0:
        return _empty("Train the agents to see learning curves.", height)
    fig = go.Figure()
    for aid in agent_ids:
        s = df[f"reward_{aid}"].rolling(window, min_periods=1).mean()
        role = roles[aid]
        fig.add_trace(go.Scatter(x=df["episode"], y=s, name=role, mode="lines",
                                 line=dict(color=theme.agent_color(role), width=2), hovertemplate="%{y:.2f}"))
    return _base(fig, height, f"Episode reward, {window}-episode moving average", ytitle="Sum of step rewards", xtitle="Training episode")


def epsilon_curve(df, agent_ids, roles, height=280):
    if df is None or len(df) == 0:
        return _empty("No training yet.", height)
    fig = go.Figure()
    for aid in agent_ids:
        fig.add_trace(go.Scatter(x=df["episode"], y=df[f"epsilon_{aid}"], name=roles[aid], mode="lines",
                                 line=dict(color=theme.agent_color(roles[aid]), width=2), hovertemplate="%{y:.3f}"))
    fig.update_yaxes(range=[0, 1.02])
    return _base(fig, height, "Exploration rate epsilon", ytitle="epsilon", xtitle="Training episode")


def td_curve(df, agent_ids, roles, window=50, height=280):
    if df is None or len(df) == 0 or f"td_{agent_ids[0]}" not in df.columns:
        return _empty("No training yet.", height)
    fig = go.Figure()
    for aid in agent_ids:
        fig.add_trace(go.Scatter(x=df["episode"], y=df[f"td_{aid}"].rolling(window, min_periods=1).mean(),
                                 name=roles[aid], mode="lines", line=dict(color=theme.agent_color(roles[aid]), width=2),
                                 hovertemplate="%{y:.3f}"))
    return _base(fig, height, "Mean |TD error| per step (learning signal, smoothed)", ytitle="|TD error|", xtitle="Training episode")


def training_capital_curve(df, agent_ids, roles, window=50, height=280):
    if df is None or len(df) == 0:
        return _empty("No training yet.", height)
    fig = go.Figure()
    for aid in agent_ids:
        fig.add_trace(go.Scatter(x=df["episode"], y=df[f"capital_{aid}"].rolling(window, min_periods=1).mean(),
                                 name=roles[aid], mode="lines", line=dict(color=theme.agent_color(roles[aid]), width=2),
                                 hovertemplate="%{y:,.0f}"))
    return _base(fig, height, "Final sleeve value per TRAINING episode (in-sample, includes exploration), smoothed",
                 ytitle="Rs", xtitle="Training episode")


def q_values_bar(qvals: np.ndarray, visits: Optional[np.ndarray] = None, title="Q-values for the selected state", height=320):
    best = int(np.argmax(qvals))
    colors = [theme.ACCENT if i == best else "#b9c9e2" for i in range(len(qvals))]
    fig = go.Figure(go.Bar(x=[f"{i}: {n}" for i, n in enumerate(cfg.ACTION_BASKET_NAMES)], y=qvals,
                           marker=dict(color=colors), text=[f"{v:.2f}" for v in qvals], textposition="outside",
                           customdata=visits if visits is not None else np.zeros(len(qvals)),
                           hovertemplate="Q = %{y:.3f}<br>times tried: %{customdata}<extra></extra>"))
    fig.update_xaxes(tickangle=-30)
    fig.update_layout(hovermode="closest")
    return _base(fig, height, title, legend=False, ytitle="Q(s, a)")


def state_grid(values_108: np.ndarray) -> np.ndarray:
    """Map a length-108 per-state vector onto a 9 x 12 grid:
    rows = (equity bucket, bond bucket); columns = (regime, gold bucket)."""
    grid = np.full((9, 12), np.nan)
    for s in range(cfg.N_STATES):
        d = decode_state(s)
        r = d["equity_bucket"] * 3 + d["bond_bucket"]
        c = d["regime"] * 3 + d["gold_bucket"]
        grid[r, c] = values_108[s]
    return grid


def _grid_axes():
    ylab = [f"Eq {BUCKET_SHORT[e]} / Bd {BUCKET_SHORT[b]}" for e in range(3) for b in range(3)]
    xlab = [f"{cfg.REGIME_NAMES[r]} / Au {BUCKET_SHORT[g]}" for r in range(4) for g in range(3)]
    return xlab, ylab


def policy_heatmap(agent, mode: str = "greedy", title: Optional[str] = None, height: int = 430):
    """mode: 'greedy' (best basket index, text) | 'maxq' | 'visits' (log10)."""
    xlab, ylab = _grid_axes()
    visits = agent.visit_counts.sum(axis=1)
    if mode == "greedy":
        z = state_grid(np.argmax(agent.q_table, axis=1).astype(float))
        txt = np.where(np.isnan(z), "", np.nan_to_num(z).astype(int).astype(str))
        visited = state_grid((visits > 0).astype(float))
        z = np.where(visited > 0, z, np.nan)
        txt = np.where(visited > 0, txt, "-")
        fig = go.Figure(go.Heatmap(z=z, x=xlab, y=ylab, text=txt, texttemplate="%{text}", zmin=0, zmax=cfg.N_ACTIONS - 1,
                                   colorscale=[[0, "#6250d6"], [0.5, "#f4e9b8"], [1, "#1baf7a"]],
                                   hovertemplate="%{y} | %{x}<br>greedy basket %{text}<extra></extra>",
                                   colorbar=dict(title="basket<br>index", tickvals=list(range(cfg.N_ACTIONS)))))
        t = title or "Learned greedy basket in each state (0 = AggressiveGrowth ... 6 = CashHeavy; '-' = never visited)"
    elif mode == "maxq":
        z = state_grid(agent.q_table.max(axis=1))
        fig = go.Figure(go.Heatmap(z=z, x=xlab, y=ylab, colorscale="Blues",
                                   hovertemplate="%{y} | %{x}<br>max Q %{z:.2f}<extra></extra>", colorbar=dict(title="max Q")))
        t = title or "Best Q-value in each state"
    else:
        z = state_grid(np.log10(visits + 1.0))
        fig = go.Figure(go.Heatmap(z=z, x=xlab, y=ylab, colorscale="Purples",
                                   customdata=state_grid(visits.astype(float)),
                                   hovertemplate="%{y} | %{x}<br>visits %{customdata:,.0f}<extra></extra>",
                                   colorbar=dict(title="log10<br>visits")))
        t = title or "How often each state was visited during training (log scale)"
    fig.update_xaxes(tickangle=-45, side="bottom")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(hovermode="closest")
    return _base(fig, height, t, legend=False)


def training_curves_from_reference(curves: Dict[str, dict], colors: Dict[str, str], title: str, ytitle: str, height=320):
    fig = go.Figure()
    for name, c in curves.items():
        fig.add_trace(go.Scatter(x=c["episode"], y=c["value"], name=name, mode="lines",
                                 line=dict(color=colors.get(name, theme.ACCENT), width=2), hovertemplate="%{y:.2f}"))
    return _base(fig, height, title, ytitle=ytitle, xtitle="Training episode")


# --------------------------------------------------------------------------- #
# Market / education
# --------------------------------------------------------------------------- #
def transition_heatmap(height=330):
    P = cfg.REGIME_TRANSITION
    fig = go.Figure(go.Heatmap(z=P, x=cfg.REGIME_NAMES, y=cfg.REGIME_NAMES, colorscale="Blues", zmin=0, zmax=1,
                               text=np.round(P, 2).astype(str), texttemplate="%{text}",
                               hovertemplate="from %{y} to %{x}: %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="from (today)")
    fig.update_xaxes(title="to (tomorrow)", side="top")
    fig.update_layout(hovermode="closest")
    return _base(fig, height, "", legend=False)


def regime_path_chart(path: Sequence[int], height=200):
    fig = go.Figure(go.Scatter(x=list(range(len(path))), y=path, mode="lines+markers", line=dict(shape="hv", color=theme.INK_2),
                               marker=dict(size=7, color=[theme.REGIME_COLORS[cfg.REGIME_NAMES[r]] for r in path]),
                               hovertemplate="step %{x}: %{text}<extra></extra>", text=[cfg.REGIME_NAMES[r] for r in path]))
    fig.update_yaxes(tickvals=list(range(4)), ticktext=cfg.REGIME_NAMES, range=[-0.4, 3.4])
    fig.update_layout(hovermode="closest")
    return _base(fig, height, "", legend=False, xtitle="Step")


def asset_return_params_heatmap(height=300):
    mu = np.array([[cfg.ASSET_RETURN_PARAMS[r][a][0] for a in range(cfg.N_ASSETS)] for r in range(4)]) * 100
    fig = go.Figure(go.Heatmap(z=mu, x=cfg.ASSET_NAMES, y=cfg.REGIME_NAMES, colorscale="RdYlGn", zmid=0,
                               text=np.round(mu, 3).astype(str), texttemplate="%{text}%",
                               hovertemplate="%{y} / %{x}: mean %{z:.3f}% per step<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(hovermode="closest")
    return _base(fig, height, "Mean return per step, by regime and asset (%)", legend=False)
