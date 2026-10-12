"""Page 4 - Portfolio Analytics: growth, returns, volatility, drawdown, Sharpe, costs, agent comparison."""

import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from dashboard import charts, theme, views
from evaluation import analytics, metrics
from experiments.engine import evaluate_agents
from experiments.run_experiments import AGENT_IDS, AGENT_ROLES

SRC_EPISODE = "The episode in the Simulation page"
SRC_OOS = "Out-of-sample evaluation of the active agents"


def _run_eval():
    ss = st.session_state
    n = int(ss.get("an_eval_n", 50))
    start = int(ss.get("an_eval_start", cfg.EVAL_SEED_START))
    seeds = list(range(start, start + n))
    caps = {a: cfg.INITIAL_CAPITAL_PER_AGENT for a in AGENT_IDS}
    res = evaluate_agents(ss["agents"], AGENT_IDS, AGENT_ROLES, caps, seeds)
    res["label"] = ss["agents_label"]
    res["seeds"] = seeds
    ss["an_eval"] = res


def episode_analytics(traj, agent_ids, roles, caps, key: str):
    """All single-episode analytics, computed from the trajectory passed in."""
    if not traj:
        st.info("No steps to analyse yet - run an episode first.")
        return
    colors = views.color_map(roles)
    mf = views.metrics_frame(traj, agent_ids, roles, caps)
    st.dataframe(views.display_metrics(mf), width="stretch")
    st.caption(metrics.SHARPE_METHODOLOGY_NOTE)
    series = views.series_by_name(traj, agent_ids, roles, caps)
    regimes = [s["current_regime"] for s in traj]
    tabs = st.tabs(["Growth & returns", "Risk", "Transaction costs", "Where profit came from", "Agent comparison", "Allocation"])
    with tabs[0]:
        sleeves, total_only = views.split_series(series)
        g1, g2 = st.columns(2)
        g1.plotly_chart(charts.portfolio_value(sleeves, colors, regimes, title="Each agent's sleeve (Rs)"),
                        width="stretch", key=f"{key}_growth")
        g2.plotly_chart(charts.portfolio_value(total_only, colors, regimes, initial=float(sum(caps.values())),
                                               title="Total portfolio (Rs)"), width="stretch", key=f"{key}_growth_total")
        st.plotly_chart(charts.cumulative_return_chart(series, colors), width="stretch", key=f"{key}_cum")
    with tabs[1]:
        w = st.slider("Rolling-volatility window (steps)", 3, 30, 10, key="an_vol_window")
        a, b = st.columns(2)
        a.plotly_chart(charts.drawdown_chart(series, colors), width="stretch", key=f"{key}_dd")
        b.plotly_chart(charts.rolling_vol_chart(series, colors, w), width="stretch", key=f"{key}_vol")
        st.plotly_chart(charts.step_return_hist(series, colors), width="stretch", key=f"{key}_hist")
    with tabs[2]:
        costs = views.txcosts_by_name(traj, agent_ids, roles)
        st.plotly_chart(charts.txcost_chart(costs, colors), width="stretch", key=f"{key}_tx")
        rows = {n: {"Total costs": theme.inr(float(np.sum(c))), "Per step (avg)": theme.inr(float(np.mean(c))),
                    "Steps with trading": int(np.sum(c > 1e-9))} for n, c in costs.items()}
        st.dataframe(pd.DataFrame(rows).T, width="stretch")
        st.caption(f"Cost model: {cfg.TRANSACTION_COST_RATE:.2%} of the amount traded (sum of absolute weight changes x capital).")
    with tabs[3]:
        sel = st.selectbox("Agent", ["All agents"] + [roles[a] for a in agent_ids], key=f"{key}_attr_agent")
        ids = agent_ids if sel == "All agents" else [a for a in agent_ids if roles[a] == sel]
        pnl = views.asset_pnl_total(traj, ids)
        st.plotly_chart(charts.asset_pnl_contribution(pnl), width="stretch", key=f"{key}_attr")
        st.caption("Gross profit/loss earned by each asset sleeve (before subtracting transaction costs).")
    with tabs[4]:
        ag = mf.drop(index=views.TOTAL, errors="ignore")
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.metric_bars(ag, "Total return", "Total return (%)", colors), width="stretch", key=f"{key}_c1")
        c2.plotly_chart(charts.metric_bars(ag, "Max drawdown", "Max drawdown (%)", colors), width="stretch", key=f"{key}_c2")
        c3, c4 = st.columns(2)
        c3.plotly_chart(charts.metric_bars(ag, "Sharpe", "Sharpe (episode-scale)", colors, pct_axis=False), width="stretch", key=f"{key}_c3")
        c4.plotly_chart(charts.risk_return_scatter({n: (ag.loc[n, "Volatility"], ag.loc[n, "Total return"]) for n in ag.index}, colors),
                        width="stretch", key=f"{key}_c4")
    with tabs[5]:
        names = [roles[a] for a in agent_ids]
        pick = st.selectbox("Agent", names, key=f"{key}_alloc_agent")
        aid = agent_ids[names.index(pick)]
        st.plotly_chart(charts.allocation_stack(views.weights_array(traj, aid), regimes, f"{pick}: portfolio weights"),
                        width="stretch", key=f"{key}_alloc")
    export = pd.DataFrame({n: v for n, v in series.items()})
    export.index.name = "step"
    st.download_button("Download value series (CSV)", export.to_csv().encode(), "portfolio_value_series.csv", "text/csv",
                       key=f"{key}_dl")


def render():
    ss = st.session_state
    st.title("Portfolio Analytics")
    st.caption("Every number and chart here is computed from actual simulation output (synthetic market) - "
               "either the episode currently in the Simulation page, or many held-out episodes.")
    src = st.radio("Data source", [SRC_EPISODE, SRC_OOS], horizontal=True, key="an_source")

    if src == SRC_EPISODE:
        sim = ss["sim"]
        st.markdown(f'<div class="small">Agents: <b>{ss["agents_label"]}</b> | market seed <b>{sim.seed}</b> | {sim.summary_line()}</div>',
                    unsafe_allow_html=True)
        episode_analytics(sim.history, sim.agent_ids, sim.roles, sim.capitals, key="an_ep")
        return

    # ---------- out-of-sample, many seeds ---------- #
    c1, c2, c3 = st.columns([1, 1, 1.4])
    c1.number_input("First evaluation seed", 0, 1_000_000, cfg.EVAL_SEED_START, 1, key="an_eval_start")
    c2.number_input("Number of episodes", 5, 200, 50, 5, key="an_eval_n")
    c3.write("")
    c3.button("Run out-of-sample evaluation", on_click=_run_eval, type="primary", width="stretch")
    res = ss.get("an_eval")
    if not res:
        st.info("Press the button to play the active (frozen, greedy) agents on fresh markets they were not trained on.")
        return
    seeds = res["seeds"]
    summ = res["summary"]
    st.markdown(f'<div class="small">Agents: <b>{res["label"]}</b> | {len(seeds)} held-out seeds ({seeds[0]}-{seeds[-1]}) | '
                'epsilon = 0, Q-tables frozen, no learning.</div>', unsafe_allow_html=True)
    rows = {}
    for aid in AGENT_IDS:
        p = summ["per_agent"][aid]
        rows[AGENT_ROLES[aid]] = {
            "Return mean": theme.pct(p["total_return_mean"], 2, True), "Return std": theme.pct(p["total_return_std"], 2),
            "Worst": theme.pct(p["total_return_min"], 2, True), "Best": theme.pct(p["total_return_max"], 2, True),
            "Loss episodes": theme.pct(p["prob_loss"], 0), "Volatility/step": theme.pct(p["volatility_mean"], 3),
            "Max drawdown": theme.pct(p["max_drawdown_mean"], 2), "Sharpe": f"{p['sharpe_mean']:.2f}",
            "Trading costs": theme.inr(p["total_txcost_mean"])}
    p = summ["combined"]
    rows["Total (all 4)"] = {
        "Return mean": theme.pct(p["total_return_mean"], 2, True), "Return std": theme.pct(p["total_return_std"], 2),
        "Worst": theme.pct(p["total_return_min"], 2, True), "Best": theme.pct(p["total_return_max"], 2, True),
        "Loss episodes": theme.pct(p["prob_loss"], 0), "Volatility/step": theme.pct(p["volatility_mean"], 3),
        "Max drawdown": theme.pct(p["max_drawdown_mean"], 2), "Sharpe": f"{p['sharpe_mean']:.2f}",
        "Trading costs": theme.inr(p["total_txcost_mean"])}
    st.dataframe(pd.DataFrame(rows).T, width="stretch")
    st.caption(metrics.SHARPE_METHODOLOGY_NOTE)

    colors = {AGENT_ROLES[a]: theme.agent_color(AGENT_ROLES[a]) for a in AGENT_IDS}
    colors["Total"] = theme.TOTAL_COLOR
    per_seed = {AGENT_ROLES[int(k)]: v for k, v in summ["per_agent_returns_per_seed"].items()}
    per_seed["Total"] = summ["combined_returns_per_seed"]
    a, b = st.columns(2)
    a.plotly_chart(charts.returns_box(per_seed, colors), width="stretch", key="an_oos_box")
    curves = {"Total": (summ["mean_equity_curve"], summ["std_equity_curve"])}
    b.plotly_chart(charts.mean_curve_band(curves, colors), width="stretch", key="an_oos_curve")

    st.markdown("---")
    st.markdown("##### Inspect one held-out episode in detail")
    pick = st.selectbox("Held-out seed", seeds, key="an_seed_pick")
    traj = res["eval_trajectories"][seeds.index(pick)]
    caps = {a: cfg.INITIAL_CAPITAL_PER_AGENT for a in AGENT_IDS}
    episode_analytics(traj, AGENT_IDS, AGENT_ROLES, caps, key="an_oos_ep")
