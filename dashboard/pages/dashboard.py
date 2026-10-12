"""Page 1 - Dashboard: portfolio value, returns, allocation, regime, agent performance."""

import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from dashboard import charts, theme, views
from dashboard.state import goto

DASH_SEED_KEY = "dash_seed"


def _run_new_episode():
    ss = st.session_state
    seed = int(ss.get(DASH_SEED_KEY, cfg.EVAL_SEED_START))
    ss["sim"].reset(seed=seed)
    ss["sim"].run_to_end()


def _reset_only():
    ss = st.session_state
    ss["sim"].reset(seed=int(ss.get(DASH_SEED_KEY, cfg.EVAL_SEED_START)))


def render():
    ss = st.session_state
    sim = ss["sim"]
    st.title("Multi-Agent Financial Resource Allocation")
    st.caption("Four independent Q-learning agents share one synthetic market and split Rs 10,00,000 into "
               "four sleeves of Rs 2,50,000. This page summarises the episode currently loaded in the simulator.")

    c1, c2, c3, c4 = st.columns([1.2, 1.3, 1.1, 1.6])
    with c1:
        st.number_input("Market seed", min_value=0, max_value=1_000_000, value=int(sim.seed), step=1, key=DASH_SEED_KEY,
                        help="Same seed = same synthetic market path. Seeds 51-100 are the held-out evaluation block "
                             "used on the Experiments page; the agents were trained on seed 42.")
    with c2:
        st.write("")
        st.button("Run a full episode", type="primary", on_click=_run_new_episode, width="stretch",
                  help="Simulates all steps of one episode on this seed with the current policies.")
    with c3:
        st.write("")
        st.button("Clear (step 0)", on_click=_reset_only, width="stretch")
    with c4:
        st.write("")
        st.button("Step through it in the Simulation page", on_click=goto, args=("Simulation",), width="stretch")

    st.markdown(f'<div class="small">Agents: <b>{ss["agents_label"]}</b> &nbsp;|&nbsp; Market seed <b>{sim.seed}</b> '
                f'&nbsp;|&nbsp; {sim.summary_line()}</div>', unsafe_allow_html=True)
    st.write("")

    traj = sim.history
    metrics_df = views.metrics_frame(traj, sim.agent_ids, sim.roles, sim.capitals)
    total = metrics_df.loc[views.TOTAL]
    start_total = float(sum(sim.capitals.values()))

    # ---------------- KPI row ----------------
    k1, k2, k3, k4, k5 = st.columns(5)
    ret = total["Total return"]
    tone = "pos" if ret > 0 else ("neg" if ret < 0 else "")
    k1.markdown(theme.kpi("Total portfolio value", theme.inr(total["Final value"]),
                          f"started at {theme.inr(start_total)}"), unsafe_allow_html=True)
    k2.markdown(theme.kpi("Return so far", theme.pct(ret, 2, signed=True),
                          f"{theme.inr(total['Final value'] - start_total)} profit/loss", tone), unsafe_allow_html=True)
    cur = sim.current_regime
    k3.markdown(theme.kpi("Market regime now", theme.regime_pill(cur), "regime for the next step" if not sim.done
                          else "episode finished"), unsafe_allow_html=True)
    agent_only = metrics_df.drop(index=views.TOTAL)
    spread = float(agent_only["Total return"].max() - agent_only["Total return"].min())
    if spread < 1e-9:
        k4.markdown(theme.kpi("Best agent", "Tie", "all four agents earned the same return"), unsafe_allow_html=True)
    else:
        best = agent_only["Total return"].idxmax()
        k4.markdown(theme.kpi("Best agent", f"{best}", f"{theme.pct(agent_only.loc[best, 'Total return'], 2, True)} return"),
                    unsafe_allow_html=True)
    k5.markdown(theme.kpi("Max drawdown (total)", theme.pct(total["Max drawdown"], 2),
                          f"trading costs {theme.inr(total['Transaction costs'])}"), unsafe_allow_html=True)
    st.write("")

    if not traj:
        st.markdown('<div class="note">No steps have been simulated yet. Press <b>Run a full episode</b> above, or open '
                    'the <b>Simulation</b> page to step through one decision at a time.</div>', unsafe_allow_html=True)

    # ---------------- main charts ----------------
    left, right = st.columns([2.1, 1])
    with left:
        series = views.series_by_name(traj, sim.agent_ids, sim.roles, sim.capitals)
        sleeves, total_only = views.split_series(series)
        colors = views.color_map(sim.roles)
        t_sleeve, t_total = st.tabs(["Each agent's sleeve (Rs)", "Total portfolio (Rs)"])
        with t_sleeve:
            st.plotly_chart(charts.portfolio_value(sleeves, colors, sim.regime_path(), initial=None,
                                                   title="Value of each agent's sleeve - shaded bands show the market regime"),
                            width="stretch", key="dash_value")
            if traj and views.sleeves_identical(series):
                st.caption("The four lines sit exactly on top of each other: in this episode all agents chose the same "
                           "baskets, so their sleeves evolve identically (see 'Experiments' for the specialization finding).")
        with t_total:
            st.plotly_chart(charts.portfolio_value(total_only, colors, sim.regime_path(), initial=start_total,
                                                   title="Total portfolio value - dotted line is the Rs 10,00,000 start"),
                            width="stretch", key="dash_total")
    with right:
        rupees = views.current_rupees(sim)
        total_by_asset = np.sum(list(rupees.values()), axis=0)
        st.plotly_chart(charts.allocation_donut(total_by_asset, "Where the money is right now"),
                        width="stretch", key="dash_donut")

    l2, r2 = st.columns([1, 1.4])
    with l2:
        st.plotly_chart(charts.allocation_bars(rupees, "Current allocation by agent (Rs)"),
                        width="stretch", key="dash_alloc_bars")
    with r2:
        st.markdown("**Agent performance (this episode)**")
        st.dataframe(views.display_metrics(metrics_df), width="stretch")
        st.caption("Sharpe is episode-scale (per-step Sharpe x sqrt(steps)), NOT annualised. Volatility is the std of "
                   "per-step returns. All figures are from the synthetic simulator.")

    tabs = st.tabs(["Allocation over time", "Return & risk by agent", "Latest step"])
    with tabs[0]:
        names = [sim.roles[a] for a in sim.agent_ids]
        pick = st.selectbox("Agent", names, key="dash_alloc_agent")
        aid = sim.agent_ids[names.index(pick)]
        st.plotly_chart(charts.allocation_stack(views.weights_array(traj, aid), sim.regime_path(),
                                                f"{pick}: actual portfolio weights over time"),
                        width="stretch", key="dash_alloc_stack")
    with tabs[1]:
        a, b = st.columns(2)
        colors = views.color_map(sim.roles)
        a.plotly_chart(charts.metric_bars(agent_only, "Total return", "Total return (%)", colors),
                       width="stretch", key="dash_bar_ret")
        b.plotly_chart(charts.metric_bars(agent_only, "Max drawdown", "Max drawdown (%)", colors),
                       width="stretch", key="dash_bar_dd")
    with tabs[2]:
        if sim.last is None:
            st.info("Nothing yet - run an episode first.")
        else:
            last = sim.last
            st.write(f"Step **{last['t']}** was generated in the **{cfg.REGIME_NAMES[last['current_regime']]}** regime; "
                     f"the market then moved to **{cfg.REGIME_NAMES[last['next_regime']]}**.")
            rows = []
            for a in sim.agent_ids:
                rows.append({"Agent": sim.roles[a], "Basket chosen": cfg.ACTION_BASKET_NAMES[last["actions"][a]],
                             "Equity/Bonds/Gold/Cash": " / ".join(f"{w:.0%}" for w in cfg.ACTION_BASKETS[last["actions"][a]]),
                             "Step reward": round(last["rewards"][a], 3),
                             "Sleeve value": theme.inr(last["capital"][a])})
            st.dataframe(pd.DataFrame(rows).set_index("Agent"), width="stretch")
