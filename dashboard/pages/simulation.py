"""Page 2 - Interactive Simulation: start / pause / step / reset, how agents decide and interact."""

import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from dashboard import charts, theme, views
from dashboard.session import POLICY_KINDS, POLICY_LABELS, SimulationSession

VIEWS = ["Overview", "How agents decide", "Money flow (Rs)", "Step log"]


# ------------------------------ callbacks ------------------------------ #
def _start():
    sim = st.session_state["sim"]
    if not sim.done:
        sim.running = True


def _pause():
    st.session_state["sim"].running = False


def _step():
    sim = st.session_state["sim"]
    sim.running = False
    sim.step()


def _step_n():
    sim = st.session_state["sim"]
    sim.running = False
    sim.run_steps(int(st.session_state.get("sim_steps_n", 10)))


def _to_end():
    sim = st.session_state["sim"]
    sim.running = False
    sim.run_to_end()


def _reset():
    st.session_state["sim"].reset()


def _apply_settings():
    """Rebuild the session with a new seed / impact setting, keeping policy choices."""
    ss = st.session_state
    old = ss["sim"]
    new = SimulationSession(agents=ss["agents"], seed=int(ss.get("sim_seed", old.seed)),
                            enable_market_impact=bool(ss.get("sim_impact", True)))
    new.policy_spec = {a: dict(v) for a, v in old.policy_spec.items()}
    ss["sim"] = new


# ------------------------------ settings ------------------------------- #
def _settings(sim):
    with st.expander("Settings: market seed, market impact and each agent's policy", expanded=(sim.t == 0)):
        c1, c2, c3 = st.columns([1, 1, 1.2])
        c1.number_input("Market seed", 0, 1_000_000, int(sim.seed), 1, key="sim_seed",
                        help="Same seed => the same synthetic market path, so you can compare policies fairly.")
        c2.checkbox("Agents' trades move the equity price (market impact)", value=sim.enable_market_impact, key="sim_impact",
                    help="When on, the capital-weighted aggregate equity weight above 40% lowers the equity return "
                         "for EVERYONE - the interaction channel between agents.")
        c3.write("")
        c3.button("Apply seed & impact setting (restarts at step 0)", on_click=_apply_settings, width="stretch")
        st.markdown("**Policy for each agent** (can be changed between steps - e.g. take manual control with *Fixed / manual basket*)")
        cols = st.columns(4)
        for col, aid in zip(cols, sim.agent_ids):
            role = sim.roles[aid]
            with col:
                st.markdown(theme.agent_label(role), unsafe_allow_html=True)
                kind = st.selectbox("Policy", POLICY_KINDS, index=POLICY_KINDS.index(sim.policy_spec[aid]["kind"]),
                                    format_func=lambda k: POLICY_LABELS[k], key=f"sim_policy_kind_{aid}",
                                    label_visibility="collapsed")
                basket = sim.policy_spec[aid]["basket"]
                if kind == "fixed":
                    basket = st.selectbox("Basket", list(range(cfg.N_ACTIONS)), index=basket,
                                          format_func=lambda i: f"{i}: {cfg.ACTION_BASKET_NAMES[i]}", key=f"sim_basket_{aid}")
                sim.set_policy(aid, kind, basket)


# ------------------------------ views ---------------------------------- #
def _overview(sim):
    traj = sim.history
    series = views.series_by_name(traj, sim.agent_ids, sim.roles, sim.capitals)
    sleeves, total_only = views.split_series(series)
    colors = views.color_map(sim.roles)
    v1, v2 = st.columns(2)
    v1.plotly_chart(charts.portfolio_value(sleeves, colors, sim.regime_path(), title="Each agent's sleeve (Rs)", height=360),
                    width="stretch", key="sim_ov_sleeves")
    v2.plotly_chart(charts.portfolio_value(total_only, colors, sim.regime_path(), initial=float(sum(sim.capitals.values())),
                                           title="Total portfolio (Rs)", height=360),
                    width="stretch", key="sim_ov_total")
    names = [sim.roles[a] for a in sim.agent_ids]
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.allocation_bars(views.current_rupees(sim), "Rupees held in each asset, by agent"),
                        width="stretch", key="sim_ov_alloc")
    with c2:
        if sim.last:
            st.plotly_chart(charts.asset_returns_bar(sim.last["base_returns"], sim.last["asset_returns"],
                                                     f"Asset returns drawn for step {sim.last['t']} (%)"),
                            width="stretch", key="sim_ov_ret")
        else:
            st.info("Asset returns appear after the first step.")


def _decisions(sim):
    last = sim.last
    if last is None:
        st.info("Press **Step** or **Start**. Each step the four agents look at their own state, pick a basket, "
                "and the shared market then pays them out.")
        return
    st.markdown(f"**Step {last['t']}** - the market was in the **{cfg.REGIME_NAMES[last['current_regime']]}** regime "
                f"(it then moved to **{cfg.REGIME_NAMES[last['next_regime']]}**).")
    cols = st.columns(2)
    for i, aid in enumerate(sim.agent_ids):
        role = sim.roles[aid]
        with cols[i % 2]:
            st.markdown(f'<div class="card"><h4>{theme.agent_label(role)} '
                        f'<span class="small">- {POLICY_LABELS[last["policy_kind"][aid]]}</span></h4>'
                        f'<div class="small"><b>State seen:</b> {last["state_desc"][aid]} (index {last["states"][aid]})<br>'
                        f'<b>Chose:</b> basket {last["actions"][aid]} "{cfg.ACTION_BASKET_NAMES[last["actions"][aid]]}" = '
                        + " / ".join(f"{n} {w:.0%}" for n, w in zip(cfg.ASSET_NAMES, cfg.ACTION_BASKETS[last["actions"][aid]]))
                        + f'<br><b>Reward received:</b> {last["rewards"][aid]:+.3f}</div></div>', unsafe_allow_html=True)
            q = last["q_values"][aid]
            if q is not None:
                st.plotly_chart(charts.q_values_bar(q, None, "Q-values the agent compared (highest = chosen)", 230),
                                width="stretch", key=f"sim_q_{aid}")
            else:
                st.caption("This policy does not use a Q-table (random / fixed).")
    st.markdown("##### How the agents affect each other")
    st.markdown(
        f"The capital-weighted **aggregate equity weight** chosen this step was **{last['agg_equity_weight']:.1%}** "
        f"(neutral baseline {cfg.MARKET_IMPACT_BASELINE_EQUITY:.0%}). The one-sided crowding penalty "
        f"`-kappa * max(0, aggregate - baseline)^2` was **{last['market_impact'] * 100:+.4f}%** applied to the equity return "
        f"for **all** agents. The regime path and the base return draw are exogenous (identical for every policy on the same seed); "
        f"this crowding term is what couples the agents.")


def _money_flow(sim):
    last = sim.last
    names = [sim.roles[a] for a in sim.agent_ids]
    pick = st.selectbox("Agent", names, key="sim_view_agent")
    aid = sim.agent_ids[names.index(pick)]
    if last is None:
        st.info("Take a step to see how an allocation percentage turns into rupees.")
        return
    pre = last["asset_values_pre"][aid]
    post = last["asset_values"][aid]
    pnl = last["asset_pnl"][aid]
    ret = last["asset_returns"]
    tw = last["target_weights"][aid]
    prev_cap = (sim.capitals[aid] if last["t"] == 1 else sim.history[-2]["capital"][aid])
    df = pd.DataFrame({
        "Target weight": [f"{w:.0%}" for w in tw],
        "Rs invested (after trading cost)": [theme.inr(v) for v in pre],
        "Asset return": [f"{r * 100:+.3f}%" for r in ret],
        "Profit / loss": [theme.inr(v) for v in pnl],
        "Rs value after step": [theme.inr(v) for v in post],
        "Weight after drift": [f"{w:.1%}" for w in last["actual_weights"][aid]],
    }, index=cfg.ASSET_NAMES)
    st.dataframe(df, width="stretch")
    cost = last["txcost"][aid]
    st.markdown(
        f"**{pick}** started the step with **{theme.inr(prev_cap)}**, paid **{theme.inr(cost)}** to rebalance "
        f"({cfg.TRANSACTION_COST_RATE:.2%} of the amount traded), invested the rest according to the target weights, then the "
        f"asset returns moved the sleeve by **{theme.inr(float(pnl.sum()))}** to **{theme.inr(last['capital'][aid])}**.")
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.weight_waterfall(pnl), width="stretch", key="sim_wf")
    c2.plotly_chart(charts.reward_components_bar(last["reward_components"][aid]), width="stretch", key="sim_rc")
    rc = last["reward_components"][aid]
    st.caption(f"Reward = return term {rc['return']:+.3f}, risk {rc['risk']:+.3f}, trading cost {rc['transaction']:+.3f}, "
               f"drawdown {rc['drawdown']:+.3f}, liquidity {rc['liquidity']:+.3f} = {rc['unclipped']:+.3f} "
               f"(clipped to +/-{cfg.REWARD_CLIP:g}: {rc['clipped']:+.3f}).")


def _log(sim):
    if not sim.history:
        st.info("No steps yet.")
        return
    rows = []
    for r in sim.history:
        row = {"step": r["t"], "regime": cfg.REGIME_NAMES[r["current_regime"]],
               "agg equity wt": round(r["agg_equity_weight"], 3), "impact %": round(r["market_impact"] * 100, 4)}
        for a in sim.agent_ids:
            row[f"{sim.roles[a]} basket"] = cfg.ACTION_BASKET_NAMES[r["actions"][a]]
            row[f"{sim.roles[a]} Rs"] = round(r["capital"][a], 2)
            row[f"{sim.roles[a]} reward"] = round(r["rewards"][a], 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", height=360)
    st.download_button("Download step log (CSV)", df.to_csv(index=False).encode(), "simulation_step_log.csv", "text/csv")


# ------------------------------ page ----------------------------------- #
def render():
    ss = st.session_state
    sim = ss["sim"]
    st.title("Interactive Simulation")
    st.caption("Step the real environment forward one decision at a time. Watching never trains the agents - "
               "training lives in the RL Lab.")
    _settings(sim)

    b = st.columns([1, 1, 1, 1.3, 1.2, 1])
    b[0].button("Start", on_click=_start, disabled=sim.done or sim.running, type="primary", width="stretch")
    b[1].button("Pause", on_click=_pause, disabled=not sim.running, width="stretch")
    b[2].button("Step", on_click=_step, disabled=sim.done, width="stretch")
    with b[3]:
        c = st.columns([1, 1.2])
        c[0].number_input("n", 1, 100, 10, 1, key="sim_steps_n", label_visibility="collapsed")
        c[1].button("Step xN", on_click=_step_n, disabled=sim.done, width="stretch")
    b[4].button("Run to end", on_click=_to_end, disabled=sim.done, width="stretch")
    b[5].button("Reset", on_click=_reset, width="stretch")
    st.slider("Auto-play speed (steps per second)", 0.5, 5.0, 2.0, 0.5, key="sim_speed")

    interval = 1.0 / float(ss.get("sim_speed", 2.0)) if sim.running else None

    @st.fragment(run_every=interval)
    def live():
        s = ss["sim"]
        stepped = False
        if s.running and not s.done:
            s.step()
            stepped = True
        c = st.columns([2, 1, 1.4])
        c[0].progress(s.t / s.episode_length, text=s.summary_line())
        c[1].markdown(f"Next step's regime: {theme.regime_pill(s.current_regime)}" if not s.done else "Episode complete",
                      unsafe_allow_html=True)
        status = "RUNNING" if s.running else ("DONE" if s.done else "PAUSED / READY")
        c[2].markdown(f"Status: **{status}** &nbsp; Total: **{theme.inr(float(s.capital_series()[-1]))}**",
                      unsafe_allow_html=True)
        view = st.radio("View", VIEWS, horizontal=True, key="sim_view", label_visibility="collapsed")
        if view == VIEWS[0]:
            _overview(s)
        elif view == VIEWS[1]:
            _decisions(s)
        elif view == VIEWS[2]:
            _money_flow(s)
        else:
            _log(s)
        if stepped and s.done:
            st.rerun()          # full rerun so the Start/Pause buttons update

    live()
