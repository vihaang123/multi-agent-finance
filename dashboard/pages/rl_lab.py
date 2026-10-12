"""Page 3 - RL Lab: configure and run real Q-learning training, watch it learn, inspect Q-values."""

import math

import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from dashboard import charts, lab_tools, reference, theme, views
from dashboard.state import reset_to_reference_agents
from environment.state_utils import BUCKET_LABELS, encode_state
from evaluation import metrics
from experiments.engine import QHyperParams, evaluate_agents
from experiments.run_experiments import AGENT_IDS, AGENT_ROLES

MAX_LAB_EPISODES = 3000


def _hp_from_widgets() -> QHyperParams:
    ss = st.session_state
    return QHyperParams(alpha=float(ss["lab_alpha"]), gamma=float(ss["lab_gamma"]),
                        epsilon_initial=float(ss["lab_eps0"]), epsilon_min=float(ss["lab_epsmin"]),
                        epsilon_decay=float(ss["lab_decay"]), q_init=float(ss["lab_qinit"]))


def _refresh_simulation_with_current_agents():
    """After training, point the simulator at the (now different) agents and replay an example episode."""
    ss = st.session_state
    sim = ss["sim"]
    sim.agents = ss["agents"]
    sim.reset()
    sim.run_to_end()
    ss["an_eval"] = None


def _train(mode: str):
    ss = st.session_state
    hp = _hp_from_widgets()
    errs = hp.validate()
    n = int(ss["lab_n_episodes"])
    seed = int(ss["lab_seed"])
    impact = bool(ss["lab_impact"])
    ep_len = cfg.EPISODE_LENGTH
    if not 1 <= n <= MAX_LAB_EPISODES:
        errs.append(f"Training episodes must be between 1 and {MAX_LAB_EPISODES} in the app (use scripts/ for longer runs).")
    if seed in cfg.EVAL_SEEDS:
        errs.append(f"Training seed {seed} lies inside the held-out evaluation block {cfg.EVAL_SEED_START}-"
                    f"{cfg.EVAL_SEED_START + cfg.N_EVAL_EPISODES - 1}; pick another so evaluation stays out-of-sample.")
    if errs:
        for e in errs:
            st.error(e)
        return

    fresh = mode.startswith("Start fresh")
    if fresh:
        trainer, agents = lab_tools.new_trainer(hp, seed, ep_len, impact)
        offset, prev_log = 0, None
    else:
        agents = ss["agents"]
        lab_tools.apply_hyperparams(agents, hp)
        trainer = ss.get("lab_trainer") if ss.get("agents_source") == "lab" else None
        if trainer is None:
            trainer = lab_tools.trainer_for_existing(agents, seed, ep_len, impact)
        offset = int(ss.get("lab_total_episodes", 0)) if ss.get("agents_source") == "lab" else 0
        prev_log = ss.get("lab_log") if ss.get("agents_source") == "lab" else None

    bar = st.progress(0.0, text="Training...")
    holder = st.empty()

    def progress(done, total):
        bar.progress(done / total, text=f"Training episode {done}/{total}")

    with st.spinner("Q-learning in progress - every number below is computed by the real training loop..."):
        df, done, stopped = lab_tools.train_chunked(trainer, n, episode_offset=offset, progress=progress)
    bar.empty()
    log = df if prev_log is None else pd.concat([prev_log, df], ignore_index=True)

    ss["agents"] = agents
    ss["agents_source"] = "lab"
    ss["agents_label"] = (f"Trained in the RL Lab: {offset + done:,} episodes, seed {seed}"
                          + (", market impact on" if impact else ", market impact off"))
    ss["lab_trainer"], ss["lab_log"], ss["lab_total_episodes"] = trainer, log, offset + done
    _refresh_simulation_with_current_agents()
    if stopped:
        holder.warning(f"Stopped early after {done} of {n} episodes to stay inside the {lab_tools.TIME_BUDGET_SECONDS:.0f}s "
                       "time budget. The agents keep what they learned; press Continue to train further.")
    else:
        holder.success(f"Finished {done} episodes. The Simulation and Dashboard pages now use these agents.")


def _reset_agents():
    reset_to_reference_agents()


def _explain_epsilon(hp: QHyperParams) -> str:
    if hp.epsilon_decay >= 1.0 or hp.epsilon_min >= hp.epsilon_initial:
        return "epsilon never decays below its start value."
    ep = math.log(max(hp.epsilon_min, 1e-9) / hp.epsilon_initial) / math.log(hp.epsilon_decay)
    return f"epsilon reaches its floor after about {ep:,.0f} episodes."


def render():
    ss = st.session_state
    st.title("RL Lab")
    st.caption("Train real tabular Q-learning agents, watch the learning curves and open the Q-tables. "
               "Nothing is pre-baked: pressing Train runs the same training loop that produced the reference results.")

    # -------------------- configuration -------------------- #
    with st.container(border=True):
        st.markdown("#### 1. Configure")
        a, b, c = st.columns(3)
        a.number_input(f"Training episodes (max {MAX_LAB_EPISODES})", 10, MAX_LAB_EPISODES, 500, 50, key="lab_n_episodes",
                       help="Each episode is 100 market steps with all 4 agents acting and learning. "
                            "Roughly 15-30 s per 1000 episodes depending on the server.")
        b.number_input("Training market seed", 0, 1_000_000, cfg.TRAIN_SEED, 1, key="lab_seed",
                       help="Seed of the single continuing market stream used for training. Must stay outside 51-100 (held-out evaluation).")
        c.checkbox("Market impact between agents", True, key="lab_impact")
        h1, h2, h3 = st.columns(3)
        h1.slider("Learning rate  alpha", 0.01, 1.0, cfg.ALPHA_LR, 0.01, key="lab_alpha",
                  help="How far each update moves Q(s,a) toward its target.")
        h2.slider("Discount factor  gamma", 0.0, 0.999, cfg.GAMMA, 0.005, key="lab_gamma",
                  help="How much future reward counts relative to immediate reward.")
        h3.slider("Initial Q-value (optimism)", -10.0, 30.0, cfg.Q_INIT_VALUE, 0.5, key="lab_qinit")
        e1, e2, e3 = st.columns(3)
        e1.slider("Starting exploration  epsilon", 0.05, 1.0, cfg.EPSILON_INITIAL, 0.05, key="lab_eps0")
        e2.slider("Minimum epsilon", 0.0, 0.5, cfg.EPSILON_MIN, 0.01, key="lab_epsmin")
        e3.slider("Epsilon decay per episode", 0.990, 1.0, cfg.EPSILON_DECAY, 0.0005, format="%.4f", key="lab_decay")
        hp = _hp_from_widgets()
        st.caption(_explain_epsilon(hp) + "  Reward weights per role, the 7 baskets, the 108 states and the market model are "
                   "fixed project configuration (see the Learn page).")

        st.markdown("#### 2. Run")
        has_lab = ss.get("agents_source") == "lab"
        r1, r2, r3, r4 = st.columns([1.1, 1.1, 1.1, 1.2])
        r1.button("Train from scratch", type="primary", on_click=_train, args=("Start fresh",), width="stretch",
                  help="Creates 4 brand-new agents (all Q-values = initial value) and trains them.")
        r2.button("Continue training", on_click=_train, args=("Continue",), width="stretch",
                  help="Keeps the current Q-tables and epsilon; alpha, gamma, epsilon floor and decay are updated.")
        r3.button("Restore reference agents", on_click=_reset_agents, width="stretch",
                  help="Go back to the precomputed 5000-episode agents.")
        r4.markdown(f'<div class="small">Active agents: <b>{ss["agents_label"]}</b></div>', unsafe_allow_html=True)

    agents = ss["agents"]
    tabs = st.tabs(["Learning curves", "Q-table inspector", "Policy maps", "Update walkthrough", "Evaluate (out-of-sample)"])

    # -------------------- learning curves -------------------- #
    with tabs[0]:
        log = ss.get("lab_log") if ss.get("agents_source") == "lab" else None
        if log is not None and len(log):
            win = st.slider("Smoothing window (episodes)", 1, 200, min(50, max(1, len(log) // 4)), key="lab_smooth")
            st.plotly_chart(charts.learning_curves(log, AGENT_IDS, AGENT_ROLES, win), width="stretch", key="lab_lc")
            c1, c2 = st.columns(2)
            c1.plotly_chart(charts.epsilon_curve(log, AGENT_IDS, AGENT_ROLES), width="stretch", key="lab_eps")
            c2.plotly_chart(charts.td_curve(log, AGENT_IDS, AGENT_ROLES, win), width="stretch", key="lab_td")
            st.plotly_chart(charts.training_capital_curve(log, AGENT_IDS, AGENT_ROLES, win), width="stretch", key="lab_cap")
            rows = {}
            for aid in AGENT_IDS:
                s = metrics.rl_training_summary(log, aid, window=min(100, len(log))) if len(log) > 1 else None
                rows[AGENT_ROLES[aid]] = {
                    "Avg reward (last 100 ep.)": round(s["avg_reward_final_window"], 2) if s else float("nan"),
                    "Reward trend (last 10% vs prior 10%)": round(s["reward_trend"], 2) if s and np.isfinite(s["reward_trend"]) else float("nan"),
                    "Final epsilon": round(agents[aid].epsilon, 3),
                    "Q-updates": int(agents[aid].n_updates),
                    "States visited (of 108)": int((agents[aid].visit_counts.sum(axis=1) > 0).sum()),
                }
            st.dataframe(pd.DataFrame(rows).T, width="stretch")
            st.caption("These are TRAINING-time (in-sample, exploring) numbers. Judge the policies on the 'Evaluate (out-of-sample)' tab.")
        else:
            ref = reference.load_reference_results()
            if ref and "D" in ref["experiments"] and "training" in ref["experiments"]["D"]:
                tr = ref["experiments"]["D"]["training"]
                st.markdown('<div class="note">Showing the <b>precomputed</b> training curves of the reference agents '
                            f'({ref["meta"]["train_episodes"]:,} episodes, generated by <code>scripts/build_reference.py</code>). '
                            'Press <b>Train from scratch</b> above to produce your own live curves.</div>', unsafe_allow_html=True)
                colors = {AGENT_ROLES[a]: theme.agent_color(AGENT_ROLES[a]) for a in AGENT_IDS}
                rew = {AGENT_ROLES[int(k)]: v for k, v in tr["reward"].items()}
                st.plotly_chart(charts.training_curves_from_reference(rew, colors, "Episode reward (50-episode moving average)", "Sum of step rewards"),
                                width="stretch", key="lab_ref_rew")
                c1, c2 = st.columns(2)
                eps = {AGENT_ROLES[int(k)]: v for k, v in tr["epsilon"].items()}
                c1.plotly_chart(charts.training_curves_from_reference(eps, colors, "Exploration rate epsilon", "epsilon"),
                                width="stretch", key="lab_ref_eps")
                td = {AGENT_ROLES[int(k)]: v for k, v in tr["td"].items()}
                c2.plotly_chart(charts.training_curves_from_reference(td, colors, "Mean |TD error| (smoothed)", "|TD error|"),
                                width="stretch", key="lab_ref_td")
            else:
                st.info("No training run yet - configure above and press Train from scratch.")

    # -------------------- Q-table inspector -------------------- #
    with tabs[1]:
        names = [AGENT_ROLES[a] for a in AGENT_IDS]
        c0, c1, c2, c3, c4 = st.columns(5)
        pick = c0.selectbox("Agent", names, key="lab_inspect_agent")
        regime = c1.selectbox("Regime", list(range(4)), format_func=lambda i: cfg.REGIME_NAMES[i], key="lab_regime")
        eqb = c2.selectbox("Equity weight", list(range(3)), format_func=lambda i: BUCKET_LABELS[i], key="lab_eq")
        bdb = c3.selectbox("Bond weight", list(range(3)), format_func=lambda i: BUCKET_LABELS[i], key="lab_bond")
        gdb = c4.selectbox("Gold weight", list(range(3)), format_func=lambda i: BUCKET_LABELS[i], key="lab_gold")
        ag = agents[AGENT_IDS[names.index(pick)]]
        s = encode_state(regime, eqb, bdb, gdb)
        q, v = ag.q_values(s), ag.visit_counts[s]
        l, r = st.columns([1.5, 1])
        l.plotly_chart(charts.q_values_bar(q, v, f"{pick}: Q(state {s}, action) - {cfg.REGIME_NAMES[regime]} regime"),
                       width="stretch", key="lab_qbar")
        with r:
            tbl = pd.DataFrame({"Basket": cfg.ACTION_BASKET_NAMES,
                                "Eq/Bd/Au/Cash": [" / ".join(f"{w:.0%}" for w in row) for row in cfg.ACTION_BASKETS],
                                "Q": np.round(q, 3), "Tried": v})
            st.dataframe(tbl, width="stretch", hide_index=True, height=290)
            visited_total = int(v.sum())
            if visited_total == 0:
                st.warning("This state was never visited during training, so its Q-values are still the initial value - "
                           "the agent has learned nothing here (it breaks ties at random).")
            else:
                st.success(f"Greedy choice here: **{cfg.ACTION_BASKET_NAMES[int(np.argmax(q))]}** "
                           f"(state tried {visited_total:,} times).")
        st.caption(f"Table size {ag.q_table.shape[0]} states x {ag.q_table.shape[1]} actions = {ag.q_table.size} numbers; "
                   f"{int((ag.visit_counts.sum(axis=1) > 0).sum())} states visited, {int(ag.n_updates):,} updates applied.")

    # -------------------- policy maps -------------------- #
    with tabs[2]:
        names = [AGENT_ROLES[a] for a in AGENT_IDS]
        m1, m2 = st.columns(2)
        pick = m1.selectbox("Agent", names, key="lab_heat_agent")
        mode = m2.radio("Map", ["greedy", "maxq", "visits"], horizontal=True, key="lab_heat_mode",
                        format_func={"greedy": "Learned basket", "maxq": "Best Q-value", "visits": "Visit counts"}.get)
        ag = agents[AGENT_IDS[names.index(pick)]]
        st.plotly_chart(charts.policy_heatmap(ag, mode), width="stretch", key="lab_heat")
        st.markdown("**What each agent does in each regime** - visit-weighted share (%) of the greedy basket choice")
        st.dataframe(lab_tools.greedy_policy_by_regime(ag).round(1), width="stretch")

    # -------------------- update walkthrough -------------------- #
    with tabs[3]:
        st.markdown("Run the first few steps of an episode on a **copy** of the current agents and see each Q-update's "
                    "arithmetic. The result is checked against the real update rule.")
        w1, w2, w3 = st.columns(3)
        pick = w1.selectbox("Agent", [AGENT_ROLES[a] for a in AGENT_IDS], key="lab_trace_agent")
        n = w2.number_input("Steps to show", 1, 20, 5, 1, key="lab_trace_n")
        seed = w3.number_input("Episode seed", 0, 1_000_000, 7, 1, key="lab_trace_seed")
        if st.button("Show the updates", key="lab_trace_btn"):
            aid = [a for a in AGENT_IDS if AGENT_ROLES[a] == pick][0]
            rows = lab_tools.trace_updates(agents, aid, int(seed), int(n), cfg.EPISODE_LENGTH)
            df = pd.DataFrame(rows)
            ok = bool(df["matches_formula"].all())
            st.dataframe(df[["step", "state_desc", "basket", "reward", "Q_old", "max_Q_next", "target", "TD_error", "Q_new"]]
                         .round(4), width="stretch", hide_index=True)
            r0 = rows[0]
            st.latex(r"Q_{new} = Q_{old} + \alpha\,\big[\,r + \gamma \max_{a'} Q(s',a') - Q_{old}\,\big]")
            st.markdown(f"Step 1: `{r0['Q_old']:.4f} + {r0['alpha']:.2f} x ({r0['reward']:.4f} + {r0['gamma']:.2f} x {r0['max_Q_next']:.4f} "
                        f"- {r0['Q_old']:.4f}) = {r0['Q_new']:.4f}`")
            (st.success if ok else st.error)("All displayed updates match the Q-learning formula." if ok
                                              else "Mismatch between the formula and the applied update!")

    # -------------------- evaluation -------------------- #
    with tabs[4]:
        st.markdown("Freeze the Q-tables (epsilon = 0, no learning) and play brand-new markets the agents have never seen.")
        e1, e2 = st.columns(2)
        e1.number_input("Held-out episodes", 5, 200, 50, 5, key="lab_eval_n")
        e2.caption(f"Evaluation seeds {cfg.EVAL_SEED_START} ... {cfg.EVAL_SEED_START} + n - 1 (disjoint from the training seed).")
        if st.button("Evaluate current agents", key="lab_eval_btn", type="primary"):
            n = int(ss["lab_eval_n"])
            seeds = list(range(cfg.EVAL_SEED_START, cfg.EVAL_SEED_START + n))
            caps = {a: cfg.INITIAL_CAPITAL_PER_AGENT for a in AGENT_IDS}
            with st.spinner("Evaluating on held-out markets..."):
                res = evaluate_agents(agents, AGENT_IDS, AGENT_ROLES, caps, seeds,
                                      enable_market_impact=bool(ss.get("lab_impact", True)))
            res["label"] = ss["agents_label"]
            res["seeds"] = seeds
            ss["an_eval"] = res
        res = ss.get("an_eval")
        if res:
            summ = res["summary"]
            st.caption(f"Result for: {res['label']}  |  {len(res['seeds'])} held-out seeds")
            rows = {}
            for aid in AGENT_IDS:
                p = summ["per_agent"][aid]
                rows[AGENT_ROLES[aid]] = {"Return (mean)": theme.pct(p["total_return_mean"], 2, True),
                                          "Return (std)": theme.pct(p["total_return_std"], 2),
                                          "Max drawdown": theme.pct(p["max_drawdown_mean"], 2),
                                          "Sharpe": f"{p['sharpe_mean']:.2f}",
                                          "Episodes with a loss": theme.pct(p["prob_loss"], 0)}
            st.dataframe(pd.DataFrame(rows).T, width="stretch")
            hist = {AGENT_ROLES[int(k)]: v for k, v in summ["action_hist"].items()}
            st.plotly_chart(charts.action_hist_heatmap(hist), width="stretch", key="lab_eval_hist")
            st.metric("Specialization score (mean pairwise TV distance of basket choices)", f"{summ['specialization']:.3f}",
                      help="0 = all four agents behave identically; 1 = completely different baskets.")
            if summ["specialization"] < 0.05:
                st.markdown('<div class="warn"><b>Honest reading:</b> the four agents converge to (almost) the same basket, so '
                            'role specialization is essentially absent even though the reward weights differ. '
                            'This is a known limitation of plain epsilon-greedy tabular Q-learning with transaction costs '
                            '(see the Experiments page and README).</div>', unsafe_allow_html=True)
        else:
            st.info("Press Evaluate to run the out-of-sample test. The result also feeds the Portfolio Analytics page.")
