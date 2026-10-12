"""Page 6 - Learn: the ideas behind the project in plain language, with small live calculators."""

import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from dashboard import charts, theme
from environment.market import MarketSimulator
from environment.state_utils import BUCKET_LABELS, decode_state, describe_state, encode_state

WORKFLOW_DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.5;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#c9c8c3", fillcolor="#ffffff"];
  edge [fontname="Helvetica", fontsize=9, color="#52514e"];
  State [label="1. Each agent sees\\nits own state s\\n(regime + its weights)", fillcolor="#eef4fd"];
  Act   [label="2. Pick an action a\\n(one of 7 baskets)\\nepsilon-greedy on Q(s,.)", fillcolor="#eef4fd"];
  Env   [label="3. Shared market\\nrebalance + trading costs\\ncrowding impact on equity\\nreturns drawn for the regime", fillcolor="#fff6e0"];
  Rew   [label="4. Each agent gets its\\nown reward r and\\nnext state s'", fillcolor="#e8f6f0"];
  Upd   [label="5. Update own Q-table\\nQ(s,a) <- Q + alpha*(r + gamma*max Q(s') - Q)", fillcolor="#e8f6f0"];
  State -> Act -> Env -> Rew -> Upd;
  Upd -> State [label="next step", style=dashed];
}
"""


def _stationary():
    P = np.asarray(cfg.REGIME_TRANSITION)
    vals, vecs = np.linalg.eig(P.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return v / v.sum()


def _tab_big_picture():
    st.markdown("""
**The problem.** You have Rs 10,00,000 to split between **Equity, Bonds, Gold and Cash**. The market changes mood
(calm, booming, crashing, jittery) and every trade costs a little. How should the money be allocated over time?

**The twist.** Instead of one decision-maker we have **four agents**, each with its own personality (Growth, Conservative,
Balanced, RiskLiquidity) and its own Rs 2,50,000. They all trade in the **same** market and all learn **by trial and error**.
""")
    st.graphviz_chart(WORKFLOW_DOT, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Markov Decision Process (MDP)")
        st.markdown("""
An MDP is the rulebook for **one** learner acting in a world: *states* it can be in, *actions* it can take, *rewards* it
receives, and *transition probabilities* for how the world moves next. The key idea (**Markov property**): the next
situation depends only on the *current* state and action, not on the whole history.

Experiment C (a single agent managing all the money) is exactly an MDP.
""")
    with c2:
        st.markdown("#### Markov Game (multi-agent)")
        st.markdown("""
With several agents acting **at the same time**, the next state and each reward depend on **everyone's** actions together.
That is a *Markov game*. Here the agents are coupled through the market: if they all pile into equity, the combined
buying pressure lowers the equity return for **all of them** (market impact).

Each agent learns **independently** (**Independent Q-learning**): it treats the others as part of the environment.
Simple, but the environment then keeps shifting as the others learn - that is why multi-agent learning is harder.
""")
    st.markdown('<div class="note"><b>Where the simulation is exogenous vs endogenous.</b> The sequence of market regimes '
                'and the base random return draws are the same for every policy on a given seed (exogenous). Only the '
                'crowding penalty on equity depends on the agents\' choices (endogenous). This keeps comparisons fair.</div>',
                unsafe_allow_html=True)


def _tab_market():
    st.markdown("""
The market is **synthetic**. A hidden **regime** - *Bear, Neutral, Bull or HighVol* - decides how the four assets behave that day.
Regimes change according to a **Markov chain**: tomorrow's regime depends only on today's, with the probabilities below.
""")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.plotly_chart(charts.transition_heatmap(), width="stretch", key="learn_trans")
    with c2:
        pi = _stationary()
        st.markdown("**Long-run share of time in each regime** (stationary distribution, computed from the matrix):")
        st.dataframe(pd.DataFrame({"Regime": cfg.REGIME_NAMES, "Share": [f"{p:.1%}" for p in pi]}).set_index("Regime"),
                     width="stretch")
        st.caption("Rows add up to 1. E.g. a Bull day is followed by another Bull day 65% of the time - regimes are 'sticky'.")
    st.plotly_chart(charts.asset_return_params_heatmap(), width="stretch", key="learn_params")
    st.caption("Each step, every asset return is drawn from a Normal distribution whose mean and spread depend on the regime. "
               "Equity wins in Bull, loses in Bear; Bonds and Gold cushion Bear; HighVol is noisy for everything.")
    st.markdown("##### Try the regime chain")
    a, b = st.columns(2)
    seed = a.number_input("Seed", 0, 100000, 1, 1, key="learn_reg_seed")
    n = b.slider("Steps", 10, 200, 60, 10, key="learn_reg_n")
    m = MarketSimulator(np.random.default_rng(int(seed)))
    m.reset()
    path = [m.regime]
    for _ in range(int(n) - 1):
        path.append(m.step_regime())
    st.plotly_chart(charts.regime_path_chart(path), width="stretch", key="learn_path")
    counts = pd.Series([cfg.REGIME_NAMES[r] for r in path]).value_counts()
    st.caption("Time spent: " + ", ".join(f"{k} {v / len(path):.0%}" for k, v in counts.items()))


def _tab_state_action():
    st.markdown("""
**State space (108 states).** A state is what an agent can 'see' when deciding:

`state = (market regime) x (equity weight bucket) x (bond weight bucket) x (gold weight bucket)` = 4 x 3 x 3 x 3 = **108**.

A weight bucket is **Low** (below 20%), **Medium** (20%-45%) or **High** (45%+). Agents see *their own* weights, not other agents'.
""")
    c = st.columns(4)
    r = c[0].selectbox("Regime", range(4), format_func=lambda i: cfg.REGIME_NAMES[i], key="learn_s_r")
    e = c[1].selectbox("Equity", range(3), format_func=lambda i: BUCKET_LABELS[i], key="learn_s_e")
    b = c[2].selectbox("Bonds", range(3), format_func=lambda i: BUCKET_LABELS[i], key="learn_s_b")
    g = c[3].selectbox("Gold", range(3), format_func=lambda i: BUCKET_LABELS[i], key="learn_s_g")
    s = encode_state(r, e, b, g)
    st.markdown(f"State index = {r} x 27 + {e} x 9 + {b} x 3 + {g} = **{s}** &nbsp; ({describe_state(s)}); decoded back: `{decode_state(s)}`")
    st.markdown("""
**Action space (7 baskets).** Choosing an action means choosing a target mix. Predefined baskets keep the table small (7 actions
instead of an infinite number of weight combinations):
""")
    tbl = pd.DataFrame(cfg.ACTION_BASKETS * 100, columns=[f"{a} %" for a in cfg.ASSET_NAMES], index=cfg.ACTION_BASKET_NAMES).round(0).astype(int)
    tbl.index.name = "Action"
    st.dataframe(tbl, width="stretch")
    st.caption("After choosing, the portfolio is rebalanced to these weights; the trading cost is "
               f"{cfg.TRANSACTION_COST_RATE:.2%} of the amount moved. Then returns make the weights drift, which is what the next state shows.")


def _tab_reward():
    st.markdown("A good reward tells each agent what 'good' means **for its personality**. Per step:")
    st.latex(r"r_i = \alpha_i\cdot 100\cdot\text{log return} - \beta_i\cdot 100\cdot\text{risk} - \lambda_i\cdot 100\cdot\text{trading cost}"
             r" - \delta_i\cdot 100\cdot\text{new drawdown} + \mu_i\cdot\text{cash weight}")
    st.markdown("then clipped to [-5, +5]. The weights per role (from `config.py`):")
    w = pd.DataFrame(cfg.REWARD_WEIGHTS).T
    w.columns = ["alpha (return)", "beta (risk)", "lambda (cost)", "delta (drawdown)", "mu (liquidity)"]
    st.dataframe(w, width="stretch")
    st.markdown("**Reward calculator** - plug in a situation and see each term (uses the real config):")
    c = st.columns(3)
    role = c[0].selectbox("Role", list(cfg.REWARD_WEIGHTS), key="learn_r_role")
    basket = c[1].selectbox("Basket chosen", range(cfg.N_ACTIONS), index=2, format_func=lambda i: cfg.ACTION_BASKET_NAMES[i], key="learn_r_basket")
    regime = c[2].selectbox("Regime", range(4), index=2, format_func=lambda i: cfg.REGIME_NAMES[i], key="learn_r_regime")
    d = st.columns(3)
    lr = d[0].slider("Portfolio log return this step (%)", -3.0, 3.0, 0.5, 0.05, key="learn_r_ret")
    tc = d[1].slider("Trading cost (% of capital)", 0.0, 0.3, 0.05, 0.01, key="learn_r_tc")
    dd = d[2].slider("New drawdown created this step (%)", 0.0, 3.0, 0.0, 0.1, key="learn_r_dd")
    wt = cfg.REWARD_WEIGHTS[role]
    weights = cfg.ACTION_BASKETS[basket]
    sigma = np.array([cfg.ASSET_RETURN_PARAMS[regime][a][1] for a in range(cfg.N_ASSETS)])
    risk = float(weights @ sigma)
    terms = {
        "Return": wt["alpha"] * (lr / 100) * cfg.REWARD_SCALE,
        "Risk penalty": -wt["beta"] * risk * cfg.REWARD_SCALE,
        "Trading cost": -wt["lam"] * (tc / 100) * cfg.REWARD_SCALE,
        "Drawdown penalty": -wt["delta"] * (dd / 100) * cfg.REWARD_SCALE,
        "Liquidity bonus": wt["liquidity_mu"] * weights[cfg.CASH],
    }
    total = sum(terms.values())
    clipped = float(np.clip(total, -cfg.REWARD_CLIP, cfg.REWARD_CLIP))
    st.dataframe(pd.DataFrame({"Points": {k: round(v, 3) for k, v in terms.items()}}), width="stretch")
    st.markdown(f"Risk of this basket in this regime = weights . sigma = **{risk:.4f}**. "
                f"Total = **{total:+.3f}** -> after clipping **{clipped:+.3f}**.")
    st.caption("Growth cares mostly about return (alpha 3.0, tiny risk penalty); Conservative weighs risk and drawdown heavily; "
               "RiskLiquidity earns a bonus for holding cash. These are fixed project parameters, not tuned to flatter results.")


def _tab_qlearning():
    st.markdown("""
**Q-learning** keeps a table `Q(s, a)`: *'how good is it, in the long run, to take action a when in state s?'*.
Ours has 108 rows (states) x 7 columns (actions). It starts with a generous **optimistic** value, then each real experience
nudges one cell:
""")
    st.latex(r"Q(s,a)\;\leftarrow\;Q(s,a)+\alpha\Big[\,r+\gamma\max_{a'}Q(s',a')-Q(s,a)\Big]")
    st.markdown("""
- **alpha** (learning rate): how big a step to take toward the new evidence.
- **gamma** (discount): how much the future matters compared with the immediate reward.
- The bracket is the **TD error** = (what happened) - (what we expected).
- **Epsilon-greedy**: with probability epsilon try a random basket (explore), otherwise take the best-known one (exploit).
  Epsilon starts at 1.0 and shrinks every episode.
""")
    st.markdown("**Update calculator:**")
    c = st.columns(5)
    q = c[0].number_input("Q(s,a) now", value=2.0, step=0.5, key="learn_q_q")
    r = c[1].number_input("Reward r", value=1.0, step=0.5, key="learn_q_r")
    mq = c[2].number_input("max Q(s',.)", value=3.0, step=0.5, key="learn_q_m")
    al = c[3].number_input("alpha", 0.01, 1.0, 0.10, 0.01, key="learn_q_a")
    ga = c[4].number_input("gamma", 0.0, 1.0, 0.95, 0.01, key="learn_q_g")
    target = r + ga * mq
    td = target - q
    st.markdown(f"target = {r:g} + {ga:g} x {mq:g} = **{target:.3f}**; TD error = {target:.3f} - {q:g} = **{td:+.3f}**; "
                f"new Q = {q:g} + {al:g} x {td:+.3f} = **{q + al * td:.4f}**")
    st.markdown("""
**Independent multi-agent Q-learning** = each agent owns a private table and learns only from its own (state, action, reward).
Nothing is shared except the market.

**Honest limitations of this baseline** (also in the README):
- Plain epsilon-greedy rarely produces long runs of the *same* action, but sustained commitment is exactly what pays when trading is costly. In our runs all four learners end up favouring a cash-heavy basket and **do not specialise by role**, and a plain fixed Balanced basket beats them out-of-sample.
- Tabular Q-learning cannot generalise between similar states; a bigger state space would need function approximation (not used here).
- All data is synthetic, so none of this says anything about real markets.
""")


def render():
    st.title("Learn: how it works")
    st.caption("Plain-language explanations of every moving part, with small calculators that use the project's real configuration.")
    tabs = st.tabs(["Big picture", "Market & regimes", "States & actions", "Reward", "Q-learning"])
    with tabs[0]:
        _tab_big_picture()
    with tabs[1]:
        _tab_market()
    with tabs[2]:
        _tab_state_action()
    with tabs[3]:
        _tab_reward()
    with tabs[4]:
        _tab_qlearning()
