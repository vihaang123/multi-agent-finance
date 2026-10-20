"""
server/pages.py - builds the content of every page as view blocks (see server/blocks.py).

Each function takes real simulation / training output and returns a list of blocks. The logic and wording are the
ones the original dashboard used; only the delivery changed (JSON blocks instead of Streamlit calls).
"""

import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config as cfg
from dashboard import charts, lab_tools, reference, theme, views
from dashboard.session import POLICY_LABELS
from environment.state_utils import BUCKET_LABELS, encode_state
from evaluation import analytics, metrics
from experiments.engine import EXPERIMENT_NAMES
from experiments.run_experiments import AGENT_IDS, AGENT_ROLES
from server import blocks as B

CAPS = {a: cfg.INITIAL_CAPITAL_PER_AGENT for a in AGENT_IDS}
SHORT = {"A": "A Random", "B": "B Fixed", "C": "C Single-agent Q", "D": "D 4-agent Q"}


def _basket_text(a: int) -> str:
    return " / ".join(f"{w:.0%}" for w in cfg.ACTION_BASKETS[a])


def _regime_name(r: int) -> str:
    return cfg.REGIME_NAMES[r]


# =========================================================================== #
# Dashboard
# =========================================================================== #
def dashboard_blocks(sim) -> List[dict]:
    traj = sim.history
    ids, roles, caps = sim.agent_ids, sim.roles, sim.capitals
    mdf = views.metrics_frame(traj, ids, roles, caps)
    total = mdf.loc[views.TOTAL]
    start_total = float(sum(caps.values()))
    ret = float(total["Total return"])
    tone = "pos" if ret > 0 else ("neg" if ret < 0 else "")
    agent_only = mdf.drop(index=views.TOTAL)
    spread = float(agent_only["Total return"].max() - agent_only["Total return"].min())
    if spread < 1e-9:
        best = B.kpi("Best agent", "Tie", "all four agents earned the same return")
    else:
        b = agent_only["Total return"].idxmax()
        best = B.kpi("Best agent", str(b), f"{theme.pct(agent_only.loc[b, 'Total return'], 2, True)} return")
    cur = sim.current_regime
    out = [B.kpis(
        B.kpi("Total portfolio value", theme.inr(total["Final value"]), f"started at {theme.inr(start_total)}"),
        B.kpi("Return so far", theme.pct(ret, 2, True), f"{theme.inr(total['Final value'] - start_total)} profit/loss", tone),
        B.kpi("Market regime now", _regime_name(cur), "regime for the next step" if not sim.done else "episode finished",
              pill=theme.REGIME_COLORS[_regime_name(cur)]),
        best,
        B.kpi("Max drawdown (total)", theme.pct(total["Max drawdown"], 2), f"trading costs {theme.inr(total['Transaction costs'])}"),
    )]
    if not traj:
        out.append(B.callout("note", "No steps have been simulated yet. Press **Run a full episode** above, or open the "
                                     "**Simulation** page to step through one decision at a time."))

    series = views.series_by_name(traj, ids, roles, caps)
    sleeves, total_only = views.split_series(series)
    colors = views.color_map(roles)
    regimes = sim.regime_path()
    sleeve_tab = [B.plot(charts.portfolio_value(sleeves, colors, regimes, initial=None,
                                                title="Value of each agent's sleeve - shaded bands show the market regime"))]
    if traj and views.sleeves_identical(series):
        sleeve_tab.append(B.caption("The four lines sit exactly on top of each other: in this episode all agents chose the same "
                                    "baskets, so their sleeves evolve identically (see 'Experiments' for the specialization finding)."))
    total_tab = [B.plot(charts.portfolio_value(total_only, colors, regimes, initial=start_total,
                                               title="Total portfolio value - dotted line is the Rs 10,00,000 start"))]
    rupees = views.current_rupees(sim)
    by_asset = np.sum(list(rupees.values()), axis=0)
    out.append(B.columns([B.tabs(("Each agent's sleeve (Rs)", sleeve_tab), ("Total portfolio (Rs)", total_tab))],
                         [B.plot(charts.allocation_donut(by_asset, "Where the money is right now"))], weights=[2.1, 1]))
    out.append(B.columns(
        [B.plot(charts.allocation_bars(rupees, "Current allocation by agent (Rs)"))],
        [B.md("**Agent performance (this episode)**"), B.table(views.display_metrics(mdf), index_name="Agent"),
         B.caption("Sharpe is episode-scale (per-step Sharpe x sqrt(steps)), NOT annualised. Volatility is the std of per-step "
                   "returns. All figures are from the synthetic simulator.")], weights=[1, 1.4]))

    alloc = B.switch("Agent", [(roles[a], [B.plot(charts.allocation_stack(
        views.weights_array(traj, a), regimes, f"{roles[a]}: actual portfolio weights over time"))]) for a in ids])
    ret_risk = [B.columns([B.plot(charts.metric_bars(agent_only, "Total return", "Total return (%)", colors))],
                          [B.plot(charts.metric_bars(agent_only, "Max drawdown", "Max drawdown (%)", colors))])]
    if sim.last is None:
        latest = [B.callout("note", "Nothing yet - run an episode first.")]
    else:
        last = sim.last
        rows = [[roles[a], cfg.ACTION_BASKET_NAMES[last["actions"][a]], _basket_text(last["actions"][a]),
                 round(last["rewards"][a], 3), theme.inr(last["capital"][a])] for a in ids]
        latest = [B.md(f"Step **{last['t']}** was generated in the **{_regime_name(last['current_regime'])}** regime; "
                       f"the market then moved to **{_regime_name(last['next_regime'])}**."),
                  B.table(rows, ["Agent", "Basket chosen", "Equity/Bonds/Gold/Cash", "Step reward", "Sleeve value"])]
    out.append(B.tabs(("Allocation over time", [alloc]), ("Return & risk by agent", ret_risk), ("Latest step", latest)))
    out.extend(scoreboard_blocks())
    return out


def scoreboard_blocks() -> List[dict]:
    """Where the learning ended up: the four experiments of the precomputed reference run, return against risk.
    Everything comes from data/reference_results.json (a real 5,000-episode run), never typed in."""
    ref = reference.load_reference_results()
    if not ref:
        return []
    res = ref["experiments"]
    pts, colors = {}, {}
    for e in "ABCD":
        if e in res:
            c = res[e]["summary"]["combined"]
            pts[SHORT[e]] = dict(ret=c["total_return_mean"], ret_std=c["total_return_std"], dd=c["max_drawdown_mean"])
            colors[SHORT[e]] = theme.EXPERIMENT_COLORS[e]
    if not pts:
        return []
    m = ref["meta"]
    lines = reading(res)
    return [
        B.heading("Where the learning ended up", 3),
        B.caption(f"Reference run: {m['train_episodes']:,} training episodes, then {cfg.N_EVAL_EPISODES} unseen markets with "
                  "learning switched off. Synthetic data."),
        B.columns([B.plot(charts.scoreboard_chart(pts, colors))],
                  [B.md("\n".join(f"- {line}" for line in lines[:3])),
                   B.caption("Full table, distributions and training curves are on the Experiments page.")], weights=[1.3, 1]),
    ]


# =========================================================================== #
# Simulation views
# =========================================================================== #
def sim_overview(sim) -> List[dict]:
    traj = sim.history
    series = views.series_by_name(traj, sim.agent_ids, sim.roles, sim.capitals)
    sleeves, total_only = views.split_series(series)
    colors = views.color_map(sim.roles)
    regimes = sim.regime_path()
    out = [B.columns(
        [B.plot(charts.portfolio_value(sleeves, colors, regimes, title="Each agent's sleeve (Rs)", height=360))],
        [B.plot(charts.portfolio_value(total_only, colors, regimes, initial=float(sum(sim.capitals.values())),
                                       title="Total portfolio (Rs)", height=360))])]
    right = ([B.plot(charts.asset_returns_bar(sim.last["base_returns"], sim.last["asset_returns"],
                                              f"Asset returns drawn for step {sim.last['t']} (%)"))]
             if sim.last else [B.callout("note", "Asset returns appear after the first step.")])
    out.append(B.columns([B.plot(charts.allocation_bars(views.current_rupees(sim), "Rupees held in each asset, by agent"))], right))
    return out


def sim_decisions(sim) -> List[dict]:
    last = sim.last
    if last is None:
        return [B.callout("note", "Press **Step** or **Start**. Each step the four agents look at their own state, pick a basket, "
                                  "and the shared market then pays them out.")]
    out = [B.md(f"**Step {last['t']}** - the market was in the **{_regime_name(last['current_regime'])}** regime "
                f"(it then moved to **{_regime_name(last['next_regime'])}**).")]
    items = []
    for aid in sim.agent_ids:
        role = sim.roles[aid]
        a = last["actions"][aid]
        body = (f"**State seen:** {last['state_desc'][aid]} (index {last['states'][aid]})\n"
                f"**Chose:** basket {a} \"{cfg.ACTION_BASKET_NAMES[a]}\" = "
                + " / ".join(f"{n} {w:.0%}" for n, w in zip(cfg.ASSET_NAMES, cfg.ACTION_BASKETS[a]))
                + f"\n**Reward received:** {last['rewards'][aid]:+.3f}")
        q = last["q_values"][aid]
        inner = ([B.plot(charts.q_values_bar(q, None, "Q-values the agent compared (highest = chosen)", 230))] if q is not None
                 else [B.caption("This policy does not use a Q-table (random / fixed).")])
        items.append({"title": role, "tag": POLICY_LABELS[last["policy_kind"][aid]], "color": theme.agent_color(role),
                      "body": body, "blocks": inner})
    out.append(B.cards(items))
    out.append(B.heading("How the agents affect each other", 5))
    out.append(B.md(
        f"The capital-weighted **aggregate equity weight** chosen this step was **{last['agg_equity_weight']:.1%}** "
        f"(neutral baseline {cfg.MARKET_IMPACT_BASELINE_EQUITY:.0%}). The one-sided crowding penalty "
        f"`-kappa * max(0, aggregate - baseline)^2` was **{last['market_impact'] * 100:+.4f}%** applied to the equity return "
        f"for **all** agents. The regime path and the base return draw are exogenous (identical for every policy on the same seed); "
        f"this crowding term is what couples the agents."))
    return out


def sim_money_flow(sim) -> List[dict]:
    last = sim.last
    if last is None:
        return [B.callout("note", "Take a step to see how an allocation percentage turns into rupees.")]
    options = []
    for aid in sim.agent_ids:
        role = sim.roles[aid]
        pre, post = last["asset_values_pre"][aid], last["asset_values"][aid]
        pnl, ret, tw = last["asset_pnl"][aid], last["asset_returns"], last["target_weights"][aid]
        prev_cap = sim.capitals[aid] if last["t"] == 1 else sim.history[-2]["capital"][aid]
        df = pd.DataFrame({
            "Target weight": [f"{w:.0%}" for w in tw],
            "Rs invested (after trading cost)": [theme.inr(v) for v in pre],
            "Asset return": [f"{r * 100:+.3f}%" for r in ret],
            "Profit / loss": [theme.inr(v) for v in pnl],
            "Rs value after step": [theme.inr(v) for v in post],
            "Weight after drift": [f"{w:.1%}" for w in last["actual_weights"][aid]],
        }, index=cfg.ASSET_NAMES)
        cost = last["txcost"][aid]
        rc = last["reward_components"][aid]
        options.append((role, [
            B.table(df, index_name="Asset"),
            B.md(f"**{role}** started the step with **{theme.inr(prev_cap)}**, paid **{theme.inr(cost)}** to rebalance "
                 f"({cfg.TRANSACTION_COST_RATE:.2%} of the amount traded), invested the rest according to the target weights, then the "
                 f"asset returns moved the sleeve by **{theme.inr(float(pnl.sum()))}** to **{theme.inr(last['capital'][aid])}**."),
            B.columns([B.plot(charts.weight_waterfall(pnl))], [B.plot(charts.reward_components_bar(rc))]),
            B.caption(f"Reward = return term {rc['return']:+.3f}, risk {rc['risk']:+.3f}, trading cost {rc['transaction']:+.3f}, "
                      f"drawdown {rc['drawdown']:+.3f}, liquidity {rc['liquidity']:+.3f} = {rc['unclipped']:+.3f} "
                      f"(clipped to +/-{cfg.REWARD_CLIP:g}: {rc['clipped']:+.3f}).")]))
    return [B.switch("Agent", options)]


def sim_log(sim) -> List[dict]:
    if not sim.history:
        return [B.callout("note", "No steps yet.")]
    rows = []
    for r in sim.history:
        row = {"step": r["t"], "regime": _regime_name(r["current_regime"]),
               "agg equity wt": round(r["agg_equity_weight"], 3), "impact %": round(r["market_impact"] * 100, 4)}
        for a in sim.agent_ids:
            row[f"{sim.roles[a]} basket"] = cfg.ACTION_BASKET_NAMES[r["actions"][a]]
            row[f"{sim.roles[a]} Rs"] = round(r["capital"][a], 2)
            row[f"{sim.roles[a]} reward"] = round(r["rewards"][a], 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    return [B.table(df, index_name=None, max_height=360),
            B.download("Download step log (CSV)", "simulation_step_log.csv", "text/csv", df.to_csv(index=False))]


SIM_VIEWS = {"overview": sim_overview, "decisions": sim_decisions, "money": sim_money_flow, "log": sim_log}


# =========================================================================== #
# Portfolio analytics
# =========================================================================== #
def episode_analytics(traj, agent_ids, roles, caps, vol_window: int = 10) -> List[dict]:
    if not traj:
        return [B.callout("note", "No steps to analyse yet - run an episode first.")]
    colors = views.color_map(roles)
    mf = views.metrics_frame(traj, agent_ids, roles, caps)
    series = views.series_by_name(traj, agent_ids, roles, caps)
    regimes = [s["current_regime"] for s in traj]
    sleeves, total_only = views.split_series(series)
    out = [B.table(views.display_metrics(mf), index_name="Portfolio"), B.caption(metrics.SHARPE_METHODOLOGY_NOTE)]

    growth = [B.columns([B.plot(charts.portfolio_value(sleeves, colors, regimes, title="Each agent's sleeve (Rs)"))],
                        [B.plot(charts.portfolio_value(total_only, colors, regimes, initial=float(sum(caps.values())),
                                                       title="Total portfolio (Rs)"))]),
              B.plot(charts.cumulative_return_chart(series, colors))]
    risk = [B.columns([B.plot(charts.drawdown_chart(series, colors))],
                      [B.plot(charts.rolling_vol_chart(series, colors, vol_window))]),
            B.plot(charts.step_return_hist(series, colors))]
    costs = views.txcosts_by_name(traj, agent_ids, roles)
    crows = {n: {"Total costs": theme.inr(float(np.sum(c))), "Per step (avg)": theme.inr(float(np.mean(c))),
                 "Steps with trading": int(np.sum(c > 1e-9))} for n, c in costs.items()}
    txc = [B.plot(charts.txcost_chart(costs, colors)), B.table(pd.DataFrame(crows).T, index_name="Portfolio"),
           B.caption(f"Cost model: {cfg.TRANSACTION_COST_RATE:.2%} of the amount traded (sum of absolute weight changes x capital).")]
    attr_opts = [("All agents", views.asset_pnl_total(traj, agent_ids))] + \
                [(roles[a], views.asset_pnl_total(traj, [a])) for a in agent_ids]
    attr = [B.switch("Agent", [(lbl, [B.plot(charts.asset_pnl_contribution(p))]) for lbl, p in attr_opts]),
            B.caption("Gross profit/loss earned by each asset sleeve (before subtracting transaction costs).")]
    ag = mf.drop(index=views.TOTAL, errors="ignore")
    comp = [B.columns([B.plot(charts.metric_bars(ag, "Total return", "Total return (%)", colors))],
                      [B.plot(charts.metric_bars(ag, "Max drawdown", "Max drawdown (%)", colors))]),
            B.columns([B.plot(charts.metric_bars(ag, "Sharpe", "Sharpe (episode-scale)", colors, pct_axis=False))],
                      [B.plot(charts.risk_return_scatter({n: (ag.loc[n, "Volatility"], ag.loc[n, "Total return"]) for n in ag.index}, colors))])]
    alloc = [B.switch("Agent", [(roles[a], [B.plot(charts.allocation_stack(views.weights_array(traj, a), regimes,
                                                                            f"{roles[a]}: portfolio weights"))]) for a in agent_ids])]
    out.append(B.tabs(("Growth & returns", growth), ("Risk", risk), ("Transaction costs", txc),
                      ("Where profit came from", attr), ("Agent comparison", comp), ("Allocation", alloc)))
    export = pd.DataFrame({n: v for n, v in series.items()})
    export.index.name = "step"
    out.append(B.download("Download value series (CSV)", "portfolio_value_series.csv", "text/csv", export.to_csv()))
    return out


def _oos_row(p: dict) -> dict:
    return {"Return mean": theme.pct(p["total_return_mean"], 2, True), "Return std": theme.pct(p["total_return_std"], 2),
            "Worst": theme.pct(p["total_return_min"], 2, True), "Best": theme.pct(p["total_return_max"], 2, True),
            "Loss episodes": theme.pct(p["prob_loss"], 0), "Volatility/step": theme.pct(p["volatility_mean"], 3),
            "Max drawdown": theme.pct(p["max_drawdown_mean"], 2), "Sharpe": f"{p['sharpe_mean']:.2f}",
            "Trading costs": theme.inr(p["total_txcost_mean"])}


def oos_blocks(result: dict, seeds: List[int], label: str) -> List[dict]:
    summ = result["summary"]
    rows = {AGENT_ROLES[a]: _oos_row(summ["per_agent"][a]) for a in AGENT_IDS}
    rows["Total (all 4)"] = _oos_row(summ["combined"])
    colors = {AGENT_ROLES[a]: theme.agent_color(AGENT_ROLES[a]) for a in AGENT_IDS}
    colors["Total"] = theme.TOTAL_COLOR
    per_seed = {AGENT_ROLES[int(k)]: v for k, v in summ["per_agent_returns_per_seed"].items()}
    per_seed["Total"] = summ["combined_returns_per_seed"]
    curves = {"Total": (summ["mean_equity_curve"], summ["std_equity_curve"])}
    return [
        B.md(f"Agents: **{label}** | {len(seeds)} held-out seeds ({seeds[0]}-{seeds[-1]}) | epsilon = 0, Q-tables frozen, no learning."),
        B.table(pd.DataFrame(rows).T, index_name="Portfolio"), B.caption(metrics.SHARPE_METHODOLOGY_NOTE),
        B.columns([B.plot(charts.returns_box(per_seed, colors))], [B.plot(charts.mean_curve_band(curves, colors))]),
    ]


# =========================================================================== #
# RL Lab
# =========================================================================== #
def lab_curves(log: Optional[pd.DataFrame], agents, smooth: int) -> List[dict]:
    if log is not None and len(log):
        win = max(1, min(int(smooth), len(log)))
        rows = {}
        for aid in AGENT_IDS:
            s = metrics.rl_training_summary(log, aid, window=min(100, len(log))) if len(log) > 1 else None
            rows[AGENT_ROLES[aid]] = {
                "Avg reward (last 100 ep.)": round(s["avg_reward_final_window"], 2) if s else None,
                "Reward trend (last 10% vs prior 10%)": round(s["reward_trend"], 2) if s and np.isfinite(s["reward_trend"]) else None,
                "Final epsilon": round(agents[aid].epsilon, 3),
                "Q-updates": int(agents[aid].n_updates),
                "States visited (of 108)": int((agents[aid].visit_counts.sum(axis=1) > 0).sum()),
            }
        return [B.plot(charts.learning_curves(log, AGENT_IDS, AGENT_ROLES, win)),
                B.columns([B.plot(charts.epsilon_curve(log, AGENT_IDS, AGENT_ROLES))],
                          [B.plot(charts.td_curve(log, AGENT_IDS, AGENT_ROLES, win))]),
                B.plot(charts.training_capital_curve(log, AGENT_IDS, AGENT_ROLES, win)),
                B.table(pd.DataFrame(rows).T, index_name="Agent"),
                B.caption("These are TRAINING-time (in-sample, exploring) numbers. Judge the policies on the "
                          "'Evaluate (out-of-sample)' tab.")]
    ref = reference.load_reference_results()
    if ref and "D" in ref["experiments"] and "training" in ref["experiments"]["D"]:
        tr = ref["experiments"]["D"]["training"]
        colors = {AGENT_ROLES[a]: theme.agent_color(AGENT_ROLES[a]) for a in AGENT_IDS}
        rew = {AGENT_ROLES[int(k)]: v for k, v in tr["reward"].items()}
        eps = {AGENT_ROLES[int(k)]: v for k, v in tr["epsilon"].items()}
        td = {AGENT_ROLES[int(k)]: v for k, v in tr["td"].items()}
        return [B.callout("note", f"Showing the **precomputed** training curves of the reference agents "
                                  f"({ref['meta']['train_episodes']:,} episodes, generated by `scripts/build_reference.py`). "
                                  "Press **Train from scratch** above to produce your own live curves."),
                B.plot(charts.training_curves_from_reference(rew, colors, "Episode reward (50-episode moving average)", "Sum of step rewards")),
                B.columns([B.plot(charts.training_curves_from_reference(eps, colors, "Exploration rate epsilon", "epsilon"))],
                          [B.plot(charts.training_curves_from_reference(td, colors, "Mean |TD error| (smoothed)", "|TD error|"))])]
    return [B.callout("note", "No training run yet - configure above and press Train from scratch.")]


def lab_inspect(agents, agent_idx: int, regime: int, eqb: int, bdb: int, gdb: int) -> List[dict]:
    aid = AGENT_IDS[agent_idx]
    role = AGENT_ROLES[aid]
    ag = agents[aid]
    s = encode_state(regime, eqb, bdb, gdb)
    q, v = ag.q_values(s), ag.visit_counts[s]
    tbl = pd.DataFrame({"Basket": cfg.ACTION_BASKET_NAMES, "Eq/Bd/Au/Cash": [_basket_text(i) for i in range(cfg.N_ACTIONS)],
                        "Q": np.round(q, 3), "Tried": v})
    visited_total = int(v.sum())
    if visited_total == 0:
        verdict = B.callout("warn", "This state was never visited during training, so its Q-values are still the initial value - "
                                    "the agent has learned nothing here (it breaks ties at random).")
    else:
        verdict = B.callout("ok", f"Greedy choice here: **{cfg.ACTION_BASKET_NAMES[int(np.argmax(q))]}** "
                                  f"(state tried {visited_total:,} times).")
    return [B.columns([B.plot(charts.q_values_bar(q, v, f"{role}: Q(state {s}, action) - {_regime_name(regime)} regime"))],
                      [B.table(tbl, index_name=None), verdict], weights=[1.5, 1]),
            B.caption(f"Table size {ag.q_table.shape[0]} states x {ag.q_table.shape[1]} actions = {ag.q_table.size} numbers; "
                      f"{int((ag.visit_counts.sum(axis=1) > 0).sum())} states visited, {int(ag.n_updates):,} updates applied.")]


def lab_policy(agents, agent_idx: int, mode: str) -> List[dict]:
    ag = agents[AGENT_IDS[agent_idx]]
    return [B.plot(charts.policy_heatmap(ag, mode)),
            B.md("**What each agent does in each regime** - visit-weighted share (%) of the greedy basket choice"),
            B.table(lab_tools.greedy_policy_by_regime(ag).round(1), index_name="Regime")]


def lab_walkthrough(agents, agent_idx: int, seed: int, n: int) -> List[dict]:
    aid = AGENT_IDS[agent_idx]
    rows = lab_tools.trace_updates(agents, aid, seed, n, cfg.EPISODE_LENGTH)
    df = pd.DataFrame(rows)
    ok = bool(df["matches_formula"].all())
    r0 = rows[0]
    show = df[["step", "state_desc", "basket", "reward", "Q_old", "max_Q_next", "target", "TD_error", "Q_new"]].round(4)
    return [B.table(show, index_name=None),
            B.math_text("Q_new = Q_old + alpha * [ r + gamma * max_a' Q(s',a') - Q_old ]"),
            B.md(f"Step 1: `{r0['Q_old']:.4f} + {r0['alpha']:.2f} x ({r0['reward']:.4f} + {r0['gamma']:.2f} x {r0['max_Q_next']:.4f} "
                 f"- {r0['Q_old']:.4f}) = {r0['Q_new']:.4f}`"),
            B.callout("ok" if ok else "error", "All displayed updates match the Q-learning formula." if ok
                      else "Mismatch between the formula and the applied update!")]


def lab_evaluate(result: dict, seeds: List[int], label: str) -> List[dict]:
    summ = result["summary"]
    rows = {}
    for aid in AGENT_IDS:
        p = summ["per_agent"][aid]
        rows[AGENT_ROLES[aid]] = {"Return (mean)": theme.pct(p["total_return_mean"], 2, True),
                                  "Return (std)": theme.pct(p["total_return_std"], 2),
                                  "Max drawdown": theme.pct(p["max_drawdown_mean"], 2), "Sharpe": f"{p['sharpe_mean']:.2f}",
                                  "Episodes with a loss": theme.pct(p["prob_loss"], 0)}
    hist = {AGENT_ROLES[int(k)]: v for k, v in summ["action_hist"].items()}
    out = [B.caption(f"Result for: {label}  |  {len(seeds)} held-out seeds"),
           B.table(pd.DataFrame(rows).T, index_name="Agent"),
           B.plot(charts.action_hist_heatmap(hist)),
           B.kpis(B.kpi("Specialization score (mean pairwise TV distance of basket choices)", f"{summ['specialization']:.3f}",
                        "0 = all four agents behave identically; 1 = completely different baskets.", ""))]
    if summ["specialization"] < 0.05:
        out.append(B.callout("warn", "**Honest reading:** the four agents converge to (almost) the same basket, so role "
                                     "specialization is essentially absent even though the reward weights differ. This is a known "
                                     "limitation of plain epsilon-greedy tabular Q-learning with transaction costs (see the "
                                     "Experiments page and README)."))
    return out


# =========================================================================== #
# Experiments
# =========================================================================== #
def comparison_table(res: dict) -> pd.DataFrame:
    rows = {}
    for e in "ABCD":
        if e not in res:
            continue
        c = res[e]["summary"]["combined"]
        rows[SHORT[e]] = {
            "Return mean": theme.pct(c["total_return_mean"], 2, True), "Return std": theme.pct(c["total_return_std"], 2),
            "Loss episodes": theme.pct(c["prob_loss"], 0), "Volatility/step": theme.pct(c["volatility_mean"], 3),
            "Max drawdown": theme.pct(c["max_drawdown_mean"], 2), "Sharpe (ep.-scale)": f"{c['sharpe_mean']:.2f}",
            "Final value": theme.inr(c["final_value_mean"]), "Trading costs": theme.inr(c["total_txcost_mean"]),
            "Specialization": f"{res[e]['summary']['specialization']:.3f}",
        }
    return pd.DataFrame(rows).T


def reading(res: dict) -> List[str]:
    """Plain-language observations generated from the numbers shown (not canned)."""
    out = []
    comb = {e: res[e]["summary"]["combined"] for e in res}
    best = max(comb, key=lambda e: comb[e]["total_return_mean"])
    safest = min(comb, key=lambda e: comb[e]["max_drawdown_mean"])
    out.append(f"Highest mean out-of-sample return: **{SHORT[best]}** ({theme.pct(comb[best]['total_return_mean'], 2, True)}). "
               f"Smallest average max drawdown: **{SHORT[safest]}** ({theme.pct(comb[safest]['max_drawdown_mean'], 2)}).")
    if "B" in comb:
        for e in ("C", "D"):
            if e in comb:
                diff = comb[e]["total_return_mean"] - comb["B"]["total_return_mean"]
                word = "beat" if diff > 0 else "did NOT beat"
                out.append(f"{SHORT[e]} {word} the fixed Balanced basket on mean return ({theme.pct(diff, 2, True)} difference), "
                           f"with {'lower' if comb[e]['volatility_mean'] < comb['B']['volatility_mean'] else 'higher'} volatility "
                           f"({theme.pct(comb[e]['volatility_mean'], 3)} vs {theme.pct(comb['B']['volatility_mean'], 3)} per step).")
    if "A" in comb:
        out.append(f"The random baseline lost money in {theme.pct(comb['A']['prob_loss'], 0)} of held-out episodes - "
                   "what an agent with no policy gets after transaction costs.")
    if "D" in res:
        modal = res["D"]["summary"]["modal_action"]
        names = {cfg.ACTION_BASKET_NAMES[int(v)] for v in modal.values()}
        spec = res["D"]["summary"]["specialization"]
        if len(names) == 1 and spec < 0.05:
            out.append(f"All four independent learners ended up favouring the same basket (**{names.pop()}**; "
                       f"specialization score {spec:.3f}), even though their reward weights encode different risk appetites. "
                       "Plain epsilon-greedy tabular Q-learning with transaction costs does not discover the sustained-commitment "
                       "strategies (e.g. always holding a growth basket) that would reward the risk-seeking role - "
                       "this is a genuine limitation of the Baseline V1 learner, reported rather than hidden.")
        else:
            out.append(f"The four D agents show a specialization score of {spec:.3f} (0 = identical behaviour).")
    return out


def experiments_report(res: dict, key: str, meta_line: str) -> List[dict]:
    if not res:
        return [B.callout("note", "No results yet.")]
    colors = {SHORT[e]: theme.EXPERIMENT_COLORS[e] for e in res}
    box = {SHORT[e]: res[e]["summary"]["combined_returns_per_seed"] for e in res}
    curves = {SHORT[e]: (res[e]["summary"]["mean_equity_curve"], res[e]["summary"]["std_equity_curve"]) for e in res}
    out = [B.md(meta_line), B.table(comparison_table(res), index_name="Experiment"),
           B.caption("Metrics are for the COMBINED Rs 10,00,000 portfolio (four Rs 2,50,000 sleeves in A, B, D; one Rs 10,00,000 account "
                     "in C), averaged over held-out episodes. Sharpe is episode-scale, not annualised. Synthetic data only."),
           B.columns([B.plot(charts.returns_box(box, colors))], [B.plot(charts.mean_curve_band(curves, colors))])]
    learned = [e for e in res if res[e].get("training")]
    if learned:
        out.append(B.heading("Training phase (in-sample) - kept separate from the evaluation above", 5))
        out.append(B.columns(*[[B.plot(charts.training_curves_from_reference(
            {SHORT[e]: res[e]["training"]["total_capital"]}, colors,
            f"{SHORT[e]}: training-phase capital (Rs, smoothed)", "Rs", 280))] for e in learned]))
    hist_rows = {}
    for e in res:
        for k, v in res[e]["summary"]["action_hist"].items():
            hist_rows[f"{SHORT[e]} / {res[e]['roles'][str(k)]}"] = v
    out.append(B.plot(charts.action_hist_heatmap(hist_rows, height=max(260, 30 * len(hist_rows) + 120))))
    out.append(B.heading("Reading these results", 5))
    out.append(B.md("\n".join(f"- {line}" for line in reading(res))))
    payload = {e: {"config": res[e]["config"], "summary": res[e]["summary"]} for e in res}
    out.append(B.download("Download results (JSON)", f"experiment_results_{key}.json", "application/json",
                          json.dumps(B.sanitize(payload), indent=1)))
    return out


# =========================================================================== #
# Learn (data-driven pieces; the prose lives in the front end)
# =========================================================================== #
def stationary_distribution() -> np.ndarray:
    P = np.asarray(cfg.REGIME_TRANSITION)
    vals, vecs = np.linalg.eig(P.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return v / v.sum()


def learn_static() -> dict:
    pi = stationary_distribution()
    tbl = pd.DataFrame(cfg.ACTION_BASKETS * 100, columns=[f"{a} %" for a in cfg.ASSET_NAMES],
                       index=cfg.ACTION_BASKET_NAMES).round(0).astype(int)
    w = pd.DataFrame(cfg.REWARD_WEIGHTS).T
    w.columns = ["alpha (return)", "beta (risk)", "lambda (cost)", "delta (drawdown)", "mu (liquidity)"]
    return {
        "transition": B.plot(charts.transition_heatmap()),
        "stationary": B.table(pd.DataFrame({"Share": [f"{p:.1%}" for p in pi]}, index=cfg.REGIME_NAMES), index_name="Regime"),
        "params": B.plot(charts.asset_return_params_heatmap()),
        "baskets": B.table(tbl, index_name="Action"),
        "reward_weights": B.table(w, index_name="Role"),
    }


def learn_regime_path(seed: int, steps: int) -> List[dict]:
    from environment.market import MarketSimulator
    m = MarketSimulator(np.random.default_rng(int(seed)))
    m.reset()
    path = [m.regime]
    for _ in range(int(steps) - 1):
        path.append(m.step_regime())
    counts = pd.Series([_regime_name(r) for r in path]).value_counts()
    return [B.plot(charts.regime_path_chart(path)),
            B.caption("Time spent: " + ", ".join(f"{k} {v / len(path):.0%}" for k, v in counts.items()))]
