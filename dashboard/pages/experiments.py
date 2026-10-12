"""Page 5 - Experiments: Random / Fixed / Single-agent Q / 4-agent Independent Q, train vs out-of-sample."""

import json

import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from dashboard import charts, reference, theme
from experiments.engine import (EXPERIMENT_NAMES, LEARNING_EXPERIMENTS, MAX_TRAIN_EPISODES, QHyperParams, RunConfig,
                                run_experiment, training_curves)

MAX_UI_EPISODES = 3000
VIEW_LIVE = "Your live run"
VIEW_REF = "Precomputed reference (5000 episodes)"
SHORT = {"A": "A Random", "B": "B Fixed", "C": "C Single-agent Q", "D": "D 4-agent Q"}


def _config_from_widgets(exp: str) -> RunConfig:
    ss = st.session_state
    hp = QHyperParams(alpha=float(ss["ex_alpha"]), gamma=float(ss["ex_gamma"]), epsilon_decay=float(ss["ex_eps_decay"]))
    return RunConfig(experiment=exp, n_train_episodes=int(ss["ex_episodes"]), train_seed=int(ss["ex_train_seed"]),
                     eval_seed_start=int(ss["ex_eval_start"]), n_eval_episodes=int(ss["ex_eval_n"]),
                     episode_length=int(ss["ex_ep_len"]), fixed_action=int(ss["ex_fixed_basket"]),
                     enable_market_impact=bool(ss["ex_impact"]), hp=hp)


def _run_selected():
    ss = st.session_state
    selected = [e for e in "ABCD" if e in ss.get("ex_selected", [])]
    if not selected:
        st.error("Select at least one experiment.")
        return
    configs, errors = {}, []
    for e in selected:
        rc = _config_from_widgets(e)
        errs = rc.validate()
        if e in LEARNING_EXPERIMENTS and rc.n_train_episodes > MAX_UI_EPISODES:
            errs.append(f"The app limits live training to {MAX_UI_EPISODES} episodes (reference run used 5000).")
        if errs:
            errors += [f"{SHORT[e]}: {m}" for m in errs]
        configs[e] = rc
    if errors:
        for m in errors:
            st.error(m)
        return
    results = {}
    bar = st.progress(0.0, text="Starting...")
    for i, e in enumerate(selected):
        rc = configs[e]
        label = f"{SHORT[e]} ({i + 1}/{len(selected)})"

        def cb(done, total, row, _label=label, _i=i):
            if done % max(1, total // 50) == 0 or done == total:
                bar.progress((_i + done / total) / len(selected), text=f"{_label}: training episode {done}/{total}")

        bar.progress(i / len(selected), text=f"{label}: " + ("training then evaluating..." if e in LEARNING_EXPERIMENTS
                                                            else "evaluating (no training phase)..."))
        r = run_experiment(rc, progress_callback=cb)
        entry = {"config": rc.to_dict(), "summary": r["summary"], "roles": {str(k): v for k, v in r["roles"].items()}}
        if r["training_df"] is not None:
            entry["training"] = training_curves(r["training_df"], r["agent_ids"])
        results[e] = entry
    bar.progress(1.0, text="Done")
    ss["exp_results"] = results
    ss["ex_view"] = VIEW_LIVE
    ss["exp_results_meta"] = {"seeds": (configs[selected[0]].eval_seeds[0], configs[selected[0]].eval_seeds[-1]),
                              "train_episodes": configs[selected[0]].n_train_episodes}


# ------------------------------------------------------------------ #
def _comparison_table(res: dict) -> pd.DataFrame:
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


def _reading(res: dict) -> list:
    """Data-driven plain-language observations (generated from the numbers shown, not canned)."""
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


def _results_block(res: dict, key: str, meta_line: str):
    if not res:
        st.info("No results yet.")
        return
    st.markdown(f'<div class="small">{meta_line}</div>', unsafe_allow_html=True)
    st.dataframe(_comparison_table(res), width="stretch")
    st.caption("Metrics are for the COMBINED Rs 10,00,000 portfolio (four Rs 2,50,000 sleeves in A, B, D; one Rs 10,00,000 account in C), "
               "averaged over held-out episodes. Sharpe is episode-scale, not annualised. Synthetic data only.")
    colors = {SHORT[e]: theme.EXPERIMENT_COLORS[e] for e in res}
    a, b = st.columns(2)
    box = {SHORT[e]: res[e]["summary"]["combined_returns_per_seed"] for e in res}
    a.plotly_chart(charts.returns_box(box, colors), width="stretch", key=f"{key}_box")
    curves = {SHORT[e]: (res[e]["summary"]["mean_equity_curve"], res[e]["summary"]["std_equity_curve"]) for e in res}
    b.plotly_chart(charts.mean_curve_band(curves, colors), width="stretch", key=f"{key}_curve")

    learned = [e for e in res if "training" in res[e]]
    if learned:
        st.markdown("##### Training phase (in-sample) - kept separate from the evaluation above")
        cols = st.columns(len(learned))
        for col, e in zip(cols, learned):
            tc = res[e]["training"]["total_capital"]
            col.plotly_chart(charts.training_curves_from_reference({SHORT[e]: tc}, colors,
                             f"{SHORT[e]}: training-phase capital (Rs, smoothed)", "Rs", 280),
                             width="stretch", key=f"{key}_tr_{e}")
    hist_rows = {}
    for e in res:
        for k, v in res[e]["summary"]["action_hist"].items():
            hist_rows[f"{SHORT[e]} / {res[e]['roles'][str(k)]}"] = v
    st.plotly_chart(charts.action_hist_heatmap(hist_rows, height=max(260, 30 * len(hist_rows) + 120)),
                    width="stretch", key=f"{key}_actions")

    st.markdown("##### Reading these results")
    for line in _reading(res):
        st.markdown(f"- {line}")
    payload = {e: {"config": res[e]["config"], "summary": res[e]["summary"]} for e in res}
    st.download_button("Download results (JSON)", json.dumps(payload, indent=1).encode(), f"experiment_results_{key}.json",
                       "application/json", key=f"{key}_dl")


def render():
    ss = st.session_state
    st.title("Experiments")
    st.caption("Compare four ways of allocating the same Rs 10,00,000 in the same synthetic market. Learning agents TRAIN on one "
               "seed, then are EVALUATED out-of-sample on a different block of seeds with exploration off and Q-tables frozen. "
               "Training and evaluation numbers are never mixed. Same configuration + same seeds = same results.")

    with st.container(border=True):
        st.markdown("#### Configure")
        st.multiselect("Experiments to run", list("ABCD"), default=list("ABCD"), key="ex_selected",
                       format_func=lambda e: EXPERIMENT_NAMES[e])
        a, b, c, d = st.columns(4)
        a.number_input(f"Training episodes (C, D; max {MAX_UI_EPISODES})", 10, MAX_UI_EPISODES, 1000, 100, key="ex_episodes")
        b.number_input("Training seed", 0, 1_000_000, cfg.TRAIN_SEED, 1, key="ex_train_seed")
        c.number_input("First evaluation seed", 0, 1_000_000, cfg.EVAL_SEED_START, 1, key="ex_eval_start")
        d.number_input("Evaluation episodes", 5, 200, cfg.N_EVAL_EPISODES, 5, key="ex_eval_n")
        e1, e2, e3, e4 = st.columns(4)
        e1.selectbox("Fixed basket for B", list(range(cfg.N_ACTIONS)), index=2, key="ex_fixed_basket",
                     format_func=lambda i: f"{i}: {cfg.ACTION_BASKET_NAMES[i]}")
        e2.slider("alpha (C, D)", 0.01, 1.0, cfg.ALPHA_LR, 0.01, key="ex_alpha")
        e3.slider("gamma (C, D)", 0.0, 0.999, cfg.GAMMA, 0.005, key="ex_gamma")
        e4.slider("epsilon decay (C, D)", 0.990, 1.0, cfg.EPSILON_DECAY, 0.0005, format="%.4f", key="ex_eps_decay")
        f1, f2 = st.columns(2)
        f1.number_input("Episode length (steps)", 20, 300, cfg.EPISODE_LENGTH, 10, key="ex_ep_len")
        f2.checkbox("Market impact between agents", True, key="ex_impact")
        st.button("Run selected experiments", type="primary", on_click=_run_selected,
                  help="Runs the real training + evaluation pipeline. About 15-30 s per 1000 training episodes for D.")

    ref = reference.load_reference_results()
    default_view = VIEW_LIVE if ss.get("exp_results") else VIEW_REF
    if "ex_view" not in ss:
        ss["ex_view"] = default_view
    st.radio("Results to show", [VIEW_REF, VIEW_LIVE], horizontal=True, key="ex_view")
    if ss["ex_view"] == VIEW_LIVE:
        res = ss.get("exp_results") or {}
        if res:
            m = ss.get("exp_results_meta", {})
            _results_block(res, "live", f"Live run | training {m.get('train_episodes', '?')} episodes | evaluation seeds "
                                         f"{m.get('seeds', ('?', '?'))[0]}-{m.get('seeds', ('?', '?'))[1]}")
        else:
            st.info("Configure above and press **Run selected experiments**; your results will appear here.")
    else:
        if ref:
            m = ref["meta"]
            exps = {e: v for e, v in ref["experiments"].items()}
            st.markdown(f'<div class="note"><b>Precomputed, not live.</b> Produced by running <code>{m["script"]}</code> '
                        f'({m["train_episodes"]:,} training episodes on seed {m["train_seed"]}, evaluation on seeds '
                        f'{m["eval_seeds"][0]}-{m["eval_seeds"][1]}, generated {m["generated_at_utc"]}). It is a cached output of the '
                        'same code this page runs; rerun that script to regenerate it. Synthetic data.</div>', unsafe_allow_html=True)
            st.write("")
            _results_block(exps, "ref", f"Reference run | {m['train_episodes']:,} training episodes | seeds "
                                         f"{m['eval_seeds'][0]}-{m['eval_seeds'][1]}")
        else:
            st.warning("Reference files (data/reference_results.json) were not found. Run `python -m scripts.build_reference` to create them.")
