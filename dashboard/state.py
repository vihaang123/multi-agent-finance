"""dashboard/state.py - Streamlit session-state bootstrap and helpers."""

import streamlit as st

import config as cfg
from dashboard import reference
from dashboard.session import SimulationSession

PAGES = ["Dashboard", "Simulation", "RL Lab", "Portfolio Analytics", "Experiments", "Learn"]

# Widget keys that should survive page switches (Streamlit drops the state of
# widgets that are not rendered in a run; re-assigning keeps them alive).
PERSIST_KEYS = [
    "sim_seed", "sim_speed", "sim_steps_n", "sim_view_agent", "dash_seed",
    "lab_n_episodes", "lab_alpha", "lab_gamma", "lab_eps0", "lab_epsmin", "lab_decay", "lab_qinit",
    "lab_seed", "lab_impact", "lab_mode", "lab_smooth", "lab_inspect_agent", "lab_regime", "lab_eq", "lab_bond",
    "lab_gold", "lab_heat_mode", "lab_heat_agent", "lab_eval_n", "lab_trace_agent", "lab_trace_n",
    "an_source", "an_eval_n", "an_eval_start", "an_agents", "an_seed_pick", "an_vol_window",
    "ex_selected", "ex_episodes", "ex_train_seed", "ex_eval_start", "ex_eval_n", "ex_fixed_basket", "ex_alpha",
    "ex_gamma", "ex_eps_decay", "ex_impact", "ex_ep_len", "ex_view",
    "sim_policy_kind_0", "sim_policy_kind_1", "sim_policy_kind_2", "sim_policy_kind_3",
    "sim_basket_0", "sim_basket_1", "sim_basket_2", "sim_basket_3", "sim_impact",
]


def persist_widget_state():
    for k in PERSIST_KEYS:
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]


def goto(page: str):
    st.session_state["nav"] = page


def init_state():
    ss = st.session_state
    if "agents" not in ss:
        ss["agents"] = reference.make_reference_agents()
        ss["agents_label"] = reference.reference_label()
        ss["agents_source"] = "reference"
        ss["lab_log"] = None
        ss["lab_trainer"] = None
        ss["lab_total_episodes"] = 0
    if "sim" not in ss:
        sim = SimulationSession(agents=ss["agents"], seed=cfg.EVAL_SEED_START)
        sim.run_to_end()                 # start with a completed example episode so the Dashboard is populated
        ss["sim"] = sim
    ss.setdefault("exp_results", {})
    ss.setdefault("an_eval", None)


def reset_to_reference_agents():
    ss = st.session_state
    ss["agents"] = reference.make_reference_agents()
    ss["agents_label"] = reference.reference_label()
    ss["agents_source"] = "reference"
    ss["lab_log"] = None
    ss["lab_trainer"] = None
    ss["lab_total_episodes"] = 0
    ss["an_eval"] = None
    ss["sim"].agents = ss["agents"]
