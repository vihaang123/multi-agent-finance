"""
Multi-Agent Reinforcement Learning for Financial Resource Allocation - Streamlit dashboard.

Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud, main file = streamlit_app.py (see DEPLOYMENT.md)

ALL DATA IN THIS APP IS SYNTHETIC (a simulated Markov-regime market). Nothing here is real market
data, investment advice, or a real-money / broker integration.
"""

import streamlit as st

st.set_page_config(page_title="Multi-Agent RL - Financial Allocation", page_icon="\U0001F4C8", layout="wide",
                   initial_sidebar_state="auto")

from dashboard import theme                      # noqa: E402
from dashboard.pages import analytics, dashboard, experiments, learn, rl_lab, simulation  # noqa: E402
from dashboard.state import PAGES, init_state, persist_widget_state  # noqa: E402

PAGE_FUNCS = {
    "Dashboard": dashboard.render,
    "Simulation": simulation.render,
    "RL Lab": rl_lab.render,
    "Portfolio Analytics": analytics.render,
    "Experiments": experiments.render,
    "Learn": learn.render,
}
PAGE_HELP = {
    "Dashboard": "Overview of the current episode",
    "Simulation": "Start / pause / step the live market",
    "RL Lab": "Train agents, inspect Q-tables",
    "Portfolio Analytics": "Returns, risk, costs, comparison",
    "Experiments": "Random vs Fixed vs Q-learning",
    "Learn": "MDP, Q-learning, reward, regimes",
}


def main():
    theme.inject_css()
    try:
        init_state()
    except Exception as exc:                      # noqa: BLE001 - show a clear message instead of a blank page
        st.error("The app could not initialise its state.")
        st.exception(exc)
        return
    persist_widget_state()

    with st.sidebar:
        st.markdown("### \U0001F4C8 Multi-Agent RL")
        st.caption("Financial resource allocation - university mini project")
        st.radio("Navigate", PAGES, key="nav", format_func=lambda p: p, captions=[PAGE_HELP[p] for p in PAGES])
        st.divider()
        st.markdown("**Active agents**")
        st.caption(st.session_state["agents_label"])
        st.divider()
        st.caption("Synthetic simulated market. Not real data, not advice, no real-money trading.")

    theme.synthetic_banner()
    page = st.session_state.get("nav", PAGES[0])
    try:
        PAGE_FUNCS[page]()
    except Exception as exc:                      # noqa: BLE001
        st.error(f"Something went wrong while rendering the '{page}' page. Details below.")
        st.exception(exc)


main()
