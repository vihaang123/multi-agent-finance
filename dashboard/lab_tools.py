"""
dashboard/lab_tools.py - helpers for the RL Lab views (policy summary and the Q-update walkthrough).
Training itself lives in server/trainstate.py.
"""

import copy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config as cfg
from environment.state_utils import describe_state
from experiments.engine import make_env
from experiments.run_experiments import AGENT_IDS, AGENT_ROLES


def greedy_policy_by_regime(agent) -> pd.DataFrame:
    """Visit-weighted share (%) of each basket as the GREEDY choice, per regime.
    Only states the agent actually visited count; shows what it learned to do."""
    visits = agent.visit_counts.sum(axis=1)
    greedy = np.argmax(agent.q_table, axis=1)
    rows = {}
    for r, rname in enumerate(cfg.REGIME_NAMES):
        sl = slice(r * 27, (r + 1) * 27)
        w = np.zeros(cfg.N_ACTIONS)
        for s_off, (v, g) in enumerate(zip(visits[sl], greedy[sl])):
            w[g] += v
        rows[rname] = (w / w.sum() * 100) if w.sum() > 0 else w
    df = pd.DataFrame(rows, index=cfg.ACTION_BASKET_NAMES).T
    df.index.name = "Regime"
    return df


def trace_updates(agents, agent_id: int, seed: int, n_steps: int, episode_length: int,
                  enable_market_impact: bool = True) -> List[dict]:
    """Worked example: run the first n_steps of an episode on a DEEP COPY of the
    agents (the real ones are untouched) and record each Q-update's arithmetic
    for one agent. The recorded value is checked against the real q_update()."""
    n_steps = max(1, min(int(n_steps), episode_length))
    ag_copy = copy.deepcopy(agents)
    caps = {a: cfg.INITIAL_CAPITAL_PER_AGENT for a in AGENT_IDS}
    env = make_env(AGENT_IDS, AGENT_ROLES, caps, seed, episode_length, cfg.REWARD_MODE, enable_market_impact)
    states = env.reset()
    rows = []
    for t in range(n_steps):
        actions = {a: ag_copy[a].select_action(states[a]) for a in AGENT_IDS}
        next_states, rewards, done, info = env.step(actions)
        ag = ag_copy[agent_id]
        s, a_, r, s2 = states[agent_id], actions[agent_id], rewards[agent_id], next_states[agent_id]
        q_old = float(ag.q_table[s, a_])
        best_next = 0.0 if done else float(np.max(ag.q_table[s2]))
        target = r + ag.gamma * best_next
        td = target - q_old
        expected_new = q_old + ag.alpha * td
        for a in AGENT_IDS:
            ag_copy[a].update(states[a], actions[a], rewards[a], next_states[a], done)
        q_new = float(ag.q_table[s, a_])
        rows.append({
            "step": t + 1, "state": s, "state_desc": describe_state(s), "action": a_,
            "basket": cfg.ACTION_BASKET_NAMES[a_], "reward": r, "next_state": s2, "Q_old": q_old,
            "max_Q_next": best_next, "target": target, "TD_error": td, "Q_new": q_new,
            "alpha": ag.alpha, "gamma": ag.gamma, "matches_formula": bool(abs(q_new - expected_new) < 1e-9),
        })
        states = next_states
        if done:
            break
    return rows
