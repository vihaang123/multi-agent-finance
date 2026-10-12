"""
dashboard/reference.py - loads the PRECOMPUTED reference run (data/*.json|npz).

These files are produced by `python -m scripts.build_reference`, which really
runs experiments A-D (5000 training episodes, 50 held-out evaluation episodes)
through experiments.engine.run_experiment. They are cached results of the same
code the app runs live - not hand-typed numbers. If the files are missing the
app still works; it just starts with untrained agents and no reference table.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import config as cfg
from agents.q_learning_agent import QLearningAgent
from experiments.run_experiments import AGENT_IDS, AGENT_ROLES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_PATH = DATA_DIR / "reference_results.json"
QTABLES_PATH = DATA_DIR / "reference_qtables.npz"


@lru_cache(maxsize=1)
def load_reference_results() -> Optional[dict]:
    try:
        with open(RESULTS_PATH, "r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=1)
def _load_qtables() -> Optional[dict]:
    try:
        with np.load(QTABLES_PATH) as z:
            return {k: z[k] for k in z.files}
    except (OSError, ValueError):
        return None


def reference_label() -> str:
    ref = load_reference_results()
    if not ref:
        return "Untrained agents (reference files not found)"
    m = ref["meta"]
    return (f"Precomputed reference agents: 4-agent Independent Q-learning, {m['train_episodes']:,} training "
            f"episodes, seed {m['train_seed']}")


def make_reference_agents() -> Dict[int, QLearningAgent]:
    """Fresh, independent agent objects holding the reference Q-tables (a new copy
    every call, so training in the RL Lab never corrupts the cached arrays)."""
    q = _load_qtables()
    agents = {aid: QLearningAgent(aid, AGENT_ROLES[aid], seed=cfg.TRAIN_SEED + aid) for aid in AGENT_IDS}
    if q is None:
        return agents
    for aid in AGENT_IDS:
        try:
            agents[aid].q_table = q[f"D_q_{aid}"].copy()
            agents[aid].visit_counts = q[f"D_visits_{aid}"].copy()
            agents[aid].epsilon = float(q[f"D_epsilon_{aid}"])
            agents[aid].n_updates = int(agents[aid].visit_counts.sum())
        except KeyError:
            return {a: QLearningAgent(a, AGENT_ROLES[a], seed=cfg.TRAIN_SEED + a) for a in AGENT_IDS}
    return agents
