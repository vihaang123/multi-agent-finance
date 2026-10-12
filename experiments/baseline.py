"""
experiments/baseline.py

Non-learning baseline agents. Implement the exact same BaseAgent
interface as QLearningAgent so the Trainer can run any of them through
an identical loop -- this is what makes Experiment A/B vs C/D a fair
comparison.
"""

import numpy as np

import config as cfg
from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Picks a uniformly random basket every step. Never learns."""

    def __init__(self, agent_id: int, role: str, n_states: int = cfg.N_STATES, n_actions: int = cfg.N_ACTIONS, seed: int = None):
        super().__init__(agent_id, role, n_states, n_actions)
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: int) -> int:
        return int(self.rng.integers(0, self.n_actions))

    def update(self, state, action, reward, next_state, done):
        pass  # no learning


class FixedAgent(BaseAgent):
    """Always picks the same predetermined basket. Never learns."""

    def __init__(self, agent_id: int, role: str, fixed_action: int, n_states: int = cfg.N_STATES, n_actions: int = cfg.N_ACTIONS):
        super().__init__(agent_id, role, n_states, n_actions)
        self.fixed_action = fixed_action

    def select_action(self, state: int) -> int:
        return self.fixed_action

    def update(self, state, action, reward, next_state, done):
        pass  # no learning
