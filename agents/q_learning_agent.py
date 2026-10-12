"""
agents/q_learning_agent.py

Independent tabular Q-Learning agent. Each instance owns its own
Q-table Q_i(s, a) -- this is what makes 4-agent training "Independent
Q-Learning": every agent learns on its own experience with no shared
parameters (the coupling between agents happens only through the shared
environment, see financial_env.py).
"""

import numpy as np

import config as cfg
from agents.base_agent import BaseAgent
from rl.q_learning import q_update, epsilon_greedy_action


class QLearningAgent(BaseAgent):
    def __init__(
        self,
        agent_id: int,
        role: str,
        n_states: int = cfg.N_STATES,
        n_actions: int = cfg.N_ACTIONS,
        alpha: float = cfg.ALPHA_LR,
        gamma: float = cfg.GAMMA,
        epsilon_initial: float = cfg.EPSILON_INITIAL,
        epsilon_min: float = cfg.EPSILON_MIN,
        epsilon_decay: float = cfg.EPSILON_DECAY,
        q_init: float = cfg.Q_INIT_VALUE,
        seed: int = None,
    ):
        super().__init__(agent_id, role, n_states, n_actions)
        self.q_table = np.full((n_states, n_actions), q_init, dtype=float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_initial
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.last_td_error = 0.0

    def select_action(self, state: int) -> int:
        return epsilon_greedy_action(self.q_table, state, self.epsilon, self.n_actions, self.rng)

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        self.last_td_error = q_update(
            self.q_table, state, action, reward, next_state, done, self.alpha, self.gamma
        )

    def on_episode_end(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
