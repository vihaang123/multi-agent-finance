"""
agents/base_agent.py

Abstract interface every agent (Q-learning or baseline) must implement,
so the Trainer and FinancialEnv never need to know which kind they hold.
"""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, agent_id: int, role: str, n_states: int, n_actions: int):
        self.agent_id = agent_id
        self.role = role
        self.n_states = n_states
        self.n_actions = n_actions

    @abstractmethod
    def select_action(self, state: int) -> int:
        ...

    @abstractmethod
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        ...

    def on_episode_end(self):
        """Optional hook, e.g. for epsilon decay. No-op by default."""
        pass
