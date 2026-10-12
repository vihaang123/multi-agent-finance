"""
rl/q_learning.py

The core tabular Q-Learning update rule, implemented from scratch
(no external RL framework). Kept as a standalone, generic function so
any agent (any number of states/actions) can reuse it.

    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
"""

import numpy as np


def q_update(
    q_table: np.ndarray,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    done: bool,
    alpha: float,
    gamma: float,
) -> float:
    """
    In-place update of q_table at (state, action). Returns the TD error
    (useful for convergence diagnostics / logging).
    """
    best_next_q = 0.0 if done else np.max(q_table[next_state])
    td_target = reward + gamma * best_next_q
    td_error = td_target - q_table[state, action]
    q_table[state, action] += alpha * td_error
    return float(td_error)


def epsilon_greedy_action(q_table: np.ndarray, state: int, epsilon: float, n_actions: int, rng: np.random.Generator) -> int:
    """
    With probability epsilon: uniformly random action.
    Otherwise: argmax_a Q(state, a), ties broken uniformly at random.
    """
    if rng.random() < epsilon:
        return int(rng.integers(0, n_actions))
    q_row = q_table[state]
    max_q = np.max(q_row)
    best_actions = np.flatnonzero(q_row == max_q)
    return int(rng.choice(best_actions))
