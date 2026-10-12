import numpy as np

import config as cfg
from agents.q_learning_agent import QLearningAgent
from experiments.baseline import RandomAgent, FixedAgent
from rl.q_learning import q_update, epsilon_greedy_action


def test_q_table_initialised_to_optimistic_value():
    agent = QLearningAgent(0, "Growth", n_states=10, n_actions=4, seed=0)
    assert agent.q_table.shape == (10, 4)
    assert np.all(agent.q_table == cfg.Q_INIT_VALUE)


def test_q_table_can_be_initialised_to_zero_explicitly():
    agent = QLearningAgent(0, "Growth", n_states=10, n_actions=4, q_init=0.0, seed=0)
    assert np.all(agent.q_table == 0.0)


def test_q_update_moves_value_towards_target():
    q = np.zeros((5, 3))
    td_error = q_update(q, state=0, action=1, reward=1.0, next_state=1, done=False, alpha=0.5, gamma=0.9)
    assert q[0, 1] == pytest_approx(0.5)
    assert td_error == pytest_approx(1.0)


def pytest_approx(x, tol=1e-9):
    class _Approx:
        def __eq__(self, other):
            return abs(other - x) < tol
    return _Approx()


def test_q_update_terminal_state_ignores_future_value():
    q = np.zeros((5, 3))
    q[1, :] = 100.0  # if bootstrapped, this would dominate
    q_update(q, state=0, action=0, reward=2.0, next_state=1, done=True, alpha=1.0, gamma=0.99)
    assert q[0, 0] == 2.0  # no bootstrap since done=True


def test_epsilon_greedy_fully_random_when_epsilon_one():
    rng = np.random.default_rng(0)
    q = np.zeros((1, 5))
    q[0, 2] = 10.0  # best action is 2, but epsilon=1 should ignore it sometimes
    actions_seen = set()
    for _ in range(200):
        a = epsilon_greedy_action(q, state=0, epsilon=1.0, n_actions=5, rng=rng)
        actions_seen.add(a)
    assert len(actions_seen) > 1  # should explore multiple actions


def test_epsilon_greedy_always_best_when_epsilon_zero():
    rng = np.random.default_rng(0)
    q = np.zeros((1, 5))
    q[0, 3] = 10.0
    for _ in range(50):
        a = epsilon_greedy_action(q, state=0, epsilon=0.0, n_actions=5, rng=rng)
        assert a == 3


def test_agent_select_action_within_bounds():
    agent = QLearningAgent(0, "Growth", n_states=cfg.N_STATES, n_actions=cfg.N_ACTIONS, seed=1)
    for s in range(0, cfg.N_STATES, 10):
        a = agent.select_action(s)
        assert 0 <= a < cfg.N_ACTIONS


def test_epsilon_decays_and_floors_at_min():
    agent = QLearningAgent(0, "Growth", epsilon_initial=1.0, epsilon_min=0.05, epsilon_decay=0.9, seed=0)
    for _ in range(500):
        agent.on_episode_end()
    assert agent.epsilon == 0.05


def test_random_agent_picks_valid_actions():
    agent = RandomAgent(0, "Growth", n_actions=cfg.N_ACTIONS, seed=0)
    for _ in range(20):
        a = agent.select_action(state=0)
        assert 0 <= a < cfg.N_ACTIONS


def test_fixed_agent_always_returns_same_action():
    agent = FixedAgent(0, "Growth", fixed_action=3, n_actions=cfg.N_ACTIONS)
    for s in [0, 5, 50, 107]:
        assert agent.select_action(s) == 3
