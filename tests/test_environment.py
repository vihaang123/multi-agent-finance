import numpy as np
import pytest

import config as cfg
from environment.financial_env import FinancialEnv
from environment.market import MarketSimulator
from environment.portfolio import Portfolio


def make_env(seed=0, episode_length=10):
    agent_ids = [0, 1, 2, 3]
    roles = {0: "Growth", 1: "Conservative", 2: "Balanced", 3: "RiskLiquidity"}
    caps = {aid: cfg.INITIAL_CAPITAL_PER_AGENT for aid in agent_ids}
    return FinancialEnv(agent_ids, roles, caps, episode_length=episode_length, seed=seed)


def test_reset_returns_valid_states_for_all_agents():
    env = make_env()
    states = env.reset()
    assert set(states.keys()) == {0, 1, 2, 3}
    for s in states.values():
        assert 0 <= s < cfg.N_STATES


def test_state_space_size_matches_config():
    assert cfg.N_STATES == cfg.N_REGIMES * (cfg.N_ALLOC_BUCKETS ** 3)
    assert cfg.N_STATES == 108


def test_action_baskets_sum_to_one_and_nonnegative():
    assert np.allclose(cfg.ACTION_BASKETS.sum(axis=1), 1.0)
    assert np.all(cfg.ACTION_BASKETS >= 0.0)


def test_invalid_action_raises():
    env = make_env()
    env.reset()
    with pytest.raises(ValueError):
        env.step({0: cfg.N_ACTIONS, 1: 0, 2: 0, 3: 0})  # out of range


def test_step_returns_expected_shapes():
    env = make_env()
    env.reset()
    actions = {aid: 2 for aid in [0, 1, 2, 3]}
    next_states, rewards, done, info = env.step(actions)
    assert set(next_states.keys()) == {0, 1, 2, 3}
    assert set(rewards.keys()) == {0, 1, 2, 3}
    assert isinstance(done, bool)
    for aid in [0, 1, 2, 3]:
        assert np.isfinite(rewards[aid])
        assert abs(rewards[aid]) <= cfg.REWARD_CLIP + 1e-9


def test_episode_terminates_after_episode_length():
    env = make_env(episode_length=5)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step({aid: 0 for aid in [0, 1, 2, 3]})
        steps += 1
    assert steps == 5


def test_allocation_weights_always_valid_after_rebalance_and_returns():
    env = make_env()
    env.reset()
    rng = np.random.default_rng(1)
    for _ in range(20):
        actions = {aid: int(rng.integers(0, cfg.N_ACTIONS)) for aid in [0, 1, 2, 3]}
        _, _, done, info = env.step(actions)
        for aid in [0, 1, 2, 3]:
            w = info["actual_weights"][aid]
            assert np.isclose(w.sum(), 1.0, atol=1e-6)
            assert np.all(w >= -1e-9)
        if done:
            break


def test_capital_never_created_from_nothing_within_a_step():
    """Total capital across agents can only move due to market returns/impact,
    not due to rebalancing itself (transaction costs only remove value)."""
    env = make_env()
    env.reset()
    total_before = sum(env.portfolios[aid].capital for aid in env.agent_ids)
    env.step({0: 1, 1: 1, 2: 1, 3: 1})
    total_after = sum(env.portfolios[aid].capital for aid in env.agent_ids)
    # After one step (rebalance cost + market return), total capital should
    # be finite and not wildly explode (sanity bound, not an exact equality
    # since market returns do change value).
    assert np.isfinite(total_after)
    assert total_after < total_before * 2  # no explosive bug


def test_market_regime_transition_matrix_is_stochastic():
    assert np.allclose(cfg.REGIME_TRANSITION.sum(axis=1), 1.0)


def test_market_simulator_reset_and_sample():
    rng = np.random.default_rng(0)
    market = MarketSimulator(rng)
    r = market.reset()
    assert r == cfg.NEUTRAL
    returns = market.sample_returns(cfg.BULL)
    assert returns.shape == (cfg.N_ASSETS,)
    assert np.all(np.isfinite(returns))


def test_info_current_regime_matches_regime_that_generated_the_return():
    """Regression test for the regime-labeling bug: info['current_regime']
    must equal the regime the environment was in BEFORE step() was called
    (i.e. the regime that actually generated this step's asset returns and
    reward), not the regime it transitions into afterward."""
    env = make_env(seed=7, episode_length=30)
    env.reset()
    rng = np.random.default_rng(3)
    for _ in range(30):
        regime_before_step = env.market.regime
        actions = {aid: int(rng.integers(0, cfg.N_ACTIONS)) for aid in [0, 1, 2, 3]}
        _, _, done, info = env.step(actions)
        assert info["current_regime"] == regime_before_step
        if done:
            break


def test_info_next_regime_matches_regime_after_transition():
    """Companion regression test: info['next_regime'] must equal the regime
    the environment is in immediately AFTER step() returns (i.e. the one
    that will generate the FOLLOWING step's return, and that next_states
    is already encoded against)."""
    env = make_env(seed=7, episode_length=30)
    env.reset()
    rng = np.random.default_rng(3)
    for _ in range(30):
        actions = {aid: int(rng.integers(0, cfg.N_ACTIONS)) for aid in [0, 1, 2, 3]}
        _, _, done, info = env.step(actions)
        regime_after_step = env.market.regime
        assert info["next_regime"] == regime_after_step
        if done:
            break


def test_current_and_next_regime_can_legitimately_differ():
    """Sanity check that the test above isn't vacuously true because regime
    never changes: over enough steps, current_regime != next_regime should
    actually happen at least once (a real regime transition occurred)."""
    env = make_env(seed=7, episode_length=200)
    env.reset()
    rng = np.random.default_rng(3)
    saw_a_transition = False
    for _ in range(200):
        actions = {aid: int(rng.integers(0, cfg.N_ACTIONS)) for aid in [0, 1, 2, 3]}
        _, _, done, info = env.step(actions)
        if info["current_regime"] != info["next_regime"]:
            saw_a_transition = True
        if done:
            break
    assert saw_a_transition


def test_portfolio_rebalance_charges_cost_and_matches_target_weights():
    port = Portfolio(initial_capital=100_000.0)
    target = np.array([0.5, 0.3, 0.1, 0.1])
    cost = port.rebalance(target)
    assert cost >= 0
    assert np.allclose(port.actual_weights, target, atol=1e-6)


def test_portfolio_drawdown_tracks_peak():
    port = Portfolio(initial_capital=100_000.0)
    port.rebalance(np.array([1.0, 0.0, 0.0, 0.0]))
    port.apply_returns(np.array([0.10, 0.0, 0.0, 0.0]))  # capital grows
    assert port.drawdown == 0.0
    port.apply_returns(np.array([-0.20, 0.0, 0.0, 0.0]))  # capital falls from new peak
    assert port.drawdown > 0.0


def test_drawdown_increase_only_fires_on_new_lows_not_lingering_history():
    """The reward-facing signal (drawdown_increase) should only be positive
    on the step where drawdown WORSENS, not on every subsequent step just
    because the portfolio is still below its historical peak. This is the
    fix for the 'hide in cash forever' reward-hacking collapse."""
    port = Portfolio(initial_capital=100_000.0)
    port.rebalance(np.array([1.0, 0.0, 0.0, 0.0]))
    port.apply_returns(np.array([-0.10, 0.0, 0.0, 0.0]))  # first drop: drawdown_increase > 0
    assert port.drawdown_increase > 0.0

    # No rebalance this step (no transaction cost) -- pure flat market return.
    port.apply_returns(np.array([0.0, 0.0, 0.0, 0.0]))
    # Drawdown (cumulative, vs peak) is still > 0 because capital hasn't recovered...
    assert port.drawdown > 0.0
    # ...but drawdown_increase must be exactly 0 since nothing got WORSE this step.
    assert port.drawdown_increase == 0.0
