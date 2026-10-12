"""Tests for dashboard.session.SimulationSession (start/pause/step/reset semantics, policies)."""

import copy

import numpy as np
import pytest

import config as cfg
from dashboard.session import SimulationSession
from dashboard import views
from experiments.engine import QHyperParams, build_q_agents
from experiments.run_experiments import AGENT_IDS, AGENT_ROLES
from training.trainer import Trainer
from experiments.engine import make_env


@pytest.fixture(scope="module")
def trained_agents():
    agents = build_q_agents(AGENT_IDS, AGENT_ROLES, QHyperParams(), 42)
    env = make_env(AGENT_IDS, AGENT_ROLES, {a: cfg.INITIAL_CAPITAL_PER_AGENT for a in AGENT_IDS}, 42, 100)
    Trainer(env, agents).train(40, verbose=False)
    return agents


def test_defaults_without_agents_use_random_policy():
    s = SimulationSession(agents=None, seed=1)
    assert all(v["kind"] == "random" for v in s.policy_spec.values())
    with pytest.raises(ValueError):
        s.set_policy(0, "trained")


def test_step_and_run_to_end(trained_agents):
    s = SimulationSession(trained_agents, seed=51, episode_length=10)
    assert s.t == 0 and s.history == [] and not s.done
    row = s.step()
    assert s.t == 1 and row["t"] == 1 and len(s.history) == 1
    assert s.run_steps(3) == 3 and s.t == 4
    assert s.run_to_end() == 6 and s.done
    assert s.step() is None and s.t == 10            # no steps after the end
    assert len(s.capital_series()) == 11


def test_reset_restores_initial_state_and_same_path(trained_agents):
    s = SimulationSession(trained_agents, seed=60, episode_length=15)
    s.run_to_end()
    first = s.capital_series().copy()
    regimes = s.regime_path()
    s.reset()
    assert s.t == 0 and s.history == [] and not s.done and not s.running
    assert np.allclose(s.capital_series(), [sum(s.capitals.values())])
    s.run_to_end()
    assert np.allclose(s.capital_series(), first) and s.regime_path() == regimes     # deterministic replay


def test_reset_with_new_seed_changes_market(trained_agents):
    s = SimulationSession(trained_agents, seed=1, episode_length=30)
    s.run_to_end()
    a = s.regime_path()
    s.reset(seed=2)
    s.run_to_end()
    assert s.seed == 2 and s.regime_path() != a


def test_running_flag_semantics(trained_agents):
    s = SimulationSession(trained_agents, seed=5, episode_length=3)
    s.running = True
    s.run_to_end()
    assert s.done and s.running is False             # finishing stops auto-play
    s.reset()
    assert s.running is False


def test_watching_never_trains_the_agents(trained_agents):
    before_q = {a: trained_agents[a].q_table.copy() for a in AGENT_IDS}
    eps = {a: trained_agents[a].epsilon for a in AGENT_IDS}
    rng_state = {a: copy.deepcopy(trained_agents[a].rng.bit_generator.state) for a in AGENT_IDS}
    visits = {a: trained_agents[a].visit_counts.copy() for a in AGENT_IDS}
    s = SimulationSession(trained_agents, seed=77, episode_length=40)
    s.run_to_end()
    for a in AGENT_IDS:
        assert np.array_equal(trained_agents[a].q_table, before_q[a])
        assert trained_agents[a].epsilon == eps[a]
        assert trained_agents[a].rng.bit_generator.state == rng_state[a]
        assert np.array_equal(trained_agents[a].visit_counts, visits[a])


def test_fixed_policy_and_manual_switching(trained_agents):
    s = SimulationSession(trained_agents, seed=3, episode_length=10)
    s.set_policy(0, "fixed", 4)
    s.step()
    assert s.last["actions"][0] == 4 and s.last["policy_kind"][0] == "fixed" and s.last["q_values"][0] is None
    s.set_policy(0, "fixed", 1)                      # manual change between steps
    s.step()
    assert s.last["actions"][0] == 1
    s.set_policy(0, "trained")
    s.step()
    assert s.last["q_values"][0] is not None and s.last["policy_kind"][0] == "trained"
    assert s.last["actions"][0] == int(np.argmax(s.last["q_values"][0]))      # greedy w.r.t. the shown Q-values


def test_invalid_policy_inputs(trained_agents):
    s = SimulationSession(trained_agents, seed=3)
    with pytest.raises(ValueError):
        s.set_policy(0, "magic")
    with pytest.raises(ValueError):
        s.set_policy(0, "fixed", 99)
    with pytest.raises(ValueError):
        SimulationSession(trained_agents, episode_length=0)


def test_market_path_is_exogenous_to_policy(trained_agents):
    """Same seed -> same regime sequence and same base return draws, whatever the agents do."""
    a = SimulationSession(trained_agents, seed=88, episode_length=40)
    b = SimulationSession(trained_agents, seed=88, episode_length=40)
    for aid in AGENT_IDS:
        b.set_policy(aid, "fixed", 0)
    a.run_to_end()
    b.run_to_end()
    assert a.regime_path() == b.regime_path()
    for ra, rb in zip(a.history, b.history):
        assert np.allclose(ra["base_returns"], rb["base_returns"])
    assert not np.allclose(a.capital_series(), b.capital_series())          # but outcomes differ


def test_history_schema_works_with_views(trained_agents):
    s = SimulationSession(trained_agents, seed=9, episode_length=12)
    s.run_to_end()
    mf = views.metrics_frame(s.history, s.agent_ids, s.roles, s.capitals)
    assert "Total" in mf.index and len(mf) == 5
    assert mf.loc["Total", "Final value"] == pytest.approx(s.capital_series()[-1])
    assert views.weights_array(s.history, 0).shape == (12, cfg.N_ASSETS)
    assert views.asset_pnl_total(s.history, s.agent_ids).shape == (cfg.N_ASSETS,)
    rup = views.current_rupees(s)
    assert sum(v.sum() for v in rup.values()) == pytest.approx(s.capital_series()[-1])


def test_current_rupees_at_start_is_all_cash(trained_agents):
    s = SimulationSession(trained_agents, seed=9)
    for v in views.current_rupees(s).values():
        assert v[cfg.CASH] == cfg.INITIAL_CAPITAL_PER_AGENT and v[:3].sum() == 0


def test_market_impact_toggle(trained_agents):
    on = SimulationSession(trained_agents, seed=4, episode_length=5, enable_market_impact=True)
    off = SimulationSession(trained_agents, seed=4, episode_length=5, enable_market_impact=False)
    for s in (on, off):
        for aid in AGENT_IDS:
            s.set_policy(aid, "fixed", 0)
        s.run_to_end()
    assert all(r["market_impact"] < 0 for r in on.history)
    assert all(r["market_impact"] == 0 for r in off.history)
