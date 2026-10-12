"""Tests for the additive dashboard instrumentation, analytics and experiment engine."""

import copy

import numpy as np
import pytest

import config as cfg
from agents.q_learning_agent import QLearningAgent
from environment.financial_env import FinancialEnv
from environment.state_utils import decode_state, describe_state, encode_state
from evaluation import analytics
from experiments.engine import (QHyperParams, RunConfig, build_q_agents, evaluate_agents, run_experiment,
                                summarize_evaluation, training_curves)
from experiments.run_experiments import AGENT_IDS, AGENT_ROLES, evaluate_out_of_sample
from training.trainer import Trainer

CAPS = {a: cfg.INITIAL_CAPITAL_PER_AGENT for a in AGENT_IDS}


def _env(seed=5, **kw):
    return FinancialEnv(AGENT_IDS, AGENT_ROLES, CAPS, seed=seed, **kw)


# --------------------------- env instrumentation --------------------------- #
class TestEnvInstrumentation:
    def test_reward_components_sum_to_reward(self):
        env = _env()
        env.reset()
        rng = np.random.default_rng(0)
        for _ in range(30):
            acts = {a: int(rng.integers(0, cfg.N_ACTIONS)) for a in AGENT_IDS}
            _, rewards, _, info = env.step(acts)
            for a in AGENT_IDS:
                c = info["reward_components"][a]
                total = c["return"] + c["risk"] + c["transaction"] + c["drawdown"] + c["liquidity"]
                assert total == pytest.approx(c["unclipped"], abs=1e-12)
                assert c["clipped"] == pytest.approx(np.clip(total, -cfg.REWARD_CLIP, cfg.REWARD_CLIP))
                assert rewards[a] == pytest.approx(c["clipped"])

    def test_rupee_bookkeeping_is_consistent(self):
        env = _env(seed=9)
        env.reset()
        prev_cap = dict(CAPS)
        for t in range(25):
            acts = {a: (t + a) % cfg.N_ACTIONS for a in AGENT_IDS}
            _, _, _, info = env.step(acts)
            for a in AGENT_IDS:
                pre, post, pnl = info["asset_values_pre"][a], info["asset_values"][a], info["asset_pnl"][a]
                assert pre.sum() == pytest.approx(prev_cap[a] - info["txcost"][a])        # cost leaves the sleeve
                assert post.sum() == pytest.approx(info["capital"][a])
                assert pnl.sum() == pytest.approx(post.sum() - pre.sum())
                assert np.allclose(post, pre * (1 + info["asset_returns"]))
            prev_cap = dict(info["capital"])

    def test_market_impact_only_touches_equity(self):
        env = _env(seed=3)
        env.reset()
        _, _, _, info = env.step({a: 0 for a in AGENT_IDS})           # everyone 70% equity -> crowding
        diff = info["asset_returns"] - info["base_returns"]
        assert info["market_impact"] < 0
        assert diff[cfg.EQUITY] == pytest.approx(info["market_impact"])
        assert np.allclose(diff[1:], 0)

    def test_impact_off_gives_equal_returns(self):
        env = _env(seed=3, enable_market_impact=False)
        env.reset()
        _, _, _, info = env.step({a: 0 for a in AGENT_IDS})
        assert np.allclose(info["asset_returns"], info["base_returns"])


# --------------------------- agent helpers --------------------------------- #
class TestAgentHelpers:
    def test_visit_counts_and_updates(self):
        ag = QLearningAgent(0, "Growth", seed=1)
        assert ag.visit_counts.shape == (cfg.N_STATES, cfg.N_ACTIONS) and ag.n_updates == 0
        ag.update(3, 2, 1.0, 4, False)
        ag.update(3, 2, 1.0, 4, False)
        assert ag.visit_counts[3, 2] == 2 and ag.visit_counts.sum() == 2 and ag.n_updates == 2

    def test_q_values_is_a_copy(self):
        ag = QLearningAgent(0, "Growth", seed=1)
        q = ag.q_values(5)
        q[:] = 99
        assert np.all(ag.q_table[5] == cfg.Q_INIT_VALUE)

    def test_greedy_action_does_not_consume_agent_rng(self):
        ag = QLearningAgent(0, "Growth", seed=1)
        ag.q_table[7] = [0, 1, 5, 5, 0, 0, 0]
        before = copy.deepcopy(ag.rng.bit_generator.state)
        picks = {ag.greedy_action(7, rng=np.random.default_rng(i)) for i in range(40)}
        assert picks == {2, 3}                                          # ties broken randomly between the maxima
        assert ag.rng.bit_generator.state == before
        assert ag.greedy_action(7) == 2                                 # no rng -> first maximum

    def test_trainer_progress_callback_offset_and_td_columns(self):
        env = _env(seed=42)
        agents = build_q_agents(AGENT_IDS, AGENT_ROLES, QHyperParams(), 42)
        calls = []
        df = Trainer(env, agents).train(5, verbose=False, progress_callback=lambda d, t, r: calls.append((d, t)),
                                        episode_offset=10)
        assert calls == [(i, 5) for i in range(1, 6)]
        assert list(df["episode"]) == [10, 11, 12, 13, 14]
        assert all(f"td_{a}" in df.columns for a in AGENT_IDS) and (df["td_0"] >= 0).all()


# --------------------------- state utils ----------------------------------- #
class TestStateUtils:
    def test_encode_decode_roundtrip_all_states(self):
        seen = set()
        for s in range(cfg.N_STATES):
            d = decode_state(s)
            assert encode_state(d["regime"], d["equity_bucket"], d["bond_bucket"], d["gold_bucket"]) == s
            seen.add(tuple(d.values()))
        assert len(seen) == cfg.N_STATES

    def test_matches_environment_discretisation(self):
        env = _env(seed=2)
        states = env.reset()
        rng = np.random.default_rng(1)
        for _ in range(20):
            states, *_ = env.step({a: int(rng.integers(0, cfg.N_ACTIONS)) for a in AGENT_IDS})
            for a in AGENT_IDS:
                assert decode_state(states[a])["regime"] == env.market.regime

    @pytest.mark.parametrize("bad", [-1, cfg.N_STATES, 1000])
    def test_decode_rejects_out_of_range(self, bad):
        with pytest.raises(ValueError):
            decode_state(bad)

    def test_encode_rejects_bad_bucket(self):
        with pytest.raises(ValueError):
            encode_state(0, 3, 0, 0)

    def test_describe(self):
        assert "Bull" in describe_state(encode_state(2, 0, 1, 2))


# --------------------------- analytics ------------------------------------- #
class TestAnalytics:
    def test_known_series(self):
        v = np.array([100.0, 110.0, 99.0, 108.9])
        assert analytics.cumulative_returns(v)[-1] == pytest.approx(0.089)
        assert analytics.drawdown_series(v).max() == pytest.approx(0.10)
        m = analytics.portfolio_metrics(v, np.array([1.0, 2.0, 0.5]))
        assert m["total_return"] == pytest.approx(0.089) and m["max_drawdown"] == pytest.approx(0.10)
        assert m["total_txcost"] == pytest.approx(3.5) and m["n_steps"] == 3

    def test_flat_series_has_zero_risk(self):
        m = analytics.portfolio_metrics(np.full(10, 100.0))
        assert m["volatility"] == 0 and m["max_drawdown"] == 0 and m["sharpe"] == 0

    def test_short_series_safe(self):
        assert analytics.portfolio_metrics(np.array([5.0]))["total_return"] == 0.0
        assert len(analytics.rolling_volatility(np.array([1.0, 2.0]), 5)) == 1

    def test_value_series_includes_initial_capital(self):
        env = _env(seed=11)
        traj = evaluate_out_of_sample(AGENT_IDS, AGENT_ROLES, CAPS,
                                      build_q_agents(AGENT_IDS, AGENT_ROLES, QHyperParams(), 1), eval_seeds=[60],
                                      episode_length=20)[0]
        vs = analytics.value_series(traj, AGENT_IDS, sum(CAPS.values()))
        assert len(vs) == 21 and vs[0] == sum(CAPS.values())
        assert analytics.txcost_series(traj, AGENT_IDS).shape == (20,)
        # sum of per-agent series == combined
        parts = sum(analytics.value_series(traj, [a], CAPS[a]) for a in AGENT_IDS)
        assert np.allclose(parts, vs)

    def test_regime_allocation_uses_current_regime(self):
        traj = [{"current_regime": 2, "actions": {0: 0}}, {"current_regime": 2, "actions": {0: 6}},
                {"current_regime": 0, "actions": {0: 6}}]
        out = analytics.regime_allocation(traj, 0)
        assert np.allclose(out[2], (cfg.ACTION_BASKETS[0] + cfg.ACTION_BASKETS[6]) / 2)
        assert np.allclose(out[0], cfg.ACTION_BASKETS[6]) and np.isnan(out[1]).all()


# --------------------------- experiment engine ----------------------------- #
class TestExperimentEngine:
    def test_same_config_same_numbers(self):
        rc = RunConfig(experiment="D", n_train_episodes=30, n_eval_episodes=5)
        a, b = run_experiment(rc)["summary"], run_experiment(rc)["summary"]
        assert a == b

    def test_different_train_seed_changes_result(self):
        a = run_experiment(RunConfig(experiment="D", n_train_episodes=30, n_eval_episodes=5, train_seed=1))["summary"]
        b = run_experiment(RunConfig(experiment="D", n_train_episodes=30, n_eval_episodes=5, train_seed=2))["summary"]
        assert a != b

    @pytest.mark.parametrize("exp,has_training", [("A", False), ("B", False), ("C", True), ("D", True)])
    def test_training_only_for_learning_experiments(self, exp, has_training):
        r = run_experiment(RunConfig(experiment=exp, n_train_episodes=10, n_eval_episodes=3))
        assert (r["training_df"] is not None) == has_training
        assert len(r["eval_trajectories"]) == 3
        assert r["summary"]["total_initial_capital"] == cfg.TOTAL_INITIAL_CAPITAL

    def test_evaluation_is_frozen(self):
        agents = build_q_agents(AGENT_IDS, AGENT_ROLES, QHyperParams(), 3)
        Trainer(_env(seed=3), agents).train(15, verbose=False)
        q_before = {a: agents[a].q_table.copy() for a in AGENT_IDS}
        eps_before = {a: agents[a].epsilon for a in AGENT_IDS}
        visits_before = {a: agents[a].visit_counts.copy() for a in AGENT_IDS}
        evaluate_agents(agents, AGENT_IDS, AGENT_ROLES, CAPS, list(range(51, 56)))
        for a in AGENT_IDS:
            assert np.array_equal(agents[a].q_table, q_before[a]) and agents[a].epsilon == eps_before[a]
            assert np.array_equal(agents[a].visit_counts, visits_before[a])

    def test_summary_matches_manual_computation(self):
        r = run_experiment(RunConfig(experiment="B", n_eval_episodes=4))
        traj = r["eval_trajectories"][0]
        manual = sum(s["capital"][a] for a in AGENT_IDS for s in [traj[-1]]) / cfg.TOTAL_INITIAL_CAPITAL - 1
        assert r["summary"]["combined_returns_per_seed"][0] == pytest.approx(manual)
        assert len(r["summary"]["mean_equity_curve"]) == cfg.EPISODE_LENGTH + 1
        # per-agent returns reported per seed, one per held-out episode
        assert all(len(v) == 4 for v in r["summary"]["per_agent_returns_per_seed"].values())

    def test_fixed_basket_choice_is_used(self):
        r = run_experiment(RunConfig(experiment="B", fixed_action=5, n_eval_episodes=2))
        assert all(v == 5 for v in r["summary"]["modal_action"].values())

    @pytest.mark.parametrize("kwargs,fragment", [
        ({"experiment": "Z"}, "Unknown experiment"),
        ({"n_train_episodes": 0}, "Training episodes"),
        ({"n_eval_episodes": 0}, "evaluation episodes"),
        ({"episode_length": 0}, "Episode length"),
        ({"fixed_action": 99}, "Fixed basket"),
        ({"reward_mode": "nope"}, "reward_mode"),
        ({"eval_seed_start": 40, "n_eval_episodes": 10}, "leak"),
        ({"hp": QHyperParams(alpha=0)}, "alpha"),
        ({"hp": QHyperParams(gamma=1.5)}, "gamma"),
        ({"hp": QHyperParams(epsilon_initial=0.1, epsilon_min=0.5)}, "epsilon_min"),
        ({"hp": QHyperParams(epsilon_decay=0)}, "epsilon_decay"),
    ])
    def test_validation_rejects_bad_config(self, kwargs, fragment):
        base = dict(experiment="D", n_train_episodes=5, n_eval_episodes=3)
        base.update(kwargs)
        with pytest.raises(ValueError, match=fragment):
            run_experiment(RunConfig(**base))

    def test_training_curves_shape(self):
        r = run_experiment(RunConfig(experiment="D", n_train_episodes=60, n_eval_episodes=2))
        tc = training_curves(r["training_df"], AGENT_IDS)
        assert set(tc) >= {"reward", "epsilon", "td", "capital", "total_capital"}
        assert tc["reward"]["0"]["episode"][-1] == 59
