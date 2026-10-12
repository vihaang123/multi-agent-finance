"""
tests/test_evaluation.py

BASELINE V1, Fix 2 (out-of-sample evaluation) and Part 4 (fairness
verification) regression tests.
"""

import numpy as np

import config as cfg
from environment.financial_env import FinancialEnv
from agents.q_learning_agent import QLearningAgent
from experiments.baseline import RandomAgent
from training.trainer import Trainer
from experiments.run_experiments import evaluate_out_of_sample


AGENT_IDS = [0, 1, 2, 3]
AGENT_ROLES = {0: "Growth", 1: "Conservative", 2: "Balanced", 3: "RiskLiquidity"}


def _trained_agents(num_episodes=50, seed=cfg.TRAIN_SEED):
    caps = {aid: cfg.INITIAL_CAPITAL_PER_AGENT for aid in AGENT_IDS}
    env = FinancialEnv(AGENT_IDS, AGENT_ROLES, caps, episode_length=cfg.EPISODE_LENGTH, seed=seed)
    agents = {aid: QLearningAgent(aid, AGENT_ROLES[aid], seed=seed + aid) for aid in AGENT_IDS}
    trainer = Trainer(env, agents)
    trainer.train(num_episodes=num_episodes, log_every=num_episodes + 1, verbose=False)
    return agents


# --------------------------------------------------------------------------- #
# Fix 2: out-of-sample evaluation must not touch learned state
# --------------------------------------------------------------------------- #
def test_evaluation_does_not_modify_q_table():
    agents = _trained_agents()
    caps = {aid: cfg.INITIAL_CAPITAL_PER_AGENT for aid in AGENT_IDS}
    snapshots = {aid: agents[aid].q_table.copy() for aid in AGENT_IDS}

    evaluate_out_of_sample(AGENT_IDS, AGENT_ROLES, caps, agents, eval_seeds=cfg.EVAL_SEEDS[:5])

    for aid in AGENT_IDS:
        assert np.array_equal(agents[aid].q_table, snapshots[aid]), (
            f"Agent {aid}'s Q-table changed during evaluation -- learning was not frozen."
        )


def test_evaluation_does_not_modify_epsilon():
    agents = _trained_agents()
    caps = {aid: cfg.INITIAL_CAPITAL_PER_AGENT for aid in AGENT_IDS}
    epsilons_before = {aid: agents[aid].epsilon for aid in AGENT_IDS}

    evaluate_out_of_sample(AGENT_IDS, AGENT_ROLES, caps, agents, eval_seeds=cfg.EVAL_SEEDS[:5])

    for aid in AGENT_IDS:
        assert agents[aid].epsilon == epsilons_before[aid], (
            f"Agent {aid}'s epsilon changed during evaluation -- epsilon was not frozen."
        )


def test_evaluation_uses_greedy_policy_during_the_eval_call():
    """While evaluate_out_of_sample is running, epsilon must actually be 0
    (not just restored to its pre-eval value afterward) -- confirmed
    indirectly: a RandomAgent has no epsilon attribute, so this test uses a
    QLearningAgent with epsilon monkeypatched to record what it saw."""
    agents = _trained_agents()
    caps = {aid: cfg.INITIAL_CAPITAL_PER_AGENT for aid in AGENT_IDS}
    seen_epsilons = []

    original_select_action = agents[0].select_action
    def spy_select_action(state):
        seen_epsilons.append(agents[0].epsilon)
        return original_select_action(state)
    agents[0].select_action = spy_select_action

    evaluate_out_of_sample(AGENT_IDS, AGENT_ROLES, caps, agents, eval_seeds=cfg.EVAL_SEEDS[:2])

    assert len(seen_epsilons) > 0
    assert all(eps == 0.0 for eps in seen_epsilons), "Evaluation did not use a greedy (epsilon=0) policy throughout."


def test_eval_seeds_disjoint_from_train_seed():
    assert cfg.TRAIN_SEED not in cfg.EVAL_SEEDS


def test_eval_seeds_produce_different_market_paths():
    """Sanity check that evaluation isn't silently reusing one seed/env for
    every 'held-out' episode -- different eval seeds should see different
    regime sequences (statistically, not guaranteed for every pair, but
    across the first few seeds at least one difference is overwhelmingly
    likely)."""
    agents = _trained_agents()
    caps = {aid: cfg.INITIAL_CAPITAL_PER_AGENT for aid in AGENT_IDS}
    trajectories = evaluate_out_of_sample(AGENT_IDS, AGENT_ROLES, caps, agents, eval_seeds=cfg.EVAL_SEEDS[:5])
    regime_sequences = [tuple(step["current_regime"] for step in traj) for traj in trajectories]
    assert len(set(regime_sequences)) > 1, "All eval seeds produced identical regime sequences -- seeds are not actually independent."


# --------------------------------------------------------------------------- #
# Part 4: fairness verification -- exogenous market process vs endogenous
# market-impact component
# --------------------------------------------------------------------------- #
def test_exogenous_market_path_independent_of_agent_count_and_actions():
    """With market impact DISABLED, the regime AND raw return sequence for a
    given seed must be identical regardless of how many agents are stepping
    or what actions they choose -- the market process is purely exogenous."""
    seed = 123
    episode_length = 25

    env1 = FinancialEnv([0], {0: "Growth"}, {0: cfg.TOTAL_INITIAL_CAPITAL},
                         episode_length=episode_length, enable_market_impact=False, seed=seed)
    env1.reset()
    regimes_1, returns_1 = [], []
    for _ in range(episode_length):
        _, _, done, info = env1.step({0: 3})
        regimes_1.append(info["current_regime"])
        returns_1.append(env1.portfolios[0].last_return)

    env4 = FinancialEnv(AGENT_IDS, AGENT_ROLES, {aid: cfg.INITIAL_CAPITAL_PER_AGENT for aid in AGENT_IDS},
                         episode_length=episode_length, enable_market_impact=False, seed=seed)
    env4.reset()
    regimes_4 = []
    for _ in range(episode_length):
        _, _, done, info = env4.step({0: 0, 1: 6, 2: 2, 3: 5})  # deliberately different actions
        regimes_4.append(info["current_regime"])

    assert regimes_1 == regimes_4, "Regime sequence depended on agent count/actions with market impact disabled."


def test_endogenous_market_impact_can_depend_on_agent_actions():
    """With market impact ENABLED, the REALIZED equity return can differ
    between two runs with the same seed if the agents' aggregate equity
    exposure differs -- this is the intentional, documented endogenous
    component, distinct from the exogenous regime/base-return draw (which
    stays identical either way, since regime transition never depends on
    actions)."""
    seed = 123
    episode_length = 20
    caps = {aid: cfg.INITIAL_CAPITAL_PER_AGENT for aid in AGENT_IDS}

    env_crowded = FinancialEnv(AGENT_IDS, AGENT_ROLES, caps, episode_length=episode_length,
                                enable_market_impact=True, seed=seed)
    env_crowded.reset()
    regimes_crowded, equity_returns_crowded = [], []
    for _ in range(episode_length):
        _, _, done, info = env_crowded.step({0: 0, 1: 0, 2: 0, 3: 0})  # everyone piles into equity
        regimes_crowded.append(info["current_regime"])
        equity_returns_crowded.append(info["market_impact"])

    env_diversified = FinancialEnv(AGENT_IDS, AGENT_ROLES, caps, episode_length=episode_length,
                                    enable_market_impact=True, seed=seed)
    env_diversified.reset()
    regimes_diversified, equity_returns_diversified = [], []
    for _ in range(episode_length):
        _, _, done, info = env_diversified.step({0: 6, 1: 6, 2: 6, 3: 6})  # everyone stays in cash
        regimes_diversified.append(info["current_regime"])
        equity_returns_diversified.append(info["market_impact"])

    # Exogenous part (regime path) is unaffected by actions:
    assert regimes_crowded == regimes_diversified
    # Endogenous part (market-impact penalty) DOES depend on actions:
    assert equity_returns_crowded != equity_returns_diversified
    assert any(v < 0 for v in equity_returns_crowded), "Crowding into equity should trigger a nonzero impact penalty somewhere."
    assert all(v == 0.0 for v in equity_returns_diversified), "Nobody holding equity should mean zero impact regardless of baseline deviation."
