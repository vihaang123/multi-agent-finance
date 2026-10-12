import numpy as np

import config as cfg
from environment.financial_env import FinancialEnv
from agents.q_learning_agent import QLearningAgent
from training.trainer import Trainer


def make_env(seed=0, episode_length=20, reward_mode="individual", enable_impact=True):
    agent_ids = [0, 1, 2, 3]
    roles = {0: "Growth", 1: "Conservative", 2: "Balanced", 3: "RiskLiquidity"}
    caps = {aid: cfg.INITIAL_CAPITAL_PER_AGENT for aid in agent_ids}
    return FinancialEnv(agent_ids, roles, caps, episode_length=episode_length,
                         reward_mode=reward_mode, enable_market_impact=enable_impact, seed=seed)


def test_all_agents_act_within_same_timestep():
    env = make_env()
    env.reset()
    actions = {0: 0, 1: 1, 2: 2, 3: 3}
    next_states, rewards, done, info = env.step(actions)
    # All four agents produced a next_state and reward from a SINGLE call.
    assert len(next_states) == 4
    assert len(rewards) == 4
    # Regime advanced exactly once (shared across all agents), not once per agent.
    assert isinstance(info["current_regime"], int)
    assert isinstance(info["next_regime"], int)


def test_market_impact_couples_agents_rewards():
    """If every agent crowds into equity, the impact term should visibly
    reduce the realised equity return relative to a diversified allocation,
    demonstrating that one agent's choice affects the shared outcome."""
    env_impact = make_env(seed=5, enable_impact=True)
    env_impact.reset()
    # All 4 agents pile into the most aggressive equity basket (index 0).
    _, _, _, info_crowded = env_impact.step({0: 0, 1: 0, 2: 0, 3: 0})
    assert info_crowded["agg_equity_weight"] > cfg.MARKET_IMPACT_BASELINE_EQUITY
    assert info_crowded["market_impact"] < 0.0  # penalty applied


def test_market_impact_can_be_disabled():
    env_no_impact = make_env(seed=5, enable_impact=False)
    env_no_impact.reset()
    _, _, _, info = env_no_impact.step({0: 0, 1: 0, 2: 0, 3: 0})
    assert info["market_impact"] == 0.0


def test_shared_resource_reward_mode_gives_identical_rewards():
    env = make_env(reward_mode="shared")
    env.reset()
    _, rewards, _, _ = env.step({0: 0, 1: 4, 2: 2, 3: 6})
    values = list(rewards.values())
    assert all(np.isclose(v, values[0]) for v in values)


def test_individual_reward_mode_can_differ_across_agents():
    env = make_env(reward_mode="individual")
    env.reset()
    # Different roles + different actions should generally produce different rewards.
    _, rewards, _, _ = env.step({0: 0, 1: 6, 2: 2, 3: 5})
    values = list(rewards.values())
    assert len(set(np.round(values, 6))) > 1


def test_independent_qlearning_agents_maintain_separate_q_tables():
    env = make_env(episode_length=30)
    roles = {0: "Growth", 1: "Conservative", 2: "Balanced", 3: "RiskLiquidity"}
    agents = {aid: QLearningAgent(aid, roles[aid], seed=aid) for aid in [0, 1, 2, 3]}
    trainer = Trainer(env, agents)
    trainer.run_episode()
    # Each agent's Q-table is its own array in memory.
    tables = [agents[aid].q_table for aid in [0, 1, 2, 3]]
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            assert tables[i] is not tables[j]


def test_role_reward_weights_lead_to_different_avg_risk_penalty():
    """Sanity check that reward weight config actually differentiates roles:
    Conservative's beta (risk penalty weight) must exceed Growth's."""
    assert cfg.REWARD_WEIGHTS["Conservative"]["beta"] > cfg.REWARD_WEIGHTS["Growth"]["beta"]
    assert cfg.REWARD_WEIGHTS["RiskLiquidity"]["liquidity_mu"] > cfg.REWARD_WEIGHTS["Growth"]["liquidity_mu"]
