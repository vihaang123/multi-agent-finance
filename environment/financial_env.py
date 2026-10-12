"""
environment/financial_env.py

The shared environment all agents act in. Owns the market simulator and
one Portfolio sleeve per agent. Exposes a simultaneous-action, multi-agent
step() API:

    states  = env.reset()                      # dict: agent_id -> state
    actions = {aid: agent.select_action(states[aid]) for aid in agent_ids}
    next_states, rewards, done, info = env.step(actions)

Multi-agent interaction mechanism (see architecture review, point 4):
every agent's chosen target weight feeds a capital-weighted AGGREGATE
equity exposure. If that aggregate crowds too far from a "neutral"
baseline, a market-impact penalty is subtracted from the realised
equity return for EVERYONE this step. This is the coupling that makes
one agent's decision affect every other agent's payoff, on top of the
shared regime/return draw they all experience.

No look-ahead: the impact term uses THIS step's just-chosen target
weights (already known to the environment when returns are drawn), never
a future return.
"""

from typing import Dict, List

import numpy as np

import config as cfg
from environment.market import MarketSimulator
from environment.portfolio import Portfolio


def _bucket(weight: float) -> int:
    edges = cfg.ALLOC_BUCKET_EDGES
    if weight < edges[0]:
        return 0
    elif weight < edges[1]:
        return 1
    return 2


class FinancialEnv:
    def __init__(
        self,
        agent_ids: List[int],
        agent_roles: Dict[int, str],
        initial_capitals: Dict[int, float],
        episode_length: int = cfg.EPISODE_LENGTH,
        reward_mode: str = cfg.REWARD_MODE,
        enable_market_impact: bool = True,
        seed: int = None,
    ):
        self.agent_ids = list(agent_ids)
        self.agent_roles = agent_roles
        self.initial_capitals = initial_capitals
        self.episode_length = episode_length
        self.reward_mode = reward_mode
        self.enable_market_impact = enable_market_impact

        self.rng = np.random.default_rng(seed)
        self.market = MarketSimulator(self.rng)
        self.portfolios = {aid: Portfolio(initial_capitals[aid]) for aid in self.agent_ids}
        self.t = 0

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #
    def reset(self) -> Dict[int, int]:
        self.market.reset()
        for p in self.portfolios.values():
            p.reset()
        self.t = 0
        return {aid: self._discretize_state(aid) for aid in self.agent_ids}

    def get_state(self, agent_id: int) -> int:
        return self._discretize_state(agent_id)

    @staticmethod
    def valid_actions() -> List[int]:
        return list(range(cfg.N_ACTIONS))

    def step(self, actions: Dict[int, int]):
        for aid, a in actions.items():
            if a < 0 or a >= cfg.N_ACTIONS:
                raise ValueError(f"Invalid action {a} for agent {aid}; must be in [0, {cfg.N_ACTIONS}).")

        target_weights = {aid: cfg.ACTION_BASKETS[actions[aid]] for aid in actions}

        # --- 1. Rebalance every sleeve to its target weights (charges cost) ---
        pre_rebalance_capital = {aid: self.portfolios[aid].capital for aid in self.agent_ids}
        txcost_frac = {}
        for aid in self.agent_ids:
            cost = self.portfolios[aid].rebalance(target_weights[aid])
            pre_cap = pre_rebalance_capital[aid]
            txcost_frac[aid] = (cost / pre_cap) if pre_cap > 0 else 0.0

        # --- 2. Aggregate (capital-weighted) equity exposure -> market impact ---
        total_capital = sum(self.portfolios[aid].capital for aid in self.agent_ids)
        if total_capital > 0:
            agg_equity_weight = sum(
                self.portfolios[aid].capital * target_weights[aid][cfg.EQUITY]
                for aid in self.agent_ids
            ) / total_capital
        else:
            agg_equity_weight = 0.0

        impact = 0.0
        if self.enable_market_impact:
            # One-sided: crowding INTO equity above the neutral baseline puts
            # downward pressure on the realized return (too much money chasing
            # the same asset). Being UNDERweight equity relative to baseline
            # must NOT be penalized the same way -- there is no real-world
            # mechanism by which other agents avoiding equity should reduce
            # the return for an agent who still holds it normally. An earlier,
            # symmetric version of this penalty (squaring the raw deviation in
            # either direction) inadvertently punished the Growth agent simply
            # for being the only one willing to hold equity while the other
            # three roles preferred cash -- a modelling bug, not a genuine
            # crowding effect. See README, Results & Analysis.
            excess = max(0.0, agg_equity_weight - cfg.MARKET_IMPACT_BASELINE_EQUITY)
            impact = -cfg.MARKET_IMPACT_KAPPA * (excess ** 2)

        # --- 3. Draw this step's stochastic market returns, apply impact to equity ---
        current_regime = self.market.regime
        base_returns = self.market.sample_returns(current_regime)
        adjusted_returns = base_returns.copy()
        adjusted_returns[cfg.EQUITY] += impact

        # --- 4. Risk exposure of the CHOSEN basket, using the CURRENT regime's
        #        known volatility (mean-variance-style: same numerical scale as
        #        returns, so beta and alpha trade off meaningfully instead of
        #        one structurally dominating the other regardless of role). ---
        sigma_vec = self.market.sigma_vector(current_regime)
        risk_exposure = {
            aid: float(np.dot(target_weights[aid], sigma_vec)) for aid in self.agent_ids
        }

        # --- 5. Apply market returns to every sleeve ---
        # (instrumentation only: rupee value held in each asset right after
        #  rebalancing and BEFORE this step's returns hit)
        asset_values_pre = {aid: self.portfolios[aid].asset_values.copy() for aid in self.agent_ids}
        total_capital_before_returns = sum(self.portfolios[aid].capital for aid in self.agent_ids)
        for aid in self.agent_ids:
            self.portfolios[aid].apply_returns(adjusted_returns)
        total_capital_after_returns = sum(self.portfolios[aid].capital for aid in self.agent_ids)
        shared_return = (
            (total_capital_after_returns - total_capital_before_returns) / total_capital_before_returns
            if total_capital_before_returns > 0 else 0.0
        )
        shared_log_return = (
            float(np.log(total_capital_after_returns / total_capital_before_returns))
            if total_capital_before_returns > 0 and total_capital_after_returns > 0 else 0.0
        )
        shared_reward = float(np.clip(shared_log_return * cfg.REWARD_SCALE, -cfg.REWARD_CLIP, cfg.REWARD_CLIP))

        # --- 6. Per-agent reward ---
        rewards = {}
        individual_rewards = {}
        reward_components = {}
        for aid in self.agent_ids:
            role = self.agent_roles[aid]
            w = cfg.REWARD_WEIGHTS[role]
            port = self.portfolios[aid]
            # Reward = the SAME five terms, in the SAME evaluation order, as before;
            # they are just also kept individually so the dashboard can display them.
            comp_return = w["alpha"] * port.last_log_return * cfg.REWARD_SCALE
            comp_risk = w["beta"] * risk_exposure[aid] * cfg.REWARD_SCALE
            comp_tx = w["lam"] * txcost_frac[aid] * cfg.REWARD_SCALE
            comp_dd = w["delta"] * port.drawdown_increase * cfg.REWARD_SCALE
            comp_liq = w["liquidity_mu"] * target_weights[aid][cfg.CASH]
            r_i = comp_return - comp_risk - comp_tx - comp_dd + comp_liq
            r_unclipped = float(r_i)
            r_i = float(np.clip(r_i, -cfg.REWARD_CLIP, cfg.REWARD_CLIP))
            reward_components[aid] = {
                "return": float(comp_return), "risk": -float(comp_risk), "transaction": -float(comp_tx),
                "drawdown": -float(comp_dd), "liquidity": float(comp_liq),
                "unclipped": r_unclipped, "clipped": r_i,
            }
            individual_rewards[aid] = r_i

            if self.reward_mode == "individual":
                rewards[aid] = r_i
            elif self.reward_mode == "shared":
                rewards[aid] = shared_reward
            elif self.reward_mode == "hybrid":
                rewards[aid] = cfg.HYBRID_ALPHA * r_i + (1 - cfg.HYBRID_ALPHA) * shared_reward
            else:
                raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

        # --- 7. Advance regime for next step, advance time ---
        next_regime = self.market.step_regime()
        self.t += 1
        done = self.t >= self.episode_length

        next_states = {aid: self._discretize_state(aid) for aid in self.agent_ids}

        info = {
            # current_regime: the regime that GENERATED this step's return/reward
            # (i.e. the one used in steps 3-4 above). This is what any analysis
            # asking "what did the agent do DURING regime X" should key on.
            "current_regime": current_regime,
            # next_regime: the regime that will be active for the NEXT decision
            # (already reflected in next_states). Exposed separately so no
            # caller has to guess which one "regime" meant -- an earlier
            # version only exposed a single "regime" field set to this value,
            # which silently mislabeled the regime-vs-allocation analysis by
            # one step. See README, Results & Analysis.
            "next_regime": next_regime,
            "agg_equity_weight": agg_equity_weight,
            "market_impact": impact,
            "shared_return": shared_return,
            "individual_rewards": individual_rewards,
            "target_weights": {aid: target_weights[aid].copy() for aid in self.agent_ids},
            "actual_weights": {aid: self.portfolios[aid].actual_weights.copy() for aid in self.agent_ids},
            "capital": {aid: self.portfolios[aid].capital for aid in self.agent_ids},
            "drawdown": {aid: self.portfolios[aid].drawdown for aid in self.agent_ids},
            "drawdown_increase": {aid: self.portfolios[aid].drawdown_increase for aid in self.agent_ids},
            "txcost_frac": txcost_frac,
            "risk_exposure": risk_exposure,
            # ---- additive dashboard instrumentation (does not affect any maths) ----
            "base_returns": base_returns.copy(),          # raw draw for the current regime (exogenous)
            "asset_returns": adjusted_returns.copy(),      # after market impact on equity
            "asset_values_pre": asset_values_pre,          # Rs per asset after rebalance, before returns
            "asset_values": {aid: self.portfolios[aid].asset_values.copy() for aid in self.agent_ids},
            "asset_pnl": {aid: self.portfolios[aid].last_asset_pnl.copy() for aid in self.agent_ids},
            "txcost": {aid: txcost_frac[aid] * pre_rebalance_capital[aid] for aid in self.agent_ids},
            "reward_components": reward_components,
            "shared_reward": shared_reward,
        }
        return next_states, rewards, done, info

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _discretize_state(self, agent_id: int) -> int:
        w = self.portfolios[agent_id].actual_weights
        eq_b = _bucket(w[cfg.EQUITY])
        bond_b = _bucket(w[cfg.BONDS])
        gold_b = _bucket(w[cfg.GOLD])
        regime = self.market.regime
        state = regime * (cfg.N_ALLOC_BUCKETS ** 3) + eq_b * (cfg.N_ALLOC_BUCKETS ** 2) + bond_b * cfg.N_ALLOC_BUCKETS + gold_b
        return int(state)
