"""
environment/portfolio.py

Tracks one agent's capital sleeve: per-asset values, actual (drifted)
weights, transaction costs on rebalancing, running peak / drawdown.

Design note: an agent's *action* sets TARGET weights. The environment
rebalances the sleeve to those target weights (charging a transaction
cost proportional to turnover), then market returns are applied over
the step. This causes the ACTUAL weights to drift away from the target
before the next decision -- that drift is real information and is what
feeds the next state, rather than the state just parroting the last
action.
"""

import numpy as np
import config as cfg


class Portfolio:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.reset()

    def reset(self):
        self.capital = self.initial_capital
        self.peak_capital = self.initial_capital
        # Start fully in cash until the first action rebalances.
        self.asset_values = np.zeros(cfg.N_ASSETS)
        self.asset_values[cfg.CASH] = self.initial_capital
        self.last_turnover_cost = 0.0
        self.last_return = 0.0
        self.last_log_return = 0.0
        self.prev_drawdown = 0.0
        self.drawdown_increase = 0.0
        # Dashboard instrumentation (read-only record, never fed back into the maths):
        # rupee P&L earned by each asset sleeve during the most recent step.
        self.last_asset_pnl = np.zeros(cfg.N_ASSETS)
        self.history = [self.initial_capital]

    @property
    def actual_weights(self) -> np.ndarray:
        if self.capital <= 0:
            return np.zeros(cfg.N_ASSETS)
        return self.asset_values / self.capital

    @property
    def drawdown(self) -> float:
        """Cumulative drawdown from the all-time (episode) peak. This is the
        standard financial metric -- used for reporting (max_drawdown, etc),
        NOT for the per-step reward (see drawdown_increase)."""
        if self.peak_capital <= 0:
            return 0.0
        return max(0.0, (self.peak_capital - self.capital) / self.peak_capital)

    def rebalance(self, target_weights: np.ndarray) -> float:
        """
        Move the sleeve to target_weights, charging a transaction cost
        proportional to turnover (sum of absolute weight changes).
        Returns the cost charged (in currency units).
        """
        current_weights = self.actual_weights
        turnover = np.sum(np.abs(target_weights - current_weights))
        cost = cfg.TRANSACTION_COST_RATE * turnover * self.capital
        cost = min(cost, self.capital)  # can't cost more than we have

        self.capital -= cost
        self.asset_values = target_weights * self.capital
        self.last_turnover_cost = cost
        return cost

    def apply_returns(self, asset_returns: np.ndarray):
        """Apply one step of realised asset returns to the sleeve."""
        pre_capital = self.capital
        values_before = self.asset_values.copy()
        self.asset_values = self.asset_values * (1.0 + asset_returns)
        self.last_asset_pnl = self.asset_values - values_before
        self.capital = float(np.sum(self.asset_values))
        self.peak_capital = max(self.peak_capital, self.capital)
        self.last_return = (self.capital - pre_capital) / pre_capital if pre_capital > 0 else 0.0
        # Reward-facing return signal: LOG return, not raw fractional return.
        # Capital evolves MULTIPLICATIVELY (compounding) step to step, but a
        # per-step reward is an ARITHMETIC sum across the episode. Those two
        # don't agree when volatility is non-trivial: E[log(1+r)] ~ E[r] -
        # Var(r)/2, so a policy with a positive average arithmetic return can
        # still lose money cumulatively (volatility drag) while still scoring
        # well on a reward built from raw arithmetic returns. Log returns ARE
        # additive across steps (sum of log returns = log of total compounded
        # growth), so summing per-step reward built from log returns is
        # consistent with the actual compounded financial outcome. See
        # README, Results & Analysis, for the empirical mismatch this fixed.
        self.last_log_return = float(np.log(self.capital / pre_capital)) if pre_capital > 0 and self.capital > 0 else 0.0
        self.history.append(self.capital)

        # Reward-facing signal: only the NEW drawdown created THIS step, not
        # the full historical distance from peak. Using cumulative drawdown
        # as a per-step penalty creates a "lingering tax" that keeps punishing
        # an agent for a single past bad stretch long after it has stopped
        # taking any risky action -- in practice this collapsed every
        # risk-averse role onto the same degenerate "hide in 100% cash
        # forever" policy (see README, Results & Analysis). Penalizing only
        # the marginal, THIS-STEP increase in drawdown keeps the signal tied
        # to the current decision instead of to history.
        current_drawdown = self.drawdown
        self.drawdown_increase = max(0.0, current_drawdown - self.prev_drawdown)
        self.prev_drawdown = current_drawdown
