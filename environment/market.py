"""
environment/market.py

Stochastic market simulator. Produces a regime-dependent random return
for every asset at every timestep, and evolves the regime itself via a
Markov transition matrix.

No look-ahead: at time t, only the regime and returns realised at t (and
everything before it) are ever exposed to agents.
"""

import numpy as np
import config as cfg


class MarketSimulator:
    def __init__(self, rng: np.random.Generator, initial_regime: int = cfg.NEUTRAL):
        self.rng = rng
        self.initial_regime = initial_regime
        self.regime = initial_regime

    def reset(self):
        self.regime = self.initial_regime
        return self.regime

    def step_regime(self):
        """Transition the regime for the *next* timestep using the Markov matrix."""
        probs = cfg.REGIME_TRANSITION[self.regime]
        self.regime = int(self.rng.choice(cfg.N_REGIMES, p=probs))
        return self.regime

    def sample_returns(self, regime: int) -> np.ndarray:
        """
        Draw one stochastic return per asset for the given regime.
        r_t ~ Normal(mu(regime, asset), sigma(regime, asset))
        Returns an array of shape (N_ASSETS,).
        """
        params = cfg.ASSET_RETURN_PARAMS[regime]
        returns = np.zeros(cfg.N_ASSETS)
        for asset_idx in range(cfg.N_ASSETS):
            mu, sigma = params[asset_idx]
            returns[asset_idx] = self.rng.normal(mu, sigma)
        return returns

    def sigma_vector(self, regime: int) -> np.ndarray:
        """
        The (known, model) per-asset volatility for a regime -- used by the
        environment as a forward-looking RISK signal at decision time (i.e.
        "how volatile is this choice expected to be given what we know about
        the current regime right now"), not a random draw. No look-ahead:
        this uses only the CURRENT regime, never next step's.
        """
        params = cfg.ASSET_RETURN_PARAMS[regime]
        return np.array([params[a][1] for a in range(cfg.N_ASSETS)])
