"""
dashboard/session.py - SimulationSession: a steppable run of the REAL
FinancialEnv.

The web API is STATELESS (Vercel serverless functions keep no memory between
requests), so a session is rebuilt on every request from (seed, market-impact
flag, the actions already taken) by `replay()`, and then advanced with `step()`.
Because the environment is deterministic given its seed and the actions, the
rebuilt session is identical to the one the user was watching.
All per-step randomness used by the *policies* (tie-breaking among equal
Q-values, the random policy) is derived from (seed, step, agent), never from a
stream that depends on how many requests came before.

The session never re-implements market, portfolio or reward maths; it only
calls FinancialEnv.step() and records everything the engine reports. Agents
are held by reference and are NEVER updated here (watching a simulation does
not train anyone) -- training happens only on the RL Lab page.

Policies (chosen per agent, changeable between steps):
  * "trained" : greedy argmax over the agent's Q-row (ties broken with an RNG
                derived from seed/step/agent, so the agent's own RNG is untouched)
  * "random"  : uniform random basket
  * "fixed"   : a basket index you choose (also serves as manual control)

History rows follow the same schema as evaluation trajectories
(current_regime, capital, drawdown, actual_weights, actions, rewards,
txcost, asset_pnl, ...) plus dashboard-only extras, so one set of analytics
functions serves both.
"""

from typing import Dict, List, Optional

import numpy as np

import config as cfg
from environment.financial_env import FinancialEnv
from environment.state_utils import describe_state
from experiments.run_experiments import AGENT_IDS, AGENT_ROLES

POLICY_KINDS = ("trained", "random", "fixed")
POLICY_CODES = {"trained": "t", "random": "r", "fixed": "f"}
CODE_TO_KIND = {v: k for k, v in POLICY_CODES.items()}
POLICY_LABELS = {"trained": "Trained Q-learning (greedy)", "random": "Random basket", "fixed": "Fixed / manual basket"}


class SimulationSession:
    def __init__(self, agents: Optional[Dict[int, object]] = None, seed: int = cfg.EVAL_SEED_START,
                 episode_length: int = cfg.EPISODE_LENGTH, enable_market_impact: bool = True,
                 agent_ids=None, roles=None, capitals=None):
        if episode_length < 1:
            raise ValueError("episode_length must be >= 1")
        self.agents = agents or {}
        self.agent_ids = list(agent_ids) if agent_ids is not None else list(AGENT_IDS)
        self.roles = dict(roles) if roles is not None else dict(AGENT_ROLES)
        self.capitals = dict(capitals) if capitals is not None else {a: cfg.INITIAL_CAPITAL_PER_AGENT for a in self.agent_ids}
        self.seed = int(seed)
        self.episode_length = int(episode_length)
        self.enable_market_impact = bool(enable_market_impact)
        self.policy_spec = {a: {"kind": "trained" if (a in self.agents and hasattr(self.agents[a], "q_table")) else "random",
                                "basket": 2} for a in self.agent_ids}
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None):
        """Return to t=0 on the same (or a new) market seed. Identical seed =
        identical market path, so different policies can be compared fairly."""
        if seed is not None:
            self.seed = int(seed)
        self.env = FinancialEnv(self.agent_ids, self.roles, self.capitals, self.episode_length,
                                reward_mode=cfg.REWARD_MODE, enable_market_impact=self.enable_market_impact,
                                seed=self.seed)
        self.states = self.env.reset()
        self.t = 0
        self.done = False
        self.history: List[dict] = []
        self.steps: List[dict] = []          # [{"a": [action per agent], "k": "tfrt"}] - what the client keeps
        self.initial_regime = self.env.market.regime

    def set_policy(self, aid: int, kind: str, basket: Optional[int] = None):
        if kind not in POLICY_KINDS:
            raise ValueError(f"Unknown policy kind '{kind}'")
        if kind == "trained" and not (aid in self.agents and hasattr(self.agents[aid], "q_table")):
            raise ValueError(f"Agent {aid} has no trained Q-table")
        if basket is not None:
            if not 0 <= int(basket) < cfg.N_ACTIONS:
                raise ValueError(f"basket must be in [0, {cfg.N_ACTIONS})")
            self.policy_spec[aid]["basket"] = int(basket)
        self.policy_spec[aid]["kind"] = kind

    @property
    def current_regime(self) -> int:
        return self.env.market.regime

    # ------------------------------------------------------------------ #
    def _q_row(self, aid: int):
        agent = self.agents.get(aid)
        return agent.q_values(self.states[aid]) if agent is not None and hasattr(agent, "q_table") else None

    def _decide(self, aid: int):
        spec = self.policy_spec[aid]
        state = self.states[aid]
        q = None
        if spec["kind"] == "trained":
            agent = self.agents[aid]
            q = agent.q_values(state)
            action = agent.greedy_action(state, rng=np.random.default_rng([self.seed, self.t, aid, 1]))
        elif spec["kind"] == "random":
            action = int(np.random.default_rng([self.seed, self.t, aid, 2]).integers(0, cfg.N_ACTIONS))
        else:
            action = int(spec["basket"])
        return action, q

    def step(self, forced: Optional[List[dict]] = None) -> Optional[dict]:
        """Advance one environment step. Returns the recorded row (None if finished).

        `forced` (used when replaying a client-held history) is {"a": [...], "k": "tfrt"}: the actions
        and policy kinds that were really used at this step, so they are re-applied rather than re-decided."""
        if self.done:
            return None
        decisions, qvals, states = {}, {}, dict(self.states)
        kinds = {}
        if forced is not None:
            acts, kcodes = forced["a"], forced["k"]
            if len(acts) != len(self.agent_ids) or len(kcodes) != len(self.agent_ids):
                raise ValueError("replayed step has the wrong number of agents")
            for i, aid in enumerate(self.agent_ids):
                a = int(acts[i])
                if not 0 <= a < cfg.N_ACTIONS:
                    raise ValueError(f"replayed action {a} out of range")
                decisions[aid], kinds[aid] = a, CODE_TO_KIND.get(kcodes[i], "fixed")
                qvals[aid] = self._q_row(aid) if kinds[aid] == "trained" else None
        else:
            for aid in self.agent_ids:
                decisions[aid], qvals[aid] = self._decide(aid)
                kinds[aid] = self.policy_spec[aid]["kind"]

        regime_before = self.env.market.regime
        next_states, rewards, done, info = self.env.step(decisions)
        self.t += 1
        row = {
            "t": self.t,
            # --- trajectory-compatible fields ---
            "current_regime": info["current_regime"], "next_regime": info["next_regime"],
            "capital": dict(info["capital"]), "drawdown": dict(info["drawdown"]),
            "actual_weights": {a: info["actual_weights"][a].copy() for a in self.agent_ids},
            "actions": dict(decisions), "rewards": dict(rewards),
            "txcost": dict(info["txcost"]),
            "asset_pnl": {a: info["asset_pnl"][a].copy() for a in self.agent_ids},
            # --- dashboard extras ---
            "regime_before": regime_before,
            "states": states, "state_desc": {a: describe_state(states[a]) for a in self.agent_ids},
            "q_values": qvals,
            "policy_kind": dict(kinds),
            "target_weights": {a: info["target_weights"][a].copy() for a in self.agent_ids},
            "asset_values_pre": {a: info["asset_values_pre"][a].copy() for a in self.agent_ids},
            "asset_values": {a: info["asset_values"][a].copy() for a in self.agent_ids},
            "asset_returns": info["asset_returns"].copy(), "base_returns": info["base_returns"].copy(),
            "market_impact": info["market_impact"], "agg_equity_weight": info["agg_equity_weight"],
            "reward_components": info["reward_components"], "risk_exposure": dict(info["risk_exposure"]),
        }
        self.history.append(row)
        self.steps.append({"a": [int(decisions[a]) for a in self.agent_ids],
                           "k": "".join(POLICY_CODES[kinds[a]] for a in self.agent_ids)})
        self.states = next_states
        self.done = bool(done)
        return row

    def replay(self, steps: List[dict]) -> "SimulationSession":
        """Re-apply a client-held list of steps (see step()). Starts from step 0."""
        if len(steps) > self.episode_length:
            raise ValueError("more steps than the episode length")
        self.reset()
        for st_ in steps:
            if self.step(forced=st_) is None:
                raise ValueError("replay continued past the end of the episode")
        return self

    def run_steps(self, n: int) -> int:
        done_steps = 0
        for _ in range(max(0, int(n))):
            if self.step() is None:
                break
            done_steps += 1
        return done_steps

    def run_to_end(self) -> int:
        return self.run_steps(self.episode_length - self.t)

    # ------------------------------------------------------------------ #
    @property
    def last(self) -> Optional[dict]:
        return self.history[-1] if self.history else None

    def capital_series(self, aid: Optional[int] = None) -> np.ndarray:
        """Value path from t=0 (initial capital) to now. aid=None -> total."""
        ids = self.agent_ids if aid is None else [aid]
        start = float(sum(self.capitals[a] for a in ids))
        return np.asarray([start] + [float(sum(r["capital"][a] for a in ids)) for r in self.history])

    def regime_path(self) -> List[int]:
        return [r["current_regime"] for r in self.history]

    def summary_line(self) -> str:
        return f"step {self.t}/{self.episode_length}" + (" (episode complete)" if self.done else "")
