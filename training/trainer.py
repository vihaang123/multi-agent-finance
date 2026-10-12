"""
training/trainer.py

Runs the episode loop against a FinancialEnv for any number of agents
(1 or many, learning or baseline -- anything implementing BaseAgent).

Logs, per episode: total & per-agent reward, final capital per agent,
final drawdown per agent, mean epsilon, and a snapshot of the average
allocation basket chosen by each agent (for later MAS/behaviour plots).
"""

from typing import Dict, List

import numpy as np
import pandas as pd

import config as cfg
from environment.financial_env import FinancialEnv


class Trainer:
    def __init__(self, env: FinancialEnv, agents: Dict[int, "BaseAgent"]):
        self.env = env
        self.agents = agents
        self.agent_ids = list(agents.keys())

    def run_episode(self, collect_trajectory: bool = False):
        states = self.env.reset()
        done = False
        episode_reward = {aid: 0.0 for aid in self.agent_ids}
        action_counts = {aid: np.zeros(cfg.N_ACTIONS, dtype=int) for aid in self.agent_ids}
        td_abs_sum = {aid: 0.0 for aid in self.agent_ids}
        n_steps = 0
        trajectory = [] if collect_trajectory else None

        while not done:
            actions = {aid: self.agents[aid].select_action(states[aid]) for aid in self.agent_ids}
            next_states, rewards, done, info = self.env.step(actions)

            for aid in self.agent_ids:
                self.agents[aid].update(states[aid], actions[aid], rewards[aid], next_states[aid], done)
                episode_reward[aid] += rewards[aid]
                action_counts[aid][actions[aid]] += 1
                td_abs_sum[aid] += abs(getattr(self.agents[aid], "last_td_error", 0.0))
            n_steps += 1

            if collect_trajectory:
                trajectory.append({
                    "current_regime": info["current_regime"],
                    "next_regime": info["next_regime"],
                    "capital": dict(info["capital"]),
                    "drawdown": dict(info["drawdown"]),
                    "actual_weights": {aid: info["actual_weights"][aid].copy() for aid in self.agent_ids},
                    "actions": dict(actions),
                    "rewards": dict(rewards),
                    "txcost": dict(info["txcost"]),
                })

            states = next_states

        for aid in self.agent_ids:
            self.agents[aid].on_episode_end()

        final_capital = {aid: self.env.portfolios[aid].capital for aid in self.agent_ids}
        final_drawdown = {aid: self.env.portfolios[aid].drawdown for aid in self.agent_ids}
        epsilons = {aid: getattr(self.agents[aid], "epsilon", 0.0) for aid in self.agent_ids}

        result = {
            "episode_reward": episode_reward,
            "final_capital": final_capital,
            "final_drawdown": final_drawdown,
            "epsilons": epsilons,
            "action_counts": action_counts,
            "mean_abs_td": {aid: td_abs_sum[aid] / max(1, n_steps) for aid in self.agent_ids},
        }
        if collect_trajectory:
            result["trajectory"] = trajectory
        return result

    def train(self, num_episodes: int = cfg.NUM_EPISODES, log_every: int = 100, verbose: bool = True,
              progress_callback=None, episode_offset: int = 0) -> pd.DataFrame:
        """progress_callback(done, total, row) is called after every episode (used by the
        dashboard's progress bar); episode_offset numbers episodes when continuing a run."""
        rows: List[dict] = []
        for ep in range(num_episodes):
            result = self.run_episode(collect_trajectory=False)
            row = {"episode": ep + episode_offset}
            for aid in self.agent_ids:
                row[f"reward_{aid}"] = result["episode_reward"][aid]
                row[f"capital_{aid}"] = result["final_capital"][aid]
                row[f"drawdown_{aid}"] = result["final_drawdown"][aid]
                row[f"epsilon_{aid}"] = result["epsilons"][aid]
                row[f"td_{aid}"] = result["mean_abs_td"][aid]
            row["total_reward"] = sum(result["episode_reward"].values())
            row["total_capital"] = sum(result["final_capital"].values())
            rows.append(row)
            if progress_callback is not None:
                progress_callback(ep + 1, num_episodes, row)

            if verbose and (ep + 1) % log_every == 0:
                mean_recent = np.mean([r["total_reward"] for r in rows[-log_every:]])
                print(f"  episode {ep + 1:5d}/{num_episodes} | "
                      f"mean total reward (last {log_every}): {mean_recent:8.3f} | "
                      f"total capital: {row['total_capital']:12,.0f} | "
                      f"eps: {row.get(f'epsilon_{self.agent_ids[0]}', 0):.3f}")

        return pd.DataFrame(rows)
