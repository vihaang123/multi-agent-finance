"""
experiments/run_experiments.py

Implements Experiments A, B, C, D from the project brief:

  A. Random allocation baseline (multi-agent, non-learning)
  B. Fixed allocation baseline (multi-agent, non-learning, "Balanced" basket)
  C. Single-agent Q-Learning (one agent, full capital)
  D. 4-agent Independent Q-Learning (one Q-table per role)

Each experiment TRAINS on a single continuing seed (cfg.TRAIN_SEED) and is
then separately, genuinely OUT-OF-SAMPLE EVALUATED on cfg.EVAL_SEEDS -- a
disjoint block of seeds, each spinning up a brand new environment, with
learning disabled and Q-tables frozen (see evaluate_out_of_sample()).
Training results and evaluation results are reported and returned
separately; they are never averaged together.

Run directly:  python -m experiments.run_experiments
"""

import os

import numpy as np
import pandas as pd

import config as cfg
from environment.financial_env import FinancialEnv
from agents.q_learning_agent import QLearningAgent
from experiments.baseline import RandomAgent, FixedAgent
from training.trainer import Trainer
from evaluation import metrics


AGENT_IDS = [0, 1, 2, 3]
AGENT_ROLES = {0: "Growth", 1: "Conservative", 2: "Balanced", 3: "RiskLiquidity"}


def evaluate_out_of_sample(
    agent_ids, agent_roles, initial_capitals, agents, eval_seeds=None,
    episode_length=cfg.EPISODE_LENGTH, reward_mode=cfg.REWARD_MODE, enable_market_impact=True,
):
    """
    TRUE out-of-sample evaluation (BASELINE V1, Fix 2).

    For each seed in eval_seeds, builds a BRAND NEW FinancialEnv -- its own
    independent np.random.default_rng(seed) -- never continuing the training
    environment's RNG stream. Runs ONE greedy episode per seed:

      - epsilon is forced to 0 for the duration (and restored afterward)
      - agents[aid].update(...) is NEVER called          -> Q-tables frozen
      - agents[aid].on_episode_end() is NEVER called      -> epsilon frozen

    Default eval_seeds is cfg.EVAL_SEEDS ([51..100] by default), guaranteed
    disjoint from cfg.TRAIN_SEED (42) -- see config.py.

    Returns the list of per-seed trajectories (one per eval seed).
    """
    if eval_seeds is None:
        eval_seeds = cfg.EVAL_SEEDS

    saved_epsilons = {}
    for aid in agent_ids:
        if hasattr(agents[aid], "epsilon"):
            saved_epsilons[aid] = agents[aid].epsilon
            agents[aid].epsilon = 0.0

    trajectories = []
    for seed in eval_seeds:
        env = FinancialEnv(
            agent_ids=agent_ids, agent_roles=agent_roles, initial_capitals=initial_capitals,
            episode_length=episode_length, reward_mode=reward_mode,
            enable_market_impact=enable_market_impact, seed=seed,
        )
        states = env.reset()
        done = False
        trajectory = []
        while not done:
            actions = {aid: agents[aid].select_action(states[aid]) for aid in agent_ids}
            next_states, rewards, done, info = env.step(actions)
            # No agents[aid].update(...) call here -- this is what makes
            # evaluation non-learning. See tests/test_evaluation.py.
            trajectory.append({
                "current_regime": info["current_regime"],
                "next_regime": info["next_regime"],
                "capital": dict(info["capital"]),
                "drawdown": dict(info["drawdown"]),
                "actual_weights": {aid: info["actual_weights"][aid].copy() for aid in agent_ids},
                "actions": dict(actions),
                "rewards": dict(rewards),
                "txcost": dict(info["txcost"]),
                "asset_pnl": {aid: info["asset_pnl"][aid].copy() for aid in agent_ids},
            })
            states = next_states
        trajectories.append(trajectory)

    for aid, eps in saved_epsilons.items():
        agents[aid].epsilon = eps

    return trajectories


def _aggregate_eval_trajectories(trajectories, agent_ids, initial_capitals):
    """Aggregate financial + MAS metrics across a list of eval trajectories
    (one independent held-out episode per eval seed)."""
    per_agent_summaries = {aid: [] for aid in agent_ids}
    per_agent_action_counts = {aid: np.zeros(cfg.N_ACTIONS) for aid in agent_ids}

    for traj in trajectories:
        for aid in agent_ids:
            value_series = [step["capital"][aid] for step in traj]
            per_agent_summaries[aid].append(metrics.financial_summary(value_series))
            for step in traj:
                per_agent_action_counts[aid][step["actions"][aid]] += 1

    aggregated = {}
    for aid in agent_ids:
        summaries = per_agent_summaries[aid]
        aggregated[aid] = {
            "final_value_mean": float(np.mean([s["final_value"] for s in summaries])),
            "final_value_std": float(np.std([s["final_value"] for s in summaries])),
            "total_return_mean": float(np.mean([s["total_return"] for s in summaries])),
            "total_return_std": float(np.std([s["total_return"] for s in summaries])),
            "volatility_mean": float(np.mean([s["volatility"] for s in summaries])),
            "max_drawdown_mean": float(np.mean([s["max_drawdown"] for s in summaries])),
            "sharpe_mean": float(np.mean([s["sharpe"] for s in summaries])),
        }

    action_hists = {
        aid: (per_agent_action_counts[aid] / per_agent_action_counts[aid].sum()
              if per_agent_action_counts[aid].sum() > 0 else per_agent_action_counts[aid])
        for aid in agent_ids
    }
    specialization = metrics.specialization_score(action_hists)
    total_final = sum(a["final_value_mean"] for a in aggregated.values())
    total_initial = sum(initial_capitals.values())
    utilisation = total_final / total_initial if total_initial > 0 else 0.0

    return aggregated, specialization, utilisation


def _print_training_summary(name, training_df, agent_ids, roles):
    print(f"\n--- {name} : TRAINING RESULTS (in-sample, seed={cfg.TRAIN_SEED}) ---")
    for aid in agent_ids:
        s = metrics.rl_training_summary(training_df, aid)
        floor_txt = (f"ep {s['episode_epsilon_floor_reached']}"
                     if s["episode_epsilon_floor_reached"] is not None else "not reached")
        print(f"  Agent {aid} ({roles[aid]:>14s}): "
              f"avg_reward(last100)={s['avg_reward_final_window']:8.3f}  "
              f"reward_trend={s['reward_trend']:+7.3f}  "
              f"final_epsilon={s['final_epsilon']:.3f}  "
              f"epsilon_floor_reached={floor_txt}")


def _print_evaluation_summary(name, aggregated, specialization, utilisation, agent_ids, roles, n_eval):
    print(f"\n--- {name} : OUT-OF-SAMPLE EVALUATION RESULTS "
          f"({n_eval} held-out seeds {cfg.EVAL_SEED_START}-{cfg.EVAL_SEED_START + n_eval - 1}, "
          f"greedy policy, Q-tables frozen) ---")
    for aid in agent_ids:
        a = aggregated[aid]
        print(f"  Agent {aid} ({roles[aid]:>14s}): "
              f"final=Rs{a['final_value_mean']:,.0f}(+/-{a['final_value_std']:,.0f})  "
              f"return={a['total_return_mean']*100:6.2f}%(+/-{a['total_return_std']*100:4.2f}%)  "
              f"vol={a['volatility_mean']*100:5.2f}%  maxDD={a['max_drawdown_mean']*100:5.2f}%  "
              f"sharpe(episode-scale)={a['sharpe_mean']:6.2f}")
    print(f"  MAS specialization score (mean pairwise TV distance of basket choice): {specialization:.3f}")
    print(f"  Resource utilisation (mean final/initial total capital): {utilisation:.3f}")


def _run_experiment(name, agent_ids, roles, initial_capitals, agents, num_episodes, train_seed,
                     reward_mode=cfg.REWARD_MODE, enable_market_impact=True):
    """Shared train -> evaluate -> report pipeline for all four experiments."""
    print(f"\n>>> {name}")
    env = FinancialEnv(
        agent_ids=agent_ids, agent_roles=roles, initial_capitals=initial_capitals,
        episode_length=cfg.EPISODE_LENGTH, reward_mode=reward_mode,
        enable_market_impact=enable_market_impact, seed=train_seed,
    )
    trainer = Trainer(env, agents)
    training_df = trainer.train(num_episodes=num_episodes, log_every=max(1, num_episodes // 5), verbose=True)
    _print_training_summary(name, training_df, agent_ids, roles)

    eval_trajectories = evaluate_out_of_sample(
        agent_ids, roles, initial_capitals, agents, eval_seeds=cfg.EVAL_SEEDS,
        episode_length=cfg.EPISODE_LENGTH, reward_mode=reward_mode, enable_market_impact=enable_market_impact,
    )
    aggregated, specialization, utilisation = _aggregate_eval_trajectories(
        eval_trajectories, agent_ids, initial_capitals
    )
    _print_evaluation_summary(name, aggregated, specialization, utilisation, agent_ids, roles, len(cfg.EVAL_SEEDS))

    return {
        "training_df": training_df,
        "eval_trajectories": eval_trajectories,
        "representative_trajectory": eval_trajectories[0],  # first held-out seed, for illustrative plots
        "aggregated_eval": aggregated,
        "specialization": specialization,
        "utilisation": utilisation,
        "agents": agents,
    }


def run_experiment_A_random(num_episodes: int, seed: int = cfg.TRAIN_SEED):
    initial_capitals = {aid: cfg.INITIAL_CAPITAL_PER_AGENT for aid in AGENT_IDS}
    agents = {aid: RandomAgent(aid, AGENT_ROLES[aid], seed=seed + aid) for aid in AGENT_IDS}
    return _run_experiment("Experiment A (Random allocation baseline)", AGENT_IDS, AGENT_ROLES,
                            initial_capitals, agents, num_episodes, seed)


def run_experiment_B_fixed(num_episodes: int, seed: int = cfg.TRAIN_SEED, fixed_action: int = 2):
    initial_capitals = {aid: cfg.INITIAL_CAPITAL_PER_AGENT for aid in AGENT_IDS}
    agents = {aid: FixedAgent(aid, AGENT_ROLES[aid], fixed_action=fixed_action) for aid in AGENT_IDS}
    return _run_experiment("Experiment B (Fixed 'Balanced' basket for all)", AGENT_IDS, AGENT_ROLES,
                            initial_capitals, agents, num_episodes, seed)


def run_experiment_C_single_agent(num_episodes: int, seed: int = cfg.TRAIN_SEED):
    agent_ids = [0]
    roles = {0: "Growth"}
    initial_capitals = {0: cfg.TOTAL_INITIAL_CAPITAL}
    agents = {0: QLearningAgent(0, "Growth", seed=seed)}
    return _run_experiment("Experiment C (Single-agent Q-Learning, full capital)", agent_ids, roles,
                            initial_capitals, agents, num_episodes, seed)


def run_experiment_D_independent_qlearning(num_episodes: int, seed: int = cfg.TRAIN_SEED, reward_mode: str = "individual"):
    initial_capitals = {aid: cfg.INITIAL_CAPITAL_PER_AGENT for aid in AGENT_IDS}
    agents = {aid: QLearningAgent(aid, AGENT_ROLES[aid], seed=seed + aid) for aid in AGENT_IDS}
    return _run_experiment(f"Experiment D (4-agent Independent Q-Learning, reward_mode={reward_mode})",
                            AGENT_IDS, AGENT_ROLES, initial_capitals, agents, num_episodes, seed,
                            reward_mode=reward_mode)


def main(num_episodes: int = cfg.NUM_EPISODES, seed: int = cfg.TRAIN_SEED):
    # Imported lazily: the web API (Vercel) never needs matplotlib, only this CLI run does.
    from visualization import plots
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    os.makedirs(cfg.PLOTS_DIR, exist_ok=True)

    result_a = run_experiment_A_random(num_episodes, seed)
    result_b = run_experiment_B_fixed(num_episodes, seed)
    result_c = run_experiment_C_single_agent(num_episodes, seed)
    result_d = run_experiment_D_independent_qlearning(num_episodes, seed)

    # Save training logs
    result_a["training_df"].to_csv(os.path.join(cfg.RESULTS_DIR, "experiment_A_random.csv"), index=False)
    result_b["training_df"].to_csv(os.path.join(cfg.RESULTS_DIR, "experiment_B_fixed.csv"), index=False)
    result_c["training_df"].to_csv(os.path.join(cfg.RESULTS_DIR, "experiment_C_single_agent.csv"), index=False)
    result_d["training_df"].to_csv(os.path.join(cfg.RESULTS_DIR, "experiment_D_independent_qlearning.csv"), index=False)

    # ---- Plots for Experiment D (the main multi-agent result) ----
    # Trajectory-based plots now use a HELD-OUT evaluation trajectory (first
    # eval seed), not a continuation of the training stream -- so even the
    # illustrative plots reflect genuinely unseen market conditions.
    df_d = result_d["training_df"]
    traj_d = result_d["representative_trajectory"]
    agg_d = result_d["aggregated_eval"]
    initial_capitals_d = {aid: cfg.INITIAL_CAPITAL_PER_AGENT for aid in AGENT_IDS}

    plots.plot_training_rewards(df_d, AGENT_IDS, AGENT_ROLES, os.path.join(cfg.PLOTS_DIR, "1_training_rewards.png"))
    plots.plot_portfolio_value(traj_d, AGENT_IDS, AGENT_ROLES, os.path.join(cfg.PLOTS_DIR, "2_portfolio_value.png"))
    for aid in AGENT_IDS:
        plots.plot_allocation_over_time(traj_d, aid, AGENT_ROLES[aid], os.path.join(cfg.PLOTS_DIR, f"3_allocation_agent{aid}.png"))
    final_caps_d_avg = {aid: agg_d[aid]["final_value_mean"] for aid in AGENT_IDS}
    plots.plot_agent_comparison(final_caps_d_avg, initial_capitals_d, AGENT_ROLES, os.path.join(cfg.PLOTS_DIR, "4_agent_comparison.png"))
    plots.plot_regime_vs_allocation(traj_d, AGENT_IDS, AGENT_ROLES, os.path.join(cfg.PLOTS_DIR, "5_regime_vs_allocation.png"))
    plots.plot_drawdown(traj_d, AGENT_IDS, AGENT_ROLES, os.path.join(cfg.PLOTS_DIR, "6_drawdown.png"))
    plots.plot_risk_vs_return(df_d, AGENT_IDS, AGENT_ROLES, os.path.join(cfg.PLOTS_DIR, "7_risk_vs_return.png"))

    # ---- Cross-experiment comparison (A vs B vs C vs D total capital growth) ----
    _plot_cross_experiment_comparison(result_a["training_df"], result_b["training_df"],
                                       result_c["training_df"], result_d["training_df"])

    print("\nAll experiments complete. CSV logs + plots saved under 'results/'.")
    return {"A": result_a, "B": result_b, "C": result_c, "D": result_d}


def _plot_cross_experiment_comparison(df_a, df_b, df_c, df_d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    window = max(1, len(df_a) // 50)
    ax.plot(df_a["episode"], df_a["total_capital"].rolling(window, min_periods=1).mean(), label="A: Random")
    ax.plot(df_b["episode"], df_b["total_capital"].rolling(window, min_periods=1).mean(), label="B: Fixed")
    ax.plot(df_c["episode"], (df_c["total_capital"].rolling(window, min_periods=1).mean()), label="C: Single-Agent Q-Learning")
    ax.plot(df_d["episode"], df_d["total_capital"].rolling(window, min_periods=1).mean(), label="D: 4-Agent Independent Q-Learning")
    ax.axhline(cfg.TOTAL_INITIAL_CAPITAL, color="gray", linestyle="--", linewidth=1, label="Initial capital")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Capital at Episode End (Rs)")
    ax.set_title("Experiment Comparison: A vs B vs C vs D (training curves)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(cfg.PLOTS_DIR, "8_experiment_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
