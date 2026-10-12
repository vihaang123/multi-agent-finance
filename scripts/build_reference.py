"""
scripts/build_reference.py

Regenerates data/reference_results.json and data/reference_qtables.npz by
ACTUALLY RUNNING the four experiments (A Random, B Fixed, C Single-agent Q,
D 4-agent Independent Q) with the default Baseline V1 configuration:

    5000 training episodes (seed 42), 50 out-of-sample evaluation episodes
    (seeds 51-100), epsilon=0 and frozen Q-tables at evaluation.

Nothing in the output is hand-written: every number comes from
experiments.engine.run_experiment. The dashboard loads these files so it
opens with real results immediately; the Experiments page can re-run any
experiment live and compare against them.

All data is from the SYNTHETIC market simulator, not real market data.

Usage:  python -m scripts.build_reference            # full 5000-episode run
        python -m scripts.build_reference --episodes 500   # quick check (do not commit)
"""

import argparse
import datetime
import json
import os
import platform
import time

import numpy as np

import config as cfg
from experiments.engine import RunConfig, run_experiment

OUT_DIR = "data"


def _curve(df, col, window=50, every=25):
    s = df[col].rolling(window, min_periods=1).mean()
    idx = list(range(0, len(s), every))
    if idx[-1] != len(s) - 1:
        idx.append(len(s) - 1)
    return {"episode": [int(df["episode"].iloc[i]) for i in idx], "value": [float(s.iloc[i]) for i in idx]}


def main(episodes: int, out_dir: str = OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "meta": {
            "title": "Reference results from a real run of experiments.engine.run_experiment",
            "data_type": "SYNTHETIC market simulator - not real market data",
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "train_episodes": episodes, "train_seed": cfg.TRAIN_SEED,
            "eval_seeds": [cfg.EVAL_SEED_START, cfg.EVAL_SEED_START + cfg.N_EVAL_EPISODES - 1],
            "episode_length": cfg.EPISODE_LENGTH,
            "python": platform.python_version(), "numpy": np.__version__,
            "script": "python -m scripts.build_reference",
        },
        "experiments": {},
    }
    qtables = {}
    for exp in "ABCD":
        t0 = time.time()
        rc = RunConfig(experiment=exp, n_train_episodes=episodes)
        print(f"running experiment {exp} ...", flush=True)
        r = run_experiment(rc)
        entry = {"config": rc.to_dict(), "summary": r["summary"], "wall_seconds": round(time.time() - t0, 1),
                 "roles": {str(k): v for k, v in r["roles"].items()}}
        df = r["training_df"]
        if df is not None:
            entry["training"] = {
                "reward": {str(a): _curve(df, f"reward_{a}") for a in r["agent_ids"]},
                "epsilon": {str(a): _curve(df, f"epsilon_{a}", window=1) for a in r["agent_ids"]},
                "td": {str(a): _curve(df, f"td_{a}") for a in r["agent_ids"]},
                "capital": {str(a): _curve(df, f"capital_{a}") for a in r["agent_ids"]},
                "total_capital": _curve(df, "total_capital"),
            }
            for a in r["agent_ids"]:
                ag = r["agents"][a]
                qtables[f"{exp}_q_{a}"] = ag.q_table
                qtables[f"{exp}_visits_{a}"] = ag.visit_counts
                qtables[f"{exp}_epsilon_{a}"] = np.array(ag.epsilon)
        result["experiments"][exp] = entry
        c = r["summary"]["combined"]
        print(f"  {exp}: OOS return {c['total_return_mean']*100:.2f}% +/- {c['total_return_std']*100:.2f}% "
              f"({entry['wall_seconds']}s)", flush=True)

    with open(os.path.join(out_dir, "reference_results.json"), "w") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(os.path.join(out_dir, "reference_qtables.npz"), **qtables)
    print("saved", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=cfg.NUM_EPISODES)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    main(a.episodes, a.out)
