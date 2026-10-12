"""
server/trainstate.py - serialisable Q-learning state, so training can continue across stateless requests.

Vercel functions keep no memory between requests. The browser therefore holds the learner's complete state
as one opaque string ("blob") and sends it back with every call that needs it:

    blob = base64( zlib( JSON{ Q-tables, visit counts, epsilons, hyper-parameters,
                              every agent's RNG state, the training environment's RNG state } ) )

Because the RNG states are included, training N episodes in one go and training them in several chunks
(blob -> train -> blob -> ...) produce bit-identical results (tests/test_trainstate.py proves it).
The blob is JSON only (never pickle) and is fully validated on the way in: it comes from the network.
"""

import base64
import json
import time
import zlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import config as cfg
from agents.q_learning_agent import QLearningAgent
from dashboard import reference
from experiments.engine import QHyperParams, build_q_agents, layout_for, make_env
from training.trainer import Trainer

BLOB_VERSION = 1
MAX_BLOB_CHARS = 1_500_000          # encoded size accepted from the network
MAX_DECODED_BYTES = 6_000_000       # decompressed size cap (zip-bomb guard)
LAYOUTS = ("C", "D")
LOG_DROP_PREFIXES = ("drawdown_",)


class BlobError(ValueError):
    """The submitted state blob is malformed or out of range."""


@dataclass
class TrainState:
    layout: str                      # "D" = four agents (RL Lab, Simulation, Analytics, Exp. D); "C" = one agent
    seed: int
    episode_length: int
    impact: bool
    reward_mode: str
    total_episodes: int
    env_rng: Optional[dict]          # None until the first training episode has run
    agents: Dict[int, QLearningAgent]
    hp: dict
    label: str = ""

    @property
    def agent_ids(self):
        return list(layout_for(self.layout)[0])

    @property
    def roles(self):
        return layout_for(self.layout)[1]


# ------------------------------------------------------------------ construction
def new_state(layout: str, hp: QHyperParams, seed: int, episode_length: int, impact: bool,
              reward_mode: str = cfg.REWARD_MODE, label: str = "") -> TrainState:
    ids, roles, _ = layout_for(layout)
    return TrainState(layout=layout, seed=int(seed), episode_length=int(episode_length), impact=bool(impact),
                      reward_mode=reward_mode, total_episodes=0, env_rng=None,
                      agents=build_q_agents(ids, roles, hp, int(seed)), hp=_hp_dict(hp), label=label)


def state_from_agents(agents: Dict[int, QLearningAgent], seed: int, episode_length: int, impact: bool,
                      hp: QHyperParams, reward_mode: str = cfg.REWARD_MODE, label: str = "") -> TrainState:
    """Continue training EXISTING agents (e.g. the reference ones) on a new market stream seeded by `seed`."""
    return TrainState(layout="D", seed=int(seed), episode_length=int(episode_length), impact=bool(impact),
                      reward_mode=reward_mode, total_episodes=0, env_rng=None, agents=agents, hp=_hp_dict(hp),
                      label=label)


def _hp_dict(hp: QHyperParams) -> dict:
    return {"alpha": hp.alpha, "gamma": hp.gamma, "epsilon_initial": hp.epsilon_initial,
            "epsilon_min": hp.epsilon_min, "epsilon_decay": hp.epsilon_decay, "q_init": hp.q_init}


# ------------------------------------------------------------------ encode / decode
def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _arr(s, dtype, shape, what: str) -> np.ndarray:
    try:
        raw = base64.b64decode(s, validate=True)
    except Exception as e:                                   # noqa: BLE001
        raise BlobError(f"{what}: not valid base64") from e
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if len(raw) != expected:
        raise BlobError(f"{what}: wrong size")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def encode(ts: TrainState) -> str:
    payload = {
        "v": BLOB_VERSION, "layout": ts.layout, "seed": ts.seed, "episode_length": ts.episode_length,
        "impact": ts.impact, "reward_mode": ts.reward_mode, "total_episodes": ts.total_episodes,
        "env_rng": ts.env_rng, "hp": ts.hp, "label": ts.label,
        "agents": {str(aid): {
            "q": _b64(ag.q_table.astype(np.float64)), "visits": _b64(ag.visit_counts.astype(np.int64)),
            "epsilon": float(ag.epsilon), "n_updates": int(ag.n_updates),
            "alpha": float(ag.alpha), "gamma": float(ag.gamma),
            "epsilon_min": float(ag.epsilon_min), "epsilon_decay": float(ag.epsilon_decay),
            "rng": ag.rng.bit_generator.state,
        } for aid, ag in ts.agents.items()},
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(zlib.compress(raw, 6)).decode("ascii")


def _num(x, name, lo=None, hi=None, integer=False):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x):
        raise BlobError(f"{name}: must be a finite number")
    if integer and int(x) != x:
        raise BlobError(f"{name}: must be an integer")
    if (lo is not None and x < lo) or (hi is not None and x > hi):
        raise BlobError(f"{name}: out of range")
    return int(x) if integer else float(x)


def decode(blob: str) -> TrainState:
    if not isinstance(blob, str) or not blob or len(blob) > MAX_BLOB_CHARS:
        raise BlobError("state blob missing or too large")
    try:
        d = zlib.decompressobj()
        raw = d.decompress(base64.b64decode(blob, validate=True), MAX_DECODED_BYTES)
        if d.unconsumed_tail:
            raise BlobError("state blob too large when decompressed")
        p = json.loads(raw)
    except BlobError:
        raise
    except Exception as e:                                   # noqa: BLE001
        raise BlobError("state blob is corrupt") from e
    if not isinstance(p, dict) or p.get("v") != BLOB_VERSION or p.get("layout") not in LAYOUTS:
        raise BlobError("unsupported state blob")
    layout = p["layout"]
    ids, roles, _ = layout_for(layout)
    ag_json = p.get("agents")
    if not isinstance(ag_json, dict) or sorted(ag_json) != sorted(str(a) for a in ids):
        raise BlobError("state blob has the wrong agents")
    agents = {}
    for aid in ids:
        a = ag_json[str(aid)]
        try:
            ag = QLearningAgent(aid, roles[aid])
            ag.q_table = _arr(a["q"], np.float64, (cfg.N_STATES, cfg.N_ACTIONS), "q-table")
            if not np.all(np.isfinite(ag.q_table)):
                raise BlobError("q-table contains non-finite numbers")
            ag.visit_counts = _arr(a["visits"], np.int64, (cfg.N_STATES, cfg.N_ACTIONS), "visit counts")
            if ag.visit_counts.min() < 0:
                raise BlobError("visit counts negative")
            ag.epsilon = _num(a["epsilon"], "epsilon", 0, 1)
            ag.n_updates = _num(a["n_updates"], "n_updates", 0, 10 ** 12, integer=True)
            ag.alpha = _num(a["alpha"], "alpha", 1e-9, 1)
            ag.gamma = _num(a["gamma"], "gamma", 0, 1)
            ag.epsilon_min = _num(a["epsilon_min"], "epsilon_min", 0, 1)
            ag.epsilon_decay = _num(a["epsilon_decay"], "epsilon_decay", 1e-9, 1)
            rng = np.random.default_rng()
            rng.bit_generator.state = a["rng"]
            ag.rng = rng
        except BlobError:
            raise
        except Exception as e:                               # noqa: BLE001
            raise BlobError(f"agent {aid} state is invalid") from e
        agents[aid] = ag
    env_rng = p.get("env_rng")
    if env_rng is not None:
        try:
            np.random.default_rng().bit_generator.state = env_rng      # validates shape/keys
        except Exception as e:                               # noqa: BLE001
            raise BlobError("environment RNG state is invalid") from e
    try:
        label = str(p.get("label", ""))[:200]
        hp = {k: float(v) for k, v in dict(p["hp"]).items()}
        return TrainState(layout=layout, seed=_num(p["seed"], "seed", 0, 2 ** 31, True),
                          episode_length=_num(p["episode_length"], "episode_length", 1, 1000, True),
                          impact=bool(p["impact"]), reward_mode=str(p["reward_mode"]),
                          total_episodes=_num(p["total_episodes"], "total_episodes", 0, 10 ** 7, True),
                          env_rng=env_rng, agents=agents, hp=hp, label=label)
    except BlobError:
        raise
    except Exception as e:                                   # noqa: BLE001
        raise BlobError("state blob is incomplete") from e


# ------------------------------------------------------------------ agents for read-only endpoints
def load_agents(blob: Optional[str]) -> Tuple[Dict[int, QLearningAgent], str]:
    """(four agents, label). blob=None -> the precomputed reference agents."""
    if blob is None:
        return reference.make_reference_agents(), reference.reference_label()
    ts = decode(blob)
    if ts.layout != "D":
        raise BlobError("this endpoint needs the four-agent state")
    return ts.agents, ts.label or "Agents trained in the RL Lab"


# ------------------------------------------------------------------ training
def train_chunk(ts: TrainState, n_episodes: int, max_seconds: Optional[float] = None) -> Tuple[pd.DataFrame, int]:
    """Train up to n_episodes more episodes IN PLACE. Returns (rows, episodes_done).

    The time guard only decides where to stop (at an episode boundary); it cannot change any number."""
    if n_episodes < 1:
        raise ValueError("n_episodes must be >= 1")
    ids, roles, caps = layout_for(ts.layout)
    env = make_env(ids, roles, caps, ts.seed, ts.episode_length, ts.reward_mode, ts.impact)
    if ts.env_rng is not None:
        env.rng.bit_generator.state = ts.env_rng             # market + env share this one Generator
    trainer = Trainer(env, ts.agents)
    t0, done, frames = time.time(), 0, []
    step = 5
    while done < n_episodes:
        k = min(step, n_episodes - done)
        frames.append(trainer.train(k, verbose=False, episode_offset=ts.total_episodes + done))
        done += k
        if max_seconds is not None and time.time() - t0 > max_seconds:
            break
    ts.total_episodes += done
    ts.env_rng = env.rng.bit_generator.state
    return pd.concat(frames, ignore_index=True), done


def apply_hyperparams(ts: TrainState, hp: QHyperParams):
    for ag in ts.agents.values():
        ag.alpha, ag.gamma = hp.alpha, hp.gamma
        ag.epsilon_min, ag.epsilon_decay = hp.epsilon_min, hp.epsilon_decay
        ag.epsilon = max(hp.epsilon_min, min(ag.epsilon, 1.0))
    ts.hp = _hp_dict(hp)


# ------------------------------------------------------------------ training-log (de)serialisation
def df_to_log(df: pd.DataFrame) -> Dict[str, list]:
    out = {}
    for c in df.columns:
        if c.startswith(LOG_DROP_PREFIXES):
            continue
        if c == "episode":
            out[c] = [int(v) for v in df[c]]
        elif c.startswith("capital_") or c == "total_capital":
            out[c] = [round(float(v), 2) for v in df[c]]
        else:
            out[c] = [round(float(v), 6) for v in df[c]]
    return out


def log_to_df(log: Optional[Dict[str, list]], max_rows: int = 25000) -> Optional[pd.DataFrame]:
    if not log:
        return None
    lengths = {len(v) for v in log.values()}
    if len(lengths) != 1 or "episode" not in log:
        raise BlobError("training log columns must have equal length and include 'episode'")
    n = lengths.pop()
    if n == 0 or n > max_rows or len(log) > 64:
        raise BlobError("training log has an invalid size")
    try:
        df = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in log.items()})
    except Exception as e:                                   # noqa: BLE001
        raise BlobError("training log must be numeric") from e
    df["episode"] = df["episode"].astype(int)
    return df
