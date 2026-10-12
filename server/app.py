"""
server/app.py - the stateless JSON API (FastAPI) behind the Next.js front end.

Deployed on Vercel as a Python serverless function (api/index.py imports `app` from here). Serverless
functions keep no memory between requests, so every request carries what it needs:
  * simulation   : seed, market-impact flag, policies and the list of steps already taken (replayed server-side)
  * trained agents: an opaque `state` string (server/trainstate.py); null = the precomputed reference agents
  * training log : plain columns of numbers kept by the browser

All endpoints are POST /api/... (plus GET /api/meta, /api/health). All market data is SYNTHETIC.
"""

import time
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import config as cfg
from dashboard import reference, theme
from dashboard.session import POLICY_LABELS, SimulationSession
from environment.state_utils import BUCKET_LABELS
from evaluation import metrics
from experiments.engine import (EXPERIMENT_NAMES, LEARNING_EXPERIMENTS, QHyperParams, RunConfig, evaluate_agents,
                                layout_for, run_experiment, summarize_evaluation, training_curves)
from experiments.run_experiments import AGENT_IDS, AGENT_ROLES, evaluate_out_of_sample
from server import blocks as B
from server import pages, trainstate

MAX_CHUNK_EPISODES = 200       # episodes trained per request
CHUNK_SECONDS = 20.0           # stop a training request after this long (at an episode boundary)
MAX_LAB_EPISODES = 3000        # total episodes one "Train" press may ask for (enforced by the UI + checked here)
MAX_UI_EPISODES = 3000

app = FastAPI(title="Multi-agent financial allocation API (synthetic data)", docs_url="/api/docs",
              openapi_url="/api/openapi.json", redoc_url=None)


# --------------------------------------------------------------------------- errors / responses
class ApiError(Exception):
    def __init__(self, message, status=400):
        self.message, self.status = message, status


@app.exception_handler(ApiError)
async def _api_error(_: Request, exc: ApiError):
    return JSONResponse({"detail": exc.message}, status_code=exc.status)


@app.exception_handler(trainstate.BlobError)
async def _blob_error(_: Request, exc: trainstate.BlobError):
    return JSONResponse({"detail": f"Saved agent state is invalid: {exc}. Restore the reference agents and try again."}, status_code=400)


@app.exception_handler(ValueError)
async def _value_error(_: Request, exc: ValueError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError):
    msgs = []
    for e in exc.errors()[:5]:
        loc = ".".join(str(x) for x in e.get("loc", []) if x != "body")
        msgs.append(f"{loc}: {e.get('msg')}")
    return JSONResponse({"detail": "Invalid request - " + "; ".join(msgs)}, status_code=422)


def ok(payload: dict) -> JSONResponse:
    return JSONResponse(B.sanitize(payload), headers={"Cache-Control": "no-store"})


# --------------------------------------------------------------------------- shared models
class HP(BaseModel):
    alpha: float = Field(cfg.ALPHA_LR, gt=0, le=1)
    gamma: float = Field(cfg.GAMMA, ge=0, le=1)
    epsilon_initial: float = Field(cfg.EPSILON_INITIAL, ge=0, le=1)
    epsilon_min: float = Field(cfg.EPSILON_MIN, ge=0, le=1)
    epsilon_decay: float = Field(cfg.EPSILON_DECAY, gt=0, le=1)
    q_init: float = Field(cfg.Q_INIT_VALUE, ge=-100, le=100)

    def to_params(self) -> QHyperParams:
        hp = QHyperParams(alpha=self.alpha, gamma=self.gamma, epsilon_initial=self.epsilon_initial,
                          epsilon_min=self.epsilon_min, epsilon_decay=self.epsilon_decay, q_init=self.q_init)
        errs = hp.validate()
        if errs:
            raise ApiError(" ".join(errs))
        return hp


Seed = Field(0, ge=0, le=1_000_000)
AgentsState = Optional[str]


class AgentsReq(BaseModel):
    agents: AgentsState = None


# --------------------------------------------------------------------------- meta / health
@app.get("/api/health")
def health():
    return ok({"status": "ok", "synthetic_data": True})


@app.get("/api/meta")
def meta():
    ref = reference.load_reference_results()
    sigma = [[cfg.ASSET_RETURN_PARAMS[r][a][1] for a in range(cfg.N_ASSETS)] for r in range(cfg.N_REGIMES)]
    return ok({
        "synthetic_text": theme.SYNTHETIC_TEXT,
        "agent_ids": AGENT_IDS, "roles": {str(a): AGENT_ROLES[a] for a in AGENT_IDS},
        "agent_colors": theme.AGENT_COLORS, "regime_colors": theme.REGIME_COLORS, "asset_colors": theme.ASSET_COLORS,
        "regimes": list(cfg.REGIME_NAMES), "assets": list(cfg.ASSET_NAMES),
        "baskets": [{"name": cfg.ACTION_BASKET_NAMES[i], "weights": [float(w) for w in cfg.ACTION_BASKETS[i]]}
                    for i in range(cfg.N_ACTIONS)],
        "bucket_labels": BUCKET_LABELS, "policy_labels": POLICY_LABELS,
        "experiment_names": EXPERIMENT_NAMES, "learning_experiments": list(LEARNING_EXPERIMENTS),
        "reward_weights": cfg.REWARD_WEIGHTS, "reward_scale": cfg.REWARD_SCALE, "reward_clip": cfg.REWARD_CLIP,
        "asset_sigma": sigma, "cash_index": cfg.CASH, "tx_cost_rate": cfg.TRANSACTION_COST_RATE,
        "episode_length": cfg.EPISODE_LENGTH, "initial_capital_per_agent": cfg.INITIAL_CAPITAL_PER_AGENT,
        "total_initial_capital": cfg.TOTAL_INITIAL_CAPITAL,
        "train_seed": cfg.TRAIN_SEED, "eval_seed_start": cfg.EVAL_SEED_START, "n_eval_episodes": cfg.N_EVAL_EPISODES,
        "hp_defaults": {"alpha": cfg.ALPHA_LR, "gamma": cfg.GAMMA, "epsilon_initial": cfg.EPSILON_INITIAL,
                        "epsilon_min": cfg.EPSILON_MIN, "epsilon_decay": cfg.EPSILON_DECAY, "q_init": cfg.Q_INIT_VALUE},
        "limits": {"lab_max_episodes": MAX_LAB_EPISODES, "ui_max_episodes": MAX_UI_EPISODES, "chunk_episodes": MAX_CHUNK_EPISODES},
        "reference": {"available": bool(ref), "label": reference.reference_label(), "meta": ref["meta"] if ref else None},
        "sharpe_note": metrics.SHARPE_METHODOLOGY_NOTE,
    })


# --------------------------------------------------------------------------- simulation / dashboard / analytics (episode)
class PolicyIn(BaseModel):
    kind: Literal["trained", "random", "fixed"] = "trained"
    basket: int = Field(2, ge=0, lt=cfg.N_ACTIONS)


class StepIn(BaseModel):
    a: List[int] = Field(..., min_length=4, max_length=4)
    k: str = Field(..., pattern="^[trf]{4}$")


class SimReq(BaseModel):
    seed: int = Field(cfg.EVAL_SEED_START, ge=0, le=1_000_000)
    impact: bool = True
    policies: List[PolicyIn] = Field(..., min_length=4, max_length=4)
    steps: List[StepIn] = Field(default_factory=list, max_length=cfg.EPISODE_LENGTH)
    advance: int = Field(0, ge=0, le=cfg.EPISODE_LENGTH)
    to_end: bool = False
    view: Literal["dashboard", "overview", "decisions", "money", "log", "analytics", "none"] = "overview"
    vol_window: int = Field(10, ge=3, le=30)
    agents: AgentsState = None


@app.post("/api/simulate")
def simulate(req: SimReq):
    agents, label = trainstate.load_agents(req.agents)
    sim = SimulationSession(agents=agents, seed=req.seed, episode_length=cfg.EPISODE_LENGTH, enable_market_impact=req.impact)
    for aid, pol in zip(sim.agent_ids, req.policies):
        sim.set_policy(aid, pol.kind, pol.basket)
    sim.replay([s.model_dump() for s in req.steps])
    if req.to_end:
        sim.run_to_end()
    elif req.advance:
        sim.run_steps(req.advance)
    if req.view == "dashboard":
        blks = pages.dashboard_blocks(sim)
    elif req.view == "analytics":
        blks = pages.episode_analytics(sim.history, sim.agent_ids, sim.roles, sim.capitals, req.vol_window)
    elif req.view == "none":
        blks = []
    else:
        blks = pages.SIM_VIEWS[req.view](sim)
    return ok({
        "t": sim.t, "done": sim.done, "episode_length": sim.episode_length, "steps": sim.steps, "seed": sim.seed,
        "summary_line": sim.summary_line(), "regime_now": cfg.REGIME_NAMES[sim.current_regime],
        "regime_color": theme.REGIME_COLORS[cfg.REGIME_NAMES[sim.current_regime]],
        "total_value": float(sim.capital_series()[-1]), "total_value_text": theme.inr(float(sim.capital_series()[-1])),
        "agents_label": label, "blocks": blks,
    })


# --------------------------------------------------------------------------- RL Lab
class LabTrainReq(BaseModel):
    mode: Literal["fresh", "continue"] = "fresh"
    state: AgentsState = None
    hp: HP = HP()
    seed: int = Field(cfg.TRAIN_SEED, ge=0, le=1_000_000)
    impact: bool = True
    episodes: int = Field(..., ge=1, le=MAX_CHUNK_EPISODES)


@app.post("/api/lab/train")
def lab_train(req: LabTrainReq):
    hp = req.hp.to_params()
    if req.state is None:
        if req.seed in cfg.EVAL_SEEDS:
            raise ApiError(f"Training seed {req.seed} lies inside the held-out evaluation block {cfg.EVAL_SEED_START}-"
                           f"{cfg.EVAL_SEED_START + cfg.N_EVAL_EPISODES - 1}; pick another so evaluation stays out-of-sample.")
        if req.mode == "fresh":
            ts = trainstate.new_state("D", hp, req.seed, cfg.EPISODE_LENGTH, req.impact)
        else:                                   # continue from the precomputed reference agents
            ts = trainstate.state_from_agents(reference.make_reference_agents(), req.seed, cfg.EPISODE_LENGTH, req.impact, hp)
            trainstate.apply_hyperparams(ts, hp)
    else:
        ts = trainstate.decode(req.state)
        if ts.layout != "D":
            raise ApiError("This state belongs to another experiment layout.")
        trainstate.apply_hyperparams(ts, hp)
    t0 = time.time()
    df, done = trainstate.train_chunk(ts, req.episodes, max_seconds=CHUNK_SECONDS)
    ts.label = (f"Trained in the RL Lab: {ts.total_episodes:,} episodes, seed {ts.seed}"
                + (", market impact on" if ts.impact else ", market impact off"))
    return ok({"state": trainstate.encode(ts), "rows": trainstate.df_to_log(df), "episodes_done": done,
               "total_episodes": ts.total_episodes, "label": ts.label, "seconds": round(time.time() - t0, 2)})


class LabCurvesReq(AgentsReq):
    log: Optional[Dict[str, List[float]]] = None
    smooth: int = Field(50, ge=1, le=500)


@app.post("/api/lab/curves")
def lab_curves(req: LabCurvesReq):
    agents, _ = trainstate.load_agents(req.agents)
    log = trainstate.log_to_df(req.log) if req.agents is not None else None
    if log is not None:
        for aid in AGENT_IDS:
            for col in (f"reward_{aid}", f"epsilon_{aid}", f"td_{aid}", f"capital_{aid}"):
                if col not in log.columns:
                    raise ApiError(f"training log is missing column {col}")
    return ok({"blocks": pages.lab_curves(log, agents, req.smooth)})


class LabInspectReq(AgentsReq):
    agent: int = Field(0, ge=0, le=3)
    regime: int = Field(0, ge=0, lt=cfg.N_REGIMES)
    eq: int = Field(0, ge=0, le=2)
    bond: int = Field(0, ge=0, le=2)
    gold: int = Field(0, ge=0, le=2)


@app.post("/api/lab/inspect")
def lab_inspect(req: LabInspectReq):
    agents, _ = trainstate.load_agents(req.agents)
    return ok({"blocks": pages.lab_inspect(agents, req.agent, req.regime, req.eq, req.bond, req.gold)})


class LabPolicyReq(AgentsReq):
    agent: int = Field(0, ge=0, le=3)
    mode: Literal["greedy", "maxq", "visits"] = "greedy"


@app.post("/api/lab/policy")
def lab_policy(req: LabPolicyReq):
    agents, _ = trainstate.load_agents(req.agents)
    return ok({"blocks": pages.lab_policy(agents, req.agent, req.mode)})


class LabWalkReq(AgentsReq):
    agent: int = Field(0, ge=0, le=3)
    steps: int = Field(5, ge=1, le=20)
    seed: int = Field(7, ge=0, le=1_000_000)


@app.post("/api/lab/walkthrough")
def lab_walkthrough(req: LabWalkReq):
    agents, _ = trainstate.load_agents(req.agents)
    return ok({"blocks": pages.lab_walkthrough(agents, req.agent, req.seed, req.steps)})


class EvalReq(AgentsReq):
    start: int = Field(cfg.EVAL_SEED_START, ge=0, le=1_000_000)
    n: int = Field(cfg.N_EVAL_EPISODES, ge=5, le=200)
    impact: bool = True


def _evaluate(req: EvalReq):
    agents, label = trainstate.load_agents(req.agents)
    seeds = list(range(req.start, req.start + req.n))
    res = evaluate_agents(agents, AGENT_IDS, AGENT_ROLES, pages.CAPS, seeds, enable_market_impact=req.impact)
    return res, seeds, label


@app.post("/api/lab/evaluate")
def lab_evaluate(req: EvalReq):
    res, seeds, label = _evaluate(req)
    return ok({"blocks": pages.lab_evaluate(res, seeds, label), "specialization": res["summary"]["specialization"]})


# --------------------------------------------------------------------------- Portfolio analytics (out-of-sample)
@app.post("/api/analytics/oos")
def analytics_oos(req: EvalReq):
    res, seeds, label = _evaluate(req)
    return ok({"blocks": pages.oos_blocks(res, seeds, label), "seeds": seeds})


class OosEpisodeReq(AgentsReq):
    seed: int = Field(cfg.EVAL_SEED_START, ge=0, le=1_000_000)
    impact: bool = True
    vol_window: int = Field(10, ge=3, le=30)


@app.post("/api/analytics/oos-episode")
def analytics_oos_episode(req: OosEpisodeReq):
    agents, _ = trainstate.load_agents(req.agents)
    traj = evaluate_out_of_sample(AGENT_IDS, AGENT_ROLES, pages.CAPS, agents, eval_seeds=[req.seed],
                                  enable_market_impact=req.impact)[0]
    return ok({"blocks": pages.episode_analytics(traj, AGENT_IDS, AGENT_ROLES, pages.CAPS, req.vol_window)})


# --------------------------------------------------------------------------- Experiments
class ExpConfig(BaseModel):
    n_train_episodes: int = Field(1000, ge=10, le=MAX_UI_EPISODES)
    train_seed: int = Field(cfg.TRAIN_SEED, ge=0, le=1_000_000)
    eval_seed_start: int = Field(cfg.EVAL_SEED_START, ge=0, le=1_000_000)
    n_eval_episodes: int = Field(cfg.N_EVAL_EPISODES, ge=5, le=200)
    episode_length: int = Field(cfg.EPISODE_LENGTH, ge=20, le=300)
    fixed_action: int = Field(2, ge=0, lt=cfg.N_ACTIONS)
    enable_market_impact: bool = True
    alpha: float = Field(cfg.ALPHA_LR, gt=0, le=1)
    gamma: float = Field(cfg.GAMMA, ge=0, le=1)
    epsilon_decay: float = Field(cfg.EPSILON_DECAY, gt=0, le=1)

    def run_config(self, experiment: str) -> RunConfig:
        rc = RunConfig(experiment=experiment, n_train_episodes=self.n_train_episodes, train_seed=self.train_seed,
                       eval_seed_start=self.eval_seed_start, n_eval_episodes=self.n_eval_episodes,
                       episode_length=self.episode_length, fixed_action=self.fixed_action,
                       enable_market_impact=self.enable_market_impact,
                       hp=QHyperParams(alpha=self.alpha, gamma=self.gamma, epsilon_decay=self.epsilon_decay))
        errs = rc.validate()
        if errs:
            raise ApiError(f"{pages.SHORT[experiment]}: " + " ".join(errs))
        return rc


class ExpTrainReq(BaseModel):
    experiment: Literal["C", "D"]
    config: ExpConfig = ExpConfig()
    state: AgentsState = None
    episodes: int = Field(..., ge=1, le=MAX_CHUNK_EPISODES)


@app.post("/api/experiments/train")
def experiments_train(req: ExpTrainReq):
    rc = req.config.run_config(req.experiment)
    if req.state is None:
        ts = trainstate.new_state(req.experiment, rc.hp, rc.train_seed, rc.episode_length, rc.enable_market_impact, rc.reward_mode)
    else:
        ts = trainstate.decode(req.state)
        if ts.layout != req.experiment:
            raise ApiError("State belongs to a different experiment.")
    if ts.total_episodes + req.episodes > rc.n_train_episodes:
        raise ApiError("Requested more training episodes than the configuration asks for.")
    df, done = trainstate.train_chunk(ts, req.episodes, max_seconds=CHUNK_SECONDS)
    return ok({"state": trainstate.encode(ts), "rows": trainstate.df_to_log(df), "episodes_done": done,
               "total_episodes": ts.total_episodes})


class ExpEvalReq(BaseModel):
    experiment: Literal["C", "D"]
    config: ExpConfig = ExpConfig()
    state: str
    log: Dict[str, List[float]]


def _entry(rc: RunConfig, summary: dict, roles: dict, training: Optional[dict]) -> dict:
    e = {"config": rc.to_dict(), "summary": summary, "roles": {str(k): v for k, v in roles.items()}}
    if training is not None:
        e["training"] = training
    return e


@app.post("/api/experiments/evaluate")
def experiments_evaluate(req: ExpEvalReq):
    rc = req.config.run_config(req.experiment)
    ts = trainstate.decode(req.state)
    if ts.layout != req.experiment:
        raise ApiError("State belongs to a different experiment.")
    ids, roles, caps = layout_for(req.experiment)
    df = trainstate.log_to_df(req.log)
    trajs = evaluate_out_of_sample(ids, roles, caps, ts.agents, eval_seeds=rc.eval_seeds, episode_length=rc.episode_length,
                                   reward_mode=rc.reward_mode, enable_market_impact=rc.enable_market_impact)
    summary = summarize_evaluation(trajs, ids, caps)
    return ok({"entry": _entry(rc, summary, roles, training_curves(df, ids))})


class ExpBaselineReq(BaseModel):
    experiment: Literal["A", "B"]
    config: ExpConfig = ExpConfig()


@app.post("/api/experiments/baseline")
def experiments_baseline(req: ExpBaselineReq):
    rc = req.config.run_config(req.experiment)
    r = run_experiment(rc)
    return ok({"entry": _entry(rc, r["summary"], r["roles"], None)})


class ExpReportReq(BaseModel):
    entries: Dict[str, Dict[str, Any]]
    meta_line: str = Field("", max_length=400)


@app.post("/api/experiments/report")
def experiments_report(req: ExpReportReq):
    entries = {}
    for e, v in req.entries.items():
        if e not in EXPERIMENT_NAMES or not all(k in v for k in ("config", "summary", "roles")):
            raise ApiError("Malformed experiment results.")
        entries[e] = v
    try:
        blks = pages.experiments_report(entries, "live", req.meta_line)
    except (KeyError, TypeError) as exc:
        raise ApiError(f"Malformed experiment results ({exc}).")
    return ok({"blocks": blks})


@app.post("/api/experiments/reference")
def experiments_reference():
    ref = reference.load_reference_results()
    if not ref:
        return ok({"available": False, "blocks": [B.callout(
            "warn", "Reference files (data/reference_results.json) were not found. Run `python -m scripts.build_reference` to create them.")]})
    m = ref["meta"]
    note = B.callout("note", f"**Precomputed, not live.** Produced by running `{m['script']}` ({m['train_episodes']:,} training episodes on "
                             f"seed {m['train_seed']}, evaluation on seeds {m['eval_seeds'][0]}-{m['eval_seeds'][1]}, generated "
                             f"{m['generated_at_utc']}). It is a cached output of the same code the live run uses; rerun that script to "
                             "regenerate it. Synthetic data.")
    blks = [note] + pages.experiments_report(ref["experiments"], "ref",
                                             f"Reference run | {m['train_episodes']:,} training episodes | seeds "
                                             f"{m['eval_seeds'][0]}-{m['eval_seeds'][1]}")
    return ok({"available": True, "blocks": blks})


# --------------------------------------------------------------------------- Learn
@app.post("/api/learn/static")
def learn_static():
    return ok(pages.learn_static())


class RegimePathReq(BaseModel):
    seed: int = Field(1, ge=0, le=100_000)
    steps: int = Field(60, ge=10, le=200)


@app.post("/api/learn/regime-path")
def learn_regime_path(req: RegimePathReq):
    return ok({"blocks": pages.learn_regime_path(req.seed, req.steps)})
