"""The stateless JSON API (server/app.py) via FastAPI's TestClient: no network, real engine underneath."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import config as cfg
from experiments.engine import QHyperParams, RunConfig, run_experiment
from server.app import app

client = TestClient(app)
POL = [{"kind": "trained", "basket": 2}] * 4


def post(path, body=None, expect=200):
    r = client.post(path, json=body if body is not None else {})
    assert r.status_code == expect, (path, r.status_code, r.text[:300])
    return r.json()


def strict_json(text):
    """JSON.parse-compatible: NaN / Infinity are not allowed."""
    return json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-finite {c}")))


def sim(steps=None, **kw):
    body = {"seed": 51, "impact": True, "policies": POL, "steps": steps or [], "view": "none"}
    body.update(kw)
    return post("/api/simulate", body)


def walk(blocks):
    for b in blocks:
        yield b
        for key in ("cols", "tabs", "options"):
            for sub in b.get(key, []):
                yield from walk(sub["blocks"])
        for it in b.get("items", []):
            if isinstance(it, dict) and "blocks" in it:
                yield from walk(it["blocks"])


# ------------------------------------------------------------------ meta / health
def test_health_and_meta():
    assert client.get("/api/health").json()["status"] == "ok"
    m = client.get("/api/meta").json()
    assert m["roles"] == {"0": "Growth", "1": "Conservative", "2": "Balanced", "3": "RiskLiquidity"}
    assert len(m["baskets"]) == cfg.N_ACTIONS and m["reference"]["available"]
    assert "Synthetic data" in m["synthetic_text"] and m["limits"]["chunk_episodes"] == 200
    assert m["train_seed"] == cfg.TRAIN_SEED and m["eval_seed_start"] == cfg.EVAL_SEED_START


# ------------------------------------------------------------------ simulation (stateless replay)
def test_simulation_is_stateless_and_deterministic():
    full = sim(to_end=True)
    assert full["t"] == 100 and full["done"] and len(full["steps"]) == 100
    again = sim(to_end=True)
    assert again["steps"] == full["steps"] and again["total_value"] == full["total_value"]
    # step by step across separate requests == one request
    steps = []
    for _ in range(12):
        steps = sim(steps, advance=1)["steps"]
    assert steps == full["steps"][:12]
    resumed = sim(steps, to_end=True)
    assert resumed["steps"] == full["steps"] and resumed["total_value"] == full["total_value"]


def test_simulation_policy_change_applies_to_future_steps_only():
    a = sim(advance=10)
    pols = [dict(p) for p in POL]
    pols[2] = {"kind": "fixed", "basket": 5}
    b = sim(a["steps"], policies=pols, advance=5)
    assert b["steps"][:10] == a["steps"] and all(s["a"][2] == 5 and s["k"][2] == "f" for s in b["steps"][10:])
    c = sim(b["steps"], policies=POL, advance=0, view="decisions")           # rebuild keeps the recorded kinds
    assert c["steps"] == b["steps"]
    assert any(bl["type"] == "cards" for bl in c["blocks"])


def test_simulation_views_return_blocks_without_non_finite_numbers():
    for view in ["dashboard", "overview", "decisions", "money", "log", "analytics"]:
        r = client.post("/api/simulate", json={"seed": 52, "impact": True, "policies": POL, "steps": [], "to_end": True, "view": view})
        assert r.status_code == 200, (view, r.text[:200])
        data = strict_json(r.text)
        assert data["blocks"], view
        assert data["agents_label"].startswith("Precomputed reference")
        kinds = {b["type"] for b in walk(data["blocks"])}
        assert "plot" in kinds or view in ("log",), view


def test_simulation_at_step_zero_and_empty_views():
    r = sim(view="dashboard")
    assert r["t"] == 0 and not r["done"]
    assert any(b["type"] == "callout" for b in r["blocks"])
    for view in ["overview", "decisions", "money", "log", "analytics"]:
        assert sim(view=view)["blocks"]


def test_simulation_validation():
    base = {"seed": 51, "impact": True, "policies": POL, "steps": []}
    post("/api/simulate", {**base, "seed": -1}, 422)
    post("/api/simulate", {**base, "policies": POL[:3]}, 422)
    post("/api/simulate", {**base, "policies": [{"kind": "magic", "basket": 1}] * 4}, 422)
    post("/api/simulate", {**base, "policies": [{"kind": "fixed", "basket": 9}] * 4}, 422)
    post("/api/simulate", {**base, "view": "nope"}, 422)
    post("/api/simulate", {**base, "steps": [{"a": [0, 0, 0, 99], "k": "tttt"}]}, 400)
    post("/api/simulate", {**base, "steps": [{"a": [0, 0, 0, 0], "k": "xxxx"}]}, 422)
    post("/api/simulate", {**base, "steps": [{"a": [0, 0, 0, 0], "k": "tttt"}] * 101}, 422)
    post("/api/simulate", {**base, "agents": "garbage"}, 400)


# ------------------------------------------------------------------ RL Lab
@pytest.fixture(scope="module")
def lab():
    first = post("/api/lab/train", {"mode": "fresh", "episodes": 30, "seed": 42, "impact": True})
    second = post("/api/lab/train", {"mode": "continue", "state": first["state"], "episodes": 20})
    log = {k: first["rows"][k] + second["rows"][k] for k in first["rows"]}
    return {"first": first, "second": second, "state": second["state"], "log": log}


def test_lab_training_chunks_equal_monolithic(lab):
    assert lab["first"]["total_episodes"] == 30 and lab["second"]["total_episodes"] == 50
    assert "50 episodes" in lab["second"]["label"]
    mono = run_experiment(RunConfig(experiment="D", n_train_episodes=50, n_eval_episodes=5))
    assert np.allclose(lab["log"]["reward_0"], mono["training_df"]["reward_0"].round(6))
    assert lab["log"]["episode"] == list(range(50))
    assert "drawdown_0" not in lab["log"]


def test_lab_views(lab):
    st = lab["state"]
    curves = post("/api/lab/curves", {"agents": st, "log": lab["log"], "smooth": 10})
    assert sum(b["type"] == "plot" for b in walk(curves["blocks"])) >= 4
    ref_curves = post("/api/lab/curves", {})
    assert any(b["type"] == "callout" for b in ref_curves["blocks"])
    assert post("/api/lab/inspect", {"agents": st, "agent": 1, "regime": 2, "eq": 1, "bond": 0, "gold": 2})["blocks"]
    for mode in ("greedy", "maxq", "visits"):
        assert post("/api/lab/policy", {"agents": st, "agent": 0, "mode": mode})["blocks"]
    walkthrough = post("/api/lab/walkthrough", {"agents": st, "agent": 0, "steps": 6, "seed": 7})
    assert any(b["type"] == "callout" and b["tone"] == "ok" for b in walkthrough["blocks"])
    ev = post("/api/lab/evaluate", {"agents": st, "n": 5})
    assert 0 <= ev["specialization"] <= 1


def test_lab_inputs_validated(lab):
    post("/api/lab/train", {"mode": "fresh", "episodes": 5, "seed": 60}, 400)                 # inside the evaluation block
    post("/api/lab/train", {"mode": "fresh", "episodes": 500, "seed": 42}, 422)               # over the per-request cap
    post("/api/lab/train", {"mode": "fresh", "episodes": 5, "seed": 42, "hp": {"alpha": 0}}, 422)
    post("/api/lab/train", {"mode": "fresh", "episodes": 5, "seed": 42, "hp": {"epsilon_min": 0.9, "epsilon_initial": 0.5}}, 400)
    post("/api/lab/inspect", {"agents": lab["state"], "agent": 9}, 422)
    post("/api/lab/inspect", {"agents": "garbage"}, 400)
    post("/api/lab/curves", {"agents": lab["state"], "log": {"episode": [0, 1], "reward_0": [1.0]}}, 400)
    post("/api/lab/curves", {"agents": lab["state"], "log": {"episode": [0], "reward_0": [1.0]}}, 400)   # missing columns
    post("/api/lab/evaluate", {"agents": lab["state"], "n": 1}, 422)


def test_lab_continue_from_reference_agents():
    r = post("/api/lab/train", {"mode": "continue", "episodes": 5, "seed": 42})
    assert r["total_episodes"] == 5 and "Trained in the RL Lab" in r["label"]


def test_trained_state_changes_the_simulation(lab):
    ref = sim(to_end=True)
    mine = sim(to_end=True, agents=lab["state"])
    assert mine["agents_label"].startswith("Trained in the RL Lab") and mine["steps"] != ref["steps"]


# ------------------------------------------------------------------ Analytics
def test_analytics_out_of_sample_matches_engine():
    r = post("/api/analytics/oos", {"n": 5, "start": 51})
    assert r["seeds"] == [51, 52, 53, 54, 55] and r["blocks"]
    ep = post("/api/analytics/oos-episode", {"seed": 53, "vol_window": 7})
    assert any(b["type"] == "tabs" for b in ep["blocks"])
    post("/api/analytics/oos", {"n": 3}, 422)


# ------------------------------------------------------------------ Experiments
CFG = {"n_train_episodes": 30, "n_eval_episodes": 5}


def test_experiments_stateless_pipeline_matches_run_experiment():
    entries = {}
    for e in "AB":
        entries[e] = post("/api/experiments/baseline", {"experiment": e, "config": CFG})["entry"]
    for e in "CD":
        a = post("/api/experiments/train", {"experiment": e, "config": CFG, "episodes": 12})
        b = post("/api/experiments/train", {"experiment": e, "config": CFG, "state": a["state"], "episodes": 18})
        log = {k: a["rows"][k] + b["rows"][k] for k in a["rows"]}
        entries[e] = post("/api/experiments/evaluate", {"experiment": e, "config": CFG, "state": b["state"], "log": log})["entry"]
    for e in "ABCD":
        mono = run_experiment(RunConfig(experiment=e, n_train_episodes=30, n_eval_episodes=5))
        got = entries[e]["summary"]["combined"]["total_return_mean"]
        assert got == pytest.approx(mono["summary"]["combined"]["total_return_mean"], abs=1e-12), e
    assert "training" in entries["D"] and "training" not in entries["A"]
    rep = post("/api/experiments/report", {"entries": entries, "meta_line": "test"})
    texts = " ".join(b.get("text", "") for b in walk(rep["blocks"]))
    assert "Highest mean out-of-sample return" in texts
    assert any(b["type"] == "download" for b in rep["blocks"])


def test_experiments_validation():
    post("/api/experiments/train", {"experiment": "D", "config": {**CFG, "train_seed": 53}, "episodes": 5}, 400)   # leaks into eval block
    post("/api/experiments/train", {"experiment": "D", "config": {"n_train_episodes": 99999}, "episodes": 5}, 422)
    post("/api/experiments/train", {"experiment": "D", "config": CFG, "episodes": 50}, 400)            # more than configured
    post("/api/experiments/train", {"experiment": "X", "config": CFG, "episodes": 5}, 422)
    post("/api/experiments/report", {"entries": {"A": {"nope": 1}}}, 400)
    post("/api/experiments/report", {"entries": {"Q": {}}}, 400)
    c = post("/api/experiments/train", {"experiment": "C", "config": CFG, "episodes": 5})
    post("/api/experiments/evaluate", {"experiment": "D", "config": CFG, "state": c["state"], "log": c["rows"]}, 400)


def test_reference_experiments_show_the_honest_findings():
    r = post("/api/experiments/reference")
    assert r["available"]
    text = " ".join(b.get("text", "") for b in walk(r["blocks"]))
    assert "Precomputed, not live" in text and "did NOT beat the fixed Balanced basket" in text
    assert "same basket" in text                      # no role specialization is reported, not hidden


# ------------------------------------------------------------------ Learn
def test_learn_endpoints():
    st = post("/api/learn/static")
    assert {"transition", "stationary", "params", "baskets", "reward_weights"} <= set(st)
    assert st["transition"]["type"] == "plot" and st["baskets"]["type"] == "table"
    shares = [float(r[1].rstrip("%")) for r in st["stationary"]["rows"]]
    assert sum(shares) == pytest.approx(100, abs=0.5)
    assert post("/api/learn/regime-path", {"seed": 3, "steps": 40})["blocks"]
    post("/api/learn/regime-path", {"seed": 3, "steps": 5}, 422)


def test_responses_are_not_cached():
    assert client.post("/api/learn/static", json={}).headers["cache-control"] == "no-store"
