"""server.trainstate: the browser-held state blob must be exact, safe and validated."""

import base64
import json
import zlib

import numpy as np
import pandas as pd
import pytest

import config as cfg
from experiments.engine import QHyperParams, RunConfig, run_experiment
from server import trainstate as T


def test_chunked_training_equals_monolithic_training_exactly():
    hp = QHyperParams()
    rc = RunConfig(experiment="D", n_train_episodes=60, n_eval_episodes=5, hp=hp)
    mono = run_experiment(rc)
    ts = T.new_state("D", hp, cfg.TRAIN_SEED, cfg.EPISODE_LENGTH, True)
    frames = []
    for k in (20, 25, 15):                          # separate "requests": serialise -> deserialise in between
        ts = T.decode(T.encode(ts))
        df, done = T.train_chunk(ts, k)
        assert done == k
        frames.append(df)
    assert pd.concat(frames, ignore_index=True).equals(mono["training_df"])
    for a in range(4):
        assert np.array_equal(ts.agents[a].q_table, mono["agents"][a].q_table)
        assert np.array_equal(ts.agents[a].visit_counts, mono["agents"][a].visit_counts)
        assert ts.agents[a].epsilon == mono["agents"][a].epsilon
        assert ts.agents[a].n_updates == mono["agents"][a].n_updates


def test_single_agent_layout_chunked_equals_monolithic():
    hp = QHyperParams()
    mono = run_experiment(RunConfig(experiment="C", n_train_episodes=30, n_eval_episodes=5, hp=hp))
    ts = T.new_state("C", hp, cfg.TRAIN_SEED, cfg.EPISODE_LENGTH, True)
    frames = []
    for k in (10, 20):
        ts = T.decode(T.encode(ts))
        frames.append(T.train_chunk(ts, k)[0])
    assert pd.concat(frames, ignore_index=True).equals(mono["training_df"])
    assert np.array_equal(ts.agents[0].q_table, mono["agents"][0].q_table)


def test_time_guard_only_changes_where_it_stops():
    hp = QHyperParams()
    a = T.new_state("D", hp, 42, 100, True)
    b = T.new_state("D", hp, 42, 100, True)
    df_a, done_a = T.train_chunk(a, 30, max_seconds=0.0)       # stops after the first 5-episode sub-chunk
    assert 5 <= done_a < 30
    T.train_chunk(a, 30 - done_a)
    T.train_chunk(b, 30)
    assert all(np.array_equal(a.agents[i].q_table, b.agents[i].q_table) for i in range(4))


def test_roundtrip_preserves_everything_including_rng():
    ts = T.new_state("D", QHyperParams(alpha=0.3), 7, 50, False, label="x")
    T.train_chunk(ts, 5)
    back = T.decode(T.encode(ts))
    assert (back.layout, back.seed, back.episode_length, back.impact, back.total_episodes, back.label) == ("D", 7, 50, False, 5, "x")
    assert back.env_rng == ts.env_rng
    for a in range(4):
        assert np.array_equal(back.agents[a].q_table, ts.agents[a].q_table)
        assert back.agents[a].rng.bit_generator.state == ts.agents[a].rng.bit_generator.state
        assert back.agents[a].alpha == 0.3


def _blob(payload: dict) -> str:
    return base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()


def _good_payload():
    ts = T.new_state("D", QHyperParams(), 1, 20, True)
    raw = zlib.decompress(base64.b64decode(T.encode(ts)))
    return json.loads(raw)


@pytest.mark.parametrize("bad", [None, "", "garbage", "!!!!", 12345, "A" * 10])
def test_garbage_blobs_rejected(bad):
    with pytest.raises(T.BlobError):
        T.decode(bad)


def test_structurally_invalid_payloads_rejected():
    good = _good_payload()
    cases = []
    p = json.loads(json.dumps(good)); p["v"] = 99; cases.append(p)
    p = json.loads(json.dumps(good)); p["layout"] = "Z"; cases.append(p)
    p = json.loads(json.dumps(good)); del p["agents"]["2"]; cases.append(p)
    p = json.loads(json.dumps(good)); p["agents"]["0"]["q"] = base64.b64encode(b"short").decode(); cases.append(p)
    p = json.loads(json.dumps(good)); p["agents"]["0"]["epsilon"] = 5; cases.append(p)
    p = json.loads(json.dumps(good)); p["agents"]["0"]["alpha"] = 0; cases.append(p)
    p = json.loads(json.dumps(good)); p["agents"]["0"]["rng"] = {"bit_generator": "PCG64", "state": "nope"}; cases.append(p)
    p = json.loads(json.dumps(good)); p["env_rng"] = {"x": 1}; cases.append(p)
    p = json.loads(json.dumps(good)); p["episode_length"] = 10 ** 9; cases.append(p)
    p = json.loads(json.dumps(good)); p["seed"] = -3; cases.append(p)
    for c in cases:
        with pytest.raises(T.BlobError):
            T.decode(_blob(c))


def test_non_finite_q_values_rejected():
    good = _good_payload()
    q = np.full((cfg.N_STATES, cfg.N_ACTIONS), np.nan)
    good["agents"]["0"]["q"] = base64.b64encode(q.tobytes()).decode()
    with pytest.raises(T.BlobError):
        T.decode(_blob(good))


def test_decompression_bomb_rejected():
    bomb = base64.b64encode(zlib.compress(b"0" * (T.MAX_DECODED_BYTES + 10_000_000), 9)).decode()
    assert len(bomb) < T.MAX_BLOB_CHARS
    with pytest.raises(T.BlobError):
        T.decode(bomb)


def test_oversized_blob_rejected():
    with pytest.raises(T.BlobError):
        T.decode("A" * (T.MAX_BLOB_CHARS + 1))


def test_load_agents_reference_and_wrong_layout():
    agents, label = T.load_agents(None)
    assert sorted(agents) == [0, 1, 2, 3] and "reference" in label.lower()
    single = T.encode(T.new_state("C", QHyperParams(), 1, 20, True))
    with pytest.raises(T.BlobError):
        T.load_agents(single)


def test_training_log_validation():
    df = pd.DataFrame({"episode": [0, 1], "reward_0": [1.0, 2.0], "drawdown_0": [0.1, 0.2]})
    log = T.df_to_log(df)
    assert "drawdown_0" not in log and log["episode"] == [0, 1]
    assert T.log_to_df(log).shape == (2, 2)
    assert T.log_to_df(None) is None
    for bad in ({"episode": [1, 2], "reward_0": [1]}, {"reward_0": [1, 2]}, {"episode": [], "reward_0": []},
                {"episode": [0], "reward_0": ["x"]}):
        with pytest.raises(T.BlobError):
            T.log_to_df(bad)
