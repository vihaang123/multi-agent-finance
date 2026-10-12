"""Chart builders produce valid figures from real data; the shipped reference results are genuine."""

import json

import numpy as np
import plotly.graph_objects as go
import pytest

import config as cfg
from dashboard import charts, reference, views
from dashboard.session import SimulationSession
from dashboard.theme import inr, pct
from experiments.engine import RunConfig, evaluate_agents, run_experiment
from experiments.run_experiments import AGENT_IDS, AGENT_ROLES


@pytest.fixture(scope="module")
def trained():
    r = run_experiment(RunConfig(experiment="D", n_train_episodes=40, n_eval_episodes=4))
    return r


@pytest.fixture(scope="module")
def sim(trained):
    s = SimulationSession(trained["agents"], seed=51, episode_length=30)
    s.run_to_end()
    return s


def test_every_chart_builder_returns_a_figure(trained, sim):
    series = views.series_by_name(sim.history, sim.agent_ids, sim.roles, sim.capitals)
    colors = views.color_map(sim.roles)
    sleeves, total = views.split_series(series)
    mf = views.metrics_frame(sim.history, sim.agent_ids, sim.roles, sim.capitals).drop(index="Total")
    figs = [
        charts.portfolio_value(sleeves, colors, sim.regime_path()), charts.portfolio_value(total, colors, sim.regime_path(), initial=1e6),
        charts.cumulative_return_chart(series, colors), charts.drawdown_chart(series, colors),
        charts.rolling_vol_chart(series, colors, 5), charts.step_return_hist(series, colors),
        charts.txcost_chart(views.txcosts_by_name(sim.history, sim.agent_ids, sim.roles), colors),
        charts.allocation_stack(views.weights_array(sim.history, 0)), charts.allocation_bars(views.current_rupees(sim)),
        charts.allocation_donut(np.sum(list(views.current_rupees(sim).values()), axis=0)),
        charts.weight_waterfall(sim.last["asset_pnl"][0]), charts.asset_pnl_contribution(views.asset_pnl_total(sim.history, AGENT_IDS)),
        charts.reward_components_bar(sim.last["reward_components"][0]),
        charts.asset_returns_bar(sim.last["base_returns"], sim.last["asset_returns"]),
        charts.metric_bars(mf, "Total return", "t", colors), charts.risk_return_scatter({n: (mf.loc[n, "Volatility"], mf.loc[n, "Total return"]) for n in mf.index}, colors),
        charts.learning_curves(trained["training_df"], AGENT_IDS, AGENT_ROLES, 5), charts.epsilon_curve(trained["training_df"], AGENT_IDS, AGENT_ROLES),
        charts.td_curve(trained["training_df"], AGENT_IDS, AGENT_ROLES, 5), charts.training_capital_curve(trained["training_df"], AGENT_IDS, AGENT_ROLES, 5),
        charts.q_values_bar(trained["agents"][0].q_values(3), trained["agents"][0].visit_counts[3]),
        *[charts.policy_heatmap(trained["agents"][1], m) for m in ("greedy", "maxq", "visits")],
        charts.transition_heatmap(), charts.regime_path_chart([0, 1, 2, 3, 2]), charts.asset_return_params_heatmap(),
        charts.action_hist_heatmap({"a": np.ones(7) / 7}),
        charts.returns_box({"x": [0.01, -0.02, 0.03]}, colors), charts.mean_curve_band({"x": ([1, 2, 3], [0.1, 0.1, 0.1])}, colors),
    ]
    assert all(isinstance(f, go.Figure) for f in figs)
    assert all(len(f.data) >= 1 for f in figs if f.layout.height and "No steps" not in str(f.layout.annotations))


def test_empty_states_do_not_crash():
    assert isinstance(charts.portfolio_value({}, {}), go.Figure)
    assert isinstance(charts.allocation_stack(np.zeros((0, 4))), go.Figure)
    assert isinstance(charts.learning_curves(None, [0], {0: "Growth"}), go.Figure)


def test_state_grid_covers_all_108_states_once():
    g = charts.state_grid(np.arange(cfg.N_STATES, dtype=float))
    assert g.shape == (9, 12) and not np.isnan(g).any() and sorted(g.ravel()) == list(range(cfg.N_STATES))


def test_formatters():
    assert inr(1000000) == "₹10,00,000" and inr(-250000) == "-₹2,50,000" and inr(999) == "₹999"
    assert inr(float("nan")) == "n/a" and pct(0.0251, 2, True) == "+2.51%" and pct(float("nan")) == "n/a"


# ----------------------- shipped reference integrity ----------------------- #
@pytest.fixture(scope="module")
def ref():
    r = reference.load_reference_results()
    assert r is not None, "data/reference_results.json missing - run python -m scripts.build_reference"
    return r


def test_reference_metadata_and_structure(ref):
    m = ref["meta"]
    assert m["train_episodes"] == 5000 and m["train_seed"] == cfg.TRAIN_SEED
    assert m["eval_seeds"] == [cfg.EVAL_SEED_START, cfg.EVAL_SEED_START + cfg.N_EVAL_EPISODES - 1]
    assert "SYNTHETIC" in m["data_type"]
    assert set(ref["experiments"]) == set("ABCD")
    for e, v in ref["experiments"].items():
        assert v["summary"]["combined"]["n_episodes"] == cfg.N_EVAL_EPISODES
        assert len(v["summary"]["combined_returns_per_seed"]) == cfg.N_EVAL_EPISODES
        assert ("training" in v) == (e in "CD")


def test_reference_eval_seeds_disjoint_from_training_seed(ref):
    lo, hi = ref["meta"]["eval_seeds"]
    assert not (lo <= ref["meta"]["train_seed"] <= hi)


def test_reference_agents_reproduce_stored_oos_results_exactly(ref):
    """The shipped Q-tables, evaluated on seeds 51-100 with the real engine, give exactly the stored
    experiment-D numbers: the JSON is not hand-written and matches the Q-tables."""
    agents = reference.make_reference_agents()
    caps = {a: cfg.INITIAL_CAPITAL_PER_AGENT for a in AGENT_IDS}
    res = evaluate_agents(agents, AGENT_IDS, AGENT_ROLES, caps, list(range(51, 101)))
    stored = ref["experiments"]["D"]["summary"]
    assert res["summary"]["combined_returns_per_seed"] == pytest.approx(stored["combined_returns_per_seed"], abs=1e-12)
    assert res["summary"]["specialization"] == pytest.approx(stored["specialization"])


@pytest.mark.parametrize("exp", ["A", "B"])
def test_reference_nonlearning_experiments_reproduce(ref, exp):
    live = run_experiment(RunConfig(experiment=exp, n_train_episodes=1))["summary"]
    stored = ref["experiments"][exp]["summary"]
    assert live["combined_returns_per_seed"] == pytest.approx(stored["combined_returns_per_seed"], abs=1e-12)


def test_reference_agents_are_independent_copies():
    a, b = reference.make_reference_agents(), reference.make_reference_agents()
    a[0].q_table[:] = 123
    assert not np.allclose(b[0].q_table, 123)
    assert a[0].visit_counts.sum() > 0 and a[0].epsilon == pytest.approx(cfg.EPSILON_MIN)
