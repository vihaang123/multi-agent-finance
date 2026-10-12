"""
End-to-end UI tests with Streamlit's AppTest: the real app script is executed, widgets are
clicked, and the resulting session state / rendered elements are checked.
"""

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import config as cfg

PAGES = ["Dashboard", "Simulation", "RL Lab", "Portfolio Analytics", "Experiments", "Learn"]
TIMEOUT = 180
APP_PATH = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def new_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=TIMEOUT).run()


def goto(at: AppTest, page: str) -> AppTest:
    at.sidebar.radio(key="nav").set_value(page).run()
    return at


def button(at: AppTest, label: str):
    hits = [b for b in at.button if b.label == label]
    assert hits, f"no button '{label}' (have {[b.label for b in at.button]})"
    return hits[0]


def clean(at: AppTest):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.fixture()
def app():
    return new_app()


# ------------------------------------------------------------------ #
def test_app_boots_with_synthetic_banner_and_default_agents(app):
    clean(app)
    html = " ".join(m.value for m in app.markdown)
    assert "Synthetic data" in html and "not real market" in html
    assert app.session_state["agents_source"] == "reference"
    assert app.session_state["sim"].done and app.session_state["sim"].t == cfg.EPISODE_LENGTH


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_errors(app, page):
    goto(app, page)
    clean(app)
    assert len(app.title) == 1
    assert "Synthetic" in " ".join(m.value for m in app.markdown)       # banner on every page


# ------------------------------- Dashboard ------------------------------- #
def test_dashboard_run_full_episode_and_clear(app):
    app.number_input(key="dash_seed").set_value(77).run()
    button(app, "Run a full episode").click().run()
    clean(app)
    sim = app.session_state["sim"]
    assert sim.seed == 77 and sim.t == cfg.EPISODE_LENGTH and sim.done
    button(app, "Clear (step 0)").click().run()
    clean(app)
    assert app.session_state["sim"].t == 0
    assert any("No steps have been simulated" in m.value for m in app.markdown)


def test_dashboard_navigation_button(app):
    button(app, "Step through it in the Simulation page").click().run()
    assert app.session_state["nav"] == "Simulation"
    assert app.title[0].value == "Interactive Simulation"


# ------------------------------- Simulation ------------------------------ #
def test_simulation_step_pause_reset_controls(app):
    goto(app, "Simulation")
    button(app, "Reset").click().run()
    sim = app.session_state["sim"]
    assert sim.t == 0 and not sim.running
    button(app, "Step").click().run()
    clean(app)
    assert app.session_state["sim"].t == 1
    app.number_input(key="sim_steps_n").set_value(7).run()
    button(app, "Step xN").click().run()
    assert app.session_state["sim"].t == 8
    button(app, "Start").click().run()
    clean(app)
    assert app.session_state["sim"].running is True
    button(app, "Pause").click().run()
    assert app.session_state["sim"].running is False
    button(app, "Run to end").click().run()
    assert app.session_state["sim"].done
    assert button(app, "Step").disabled and button(app, "Start").disabled        # nothing left to do
    button(app, "Reset").click().run()
    assert app.session_state["sim"].t == 0


def test_simulation_manual_policy_changes_next_action(app):
    goto(app, "Simulation")
    button(app, "Reset").click().run()
    app.selectbox(key="sim_policy_kind_0").set_value("fixed").run()
    app.selectbox(key="sim_basket_0").set_value(5).run()
    button(app, "Step").click().run()
    clean(app)
    assert app.session_state["sim"].last["actions"][0] == 5


def test_simulation_views_all_render_after_steps(app):
    goto(app, "Simulation")
    button(app, "Reset").click().run()
    button(app, "Step xN").click().run()
    for view in ["Overview", "How agents decide", "Money flow (Rs)", "Step log"]:
        app.radio(key="sim_view").set_value(view).run()
        clean(app)
    # money flow reconciles: table exists with the four assets
    app.radio(key="sim_view").set_value("Money flow (Rs)").run()
    assert any(set(cfg.ASSET_NAMES) <= set(df.value.index) for df in app.dataframe)


def test_simulation_apply_settings_restarts_with_new_seed(app):
    goto(app, "Simulation")
    app.number_input(key="sim_seed").set_value(123).run()
    button(app, "Apply seed & impact setting (restarts at step 0)").click().run()
    clean(app)
    sim = app.session_state["sim"]
    assert sim.seed == 123 and sim.t == 0


# ------------------------------- RL Lab ---------------------------------- #
def _quick_hyper(at, n=20):
    at.number_input(key="lab_n_episodes").set_value(n)
    return at


def test_rl_lab_train_from_scratch_genuinely_learns(app):
    goto(app, "RL Lab")
    _quick_hyper(app, 30).run()
    button(app, "Train from scratch").click().run()
    clean(app)
    ss = app.session_state
    assert ss["agents_source"] == "lab" and ss["lab_total_episodes"] == 30
    assert len(ss["lab_log"]) == 30 and list(ss["lab_log"]["episode"]) == list(range(30))
    for a in ss["agents"].values():
        assert a.n_updates == 30 * cfg.EPISODE_LENGTH                       # one Q-update per step
        assert not np.allclose(a.q_table, cfg.Q_INIT_VALUE)                 # table really changed
        assert a.epsilon < cfg.EPSILON_INITIAL                              # exploration decayed
    # simulation was refreshed to use the new agents
    assert ss["sim"].agents is ss["agents"] and ss["sim"].done


def test_rl_lab_continue_extends_training(app):
    goto(app, "RL Lab")
    _quick_hyper(app, 20).run()
    button(app, "Train from scratch").click().run()
    q1 = {k: v.q_table.copy() for k, v in app.session_state["agents"].items()}
    button(app, "Continue training").click().run()
    clean(app)
    ss = app.session_state
    assert ss["lab_total_episodes"] == 40 and list(ss["lab_log"]["episode"]) == list(range(40))
    assert any(not np.array_equal(q1[k], ss["agents"][k].q_table) for k in q1)
    assert ss["agents"][0].n_updates == 40 * cfg.EPISODE_LENGTH


def test_rl_lab_hyperparameters_are_applied(app):
    goto(app, "RL Lab")
    _quick_hyper(app, 10)
    app.slider(key="lab_alpha").set_value(0.5)
    app.slider(key="lab_gamma").set_value(0.5)
    app.slider(key="lab_qinit").set_value(5.0)
    app.slider(key="lab_decay").set_value(0.99)
    app.run()
    button(app, "Train from scratch").click().run()
    clean(app)
    a = app.session_state["agents"][0]
    assert a.alpha == 0.5 and a.gamma == 0.5 and a.epsilon_decay == 0.99
    assert a.epsilon == pytest.approx(0.99 ** 10)


def test_rl_lab_rejects_invalid_settings(app):
    goto(app, "RL Lab")
    app.slider(key="lab_eps0").set_value(0.1)
    app.slider(key="lab_epsmin").set_value(0.4)
    app.run()
    button(app, "Train from scratch").click().run()
    assert any("epsilon_min" in e.value for e in app.error)
    assert app.session_state["agents_source"] == "reference"          # nothing was trained


def test_rl_lab_rejects_training_seed_inside_eval_block(app):
    goto(app, "RL Lab")
    app.number_input(key="lab_seed").set_value(60).run()
    button(app, "Train from scratch").click().run()
    assert any("held-out" in e.value for e in app.error)
    assert app.session_state["agents_source"] == "reference"


def test_rl_lab_restore_reference_agents(app):
    goto(app, "RL Lab")
    _quick_hyper(app, 10).run()
    button(app, "Train from scratch").click().run()
    assert app.session_state["agents_source"] == "lab"
    button(app, "Restore reference agents").click().run()
    clean(app)
    assert app.session_state["agents_source"] == "reference"
    assert app.session_state["lab_log"] is None


def test_rl_lab_q_inspector_walkthrough_and_evaluation(app):
    goto(app, "RL Lab")
    _quick_hyper(app, 15).run()
    button(app, "Train from scratch").click().run()
    app.selectbox(key="lab_regime").set_value(2).run()
    clean(app)
    button(app, "Show the updates").click().run()
    clean(app)
    assert any("match the Q-learning formula" in s.value for s in app.success)
    app.number_input(key="lab_eval_n").set_value(5).run()
    button(app, "Evaluate current agents").click().run()
    clean(app)
    res = app.session_state["an_eval"]
    assert len(res["eval_trajectories"]) == 5 and res["seeds"][0] == cfg.EVAL_SEED_START


def test_rl_lab_reference_curves_are_labeled_precomputed(app):
    goto(app, "RL Lab")
    assert any("precomputed" in m.value.lower() for m in app.markdown)


# ------------------------------- Analytics ------------------------------- #
def test_analytics_episode_source(app):
    goto(app, "Portfolio Analytics")
    clean(app)
    assert len(app.dataframe) >= 2
    assert app.get("plotly_chart")


def test_analytics_empty_episode_is_handled(app):
    goto(app, "Simulation")
    button(app, "Reset").click().run()
    goto(app, "Portfolio Analytics")
    clean(app)
    assert any("No steps" in i.value for i in app.info)


def test_analytics_out_of_sample_run_and_single_seed_inspection(app):
    goto(app, "Portfolio Analytics")
    app.radio(key="an_source").set_value("Out-of-sample evaluation of the active agents").run()
    app.number_input(key="an_eval_n").set_value(5).run()
    button(app, "Run out-of-sample evaluation").click().run()
    clean(app)
    res = app.session_state["an_eval"]
    assert res["seeds"] == list(range(51, 56))
    app.selectbox(key="an_seed_pick").set_value(53).run()
    clean(app)


# ------------------------------- Experiments ----------------------------- #
def test_experiments_live_run_all_four(app):
    goto(app, "Experiments")
    app.number_input(key="ex_episodes").set_value(100)
    app.number_input(key="ex_eval_n").set_value(5)
    app.run()
    button(app, "Run selected experiments").click().run()
    clean(app)
    res = app.session_state["exp_results"]
    assert set(res) == set("ABCD")
    for e in "ABCD":
        assert len(res[e]["summary"]["combined_returns_per_seed"]) == 5
    assert "training" in res["D"] and "training" not in res["A"]


def test_experiments_same_config_reproduces(app):
    goto(app, "Experiments")
    app.multiselect(key="ex_selected").set_value(["A", "B"])
    app.number_input(key="ex_eval_n").set_value(5)
    app.run()
    button(app, "Run selected experiments").click().run()
    first = app.session_state["exp_results"]["A"]["summary"]["combined_returns_per_seed"]
    button(app, "Run selected experiments").click().run()
    assert app.session_state["exp_results"]["A"]["summary"]["combined_returns_per_seed"] == first


def test_experiments_blocks_train_eval_overlap(app):
    goto(app, "Experiments")
    app.multiselect(key="ex_selected").set_value(["C"])
    app.number_input(key="ex_eval_start").set_value(40)
    app.number_input(key="ex_eval_n").set_value(10)
    app.run()
    button(app, "Run selected experiments").click().run()
    assert any("leak" in e.value for e in app.error)
    assert not app.session_state["exp_results"]


def test_experiments_requires_a_selection(app):
    goto(app, "Experiments")
    app.multiselect(key="ex_selected").set_value([])
    app.run()
    button(app, "Run selected experiments").click().run()
    assert any("at least one" in e.value for e in app.error)


def test_experiments_reference_tab_present_and_labeled(app):
    goto(app, "Experiments")
    text = " ".join(m.value for m in app.markdown)
    assert "Precomputed, not live" in text and "scripts.build_reference" in text


# ------------------------------- Learn ----------------------------------- #
def test_learn_calculators_respond(app):
    goto(app, "Learn")
    clean(app)
    app.number_input(key="learn_q_r").set_value(4.0).run()
    clean(app)
    text = " ".join(m.value for m in app.markdown)
    assert "new Q" in text
    app.selectbox(key="learn_r_role").set_value("Conservative").run()
    clean(app)


def test_state_survives_page_switches(app):
    goto(app, "RL Lab")
    app.slider(key="lab_alpha").set_value(0.37).run()
    goto(app, "Simulation")
    goto(app, "RL Lab")
    assert app.slider(key="lab_alpha").value == pytest.approx(0.37)
