"""
Browser end-to-end tests (Playwright + Chromium) for the Next.js front end talking to the real Python API.

Skipped unless E2E_BASE_URL is set, e.g. with both servers running locally:

    python -m uvicorn api.index:app --port 8000 &
    LOCAL_API_URL=http://127.0.0.1:8000 npx next build && LOCAL_API_URL=http://127.0.0.1:8000 npx next start -p 3000 &
    E2E_BASE_URL=http://localhost:3000 python -m pytest tests/e2e -q

(or point E2E_BASE_URL at a deployed Vercel URL). Screenshots go to E2E_SHOTS if set.
"""

import os
import re
import time
from pathlib import Path

import pytest

BASE = os.environ.get("E2E_BASE_URL")
pytestmark = pytest.mark.skipif(not BASE, reason="set E2E_BASE_URL to run browser tests")
playwright_sync = pytest.importorskip("playwright.sync_api")
SHOTS = os.environ.get("E2E_SHOTS")


@pytest.fixture(scope="module")
def browser():
    with playwright_sync.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture()
def page(browser):
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    pg = ctx.new_page()
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    pg.on("console", lambda m: pg.errors.append(m.text) if m.type == "error" else None)
    pg.set_default_timeout(60000)
    yield pg
    ctx.close()


def shot(pg, name):
    if SHOTS:
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        pg.screenshot(path=str(Path(SHOTS) / f"{name}.png"), full_page=True)


def plots(pg):
    return pg.locator(".js-plotly-plot").count()


def go(pg, path):
    pg.goto(BASE + path)
    pg.wait_for_selector("main .syn-banner")


def test_every_page_shows_the_synthetic_banner_and_has_no_console_errors(page):
    for path in ["/", "/simulation", "/rl-lab", "/analytics", "/experiments", "/learn"]:
        go(page, path)
        assert "Synthetic data" in page.locator(".syn-banner").inner_text()
        page.wait_for_selector("h1")
        time.sleep(0.5)
    assert page.errors == [], page.errors


def test_dashboard_runs_an_episode_and_shows_charts(page):
    go(page, "/")
    page.wait_for_selector(".kpi")
    page.wait_for_selector(".js-plotly-plot")
    assert page.locator(".kpi").count() == 5
    assert "step 100/100" in page.content()
    page.get_by_role("button", name="Clear (step 0)").click()
    page.wait_for_selector("text=step 0/100")
    assert "No steps have been simulated yet" in page.content()
    page.get_by_role("button", name="Run a full episode").click()
    page.wait_for_selector("text=step 100/100")
    shot(page, "dashboard")
    assert page.errors == [], page.errors


def test_simulation_step_play_pause_policy_and_views(page):
    go(page, "/simulation")
    page.wait_for_selector("text=step 100/100")                # example episode played on first load
    page.get_by_role("button", name="Reset", exact=True).click()
    page.wait_for_selector("text=step 0/100")
    page.get_by_role("button", name="Step", exact=True).click()
    page.wait_for_selector("text=step 1/100")
    page.get_by_role("button", name="Step xN").click()
    page.wait_for_selector("text=step 11/100")
    # manual control: switch Balanced to a fixed basket, then step and see it in the decisions view
    page.locator("details.settings summary").click() if not page.locator("details.settings[open]").count() else None
    sels = page.locator("details.settings select")
    sels.nth(2).select_option("fixed")
    page.get_by_role("button", name="Step", exact=True).click()
    page.wait_for_selector("text=step 12/100")
    page.get_by_text("How agents decide").click()
    page.wait_for_selector("text=Reward received")
    assert "Fixed / manual basket" in page.content()
    page.get_by_text("Money flow (Rs)").click()
    page.wait_for_selector("text=Rs invested (after trading cost)")
    page.get_by_text("Step log").click()
    page.wait_for_selector("text=Download step log (CSV)")
    page.get_by_text("Overview", exact=True).click()
    # auto-play advances then pauses
    page.get_by_role("button", name="Start", exact=True).click()
    page.wait_for_function("() => /step (1[5-9]|[2-9][0-9])\\/100/.test(document.body.innerText)")
    page.get_by_role("button", name="Pause").click()
    time.sleep(1.5)
    t1 = re.search(r"step (\d+)/100", page.inner_text("main")).group(1)
    time.sleep(1.5)
    t2 = re.search(r"step (\d+)/100", page.inner_text("main")).group(1)
    assert t1 == t2, "paused simulation must not keep advancing"
    page.get_by_role("button", name="Run to end").click()
    page.wait_for_selector("text=Episode complete")
    shot(page, "simulation")
    assert page.errors == [], page.errors


def test_simulation_survives_navigation(page):
    go(page, "/simulation")
    page.wait_for_selector("text=step 100/100")
    page.get_by_role("button", name="Reset", exact=True).click()
    page.wait_for_selector("text=step 0/100")
    page.get_by_role("button", name="Step xN").click()
    page.wait_for_selector("text=step 10/100")
    page.get_by_role("link", name="Dashboard").click()
    page.wait_for_selector("text=step 10/100")


def test_rl_lab_train_inspect_policy_walkthrough_evaluate_restore(page):
    go(page, "/rl-lab")
    page.wait_for_selector("text=Precomputed reference agents")
    page.wait_for_selector(".js-plotly-plot")                  # precomputed reference curves
    # small live training run
    page.get_by_label("Training episodes (max 3000)").fill("100")
    page.get_by_label("Training episodes (max 3000)").blur()
    page.get_by_role("button", name="Train from scratch").click()
    page.wait_for_selector("text=Finished 100 episodes")
    assert "Trained in the RL Lab: 100 episodes" in page.inner_text("main")
    page.wait_for_selector("text=Q-updates")
    shot(page, "rl_lab_curves")
    # continue
    page.get_by_role("button", name="Continue training").click()
    page.wait_for_selector("text=Trained in the RL Lab: 200 episodes")
    page.get_by_role("tab", name="Q-table inspector").click()
    page.wait_for_selector("text=Table size 108 states x 7 actions")
    page.get_by_role("tab", name="Policy maps").click()
    page.wait_for_selector("text=What each agent does in each regime")
    page.get_by_role("tab", name="Update walkthrough").click()
    page.get_by_role("button", name="Show the updates").click()
    page.wait_for_selector("text=All displayed updates match the Q-learning formula.")
    page.get_by_role("tab", name="Evaluate (out-of-sample)").click()
    page.get_by_role("button", name="Evaluate current agents").click()
    page.wait_for_selector("text=Specialization score")
    shot(page, "rl_lab_eval")
    # dashboard now uses the lab agents
    page.get_by_role("link", name="Dashboard").click()
    page.wait_for_selector("text=Trained in the RL Lab: 200 episodes")
    page.wait_for_selector("text=step 100/100")
    page.get_by_role("link", name="RL Lab").click()
    page.get_by_role("button", name="Restore reference agents").click()
    page.wait_for_selector("text=Precomputed reference agents")
    assert page.errors == [], page.errors


def test_rl_lab_rejects_training_seed_inside_evaluation_block(page):
    go(page, "/rl-lab")
    page.get_by_label("Training market seed").fill("60")
    page.get_by_label("Training market seed").blur()
    page.get_by_role("button", name="Train from scratch").click()
    page.wait_for_selector("text=lies inside the held-out evaluation block")


def test_analytics_episode_and_out_of_sample(page):
    go(page, "/analytics")
    page.wait_for_selector("text=Growth & returns")
    page.wait_for_selector(".js-plotly-plot")
    page.get_by_role("tab", name="Risk").click()
    page.get_by_role("tab", name="Transaction costs").click()
    page.wait_for_selector("text=Cost model")
    page.get_by_text("Out-of-sample evaluation of the active agents").click()
    page.get_by_label("Number of episodes").fill("10")
    page.get_by_label("Number of episodes").blur()
    page.get_by_role("button", name="Run out-of-sample evaluation").click()
    page.wait_for_selector("text=Inspect one held-out episode in detail")
    page.wait_for_selector("text=Growth & returns")
    shot(page, "analytics_oos")
    assert page.errors == [], page.errors


def test_experiments_reference_and_small_live_run(page):
    go(page, "/experiments")
    page.wait_for_selector("text=Precomputed, not live")
    page.wait_for_selector("text=Reading these results")
    assert "Specialization" in page.content()
    page.get_by_label("Training episodes (C, D; max 3000)").fill("50")
    page.get_by_label("Training episodes (C, D; max 3000)").blur()
    page.get_by_label("Evaluation episodes").fill("5")
    page.get_by_label("Evaluation episodes").blur()
    page.get_by_role("button", name="Run selected experiments").click()
    page.wait_for_selector("text=Live run | training 50 episodes", timeout=120000)
    page.wait_for_selector("text=Reading these results")
    shot(page, "experiments_live")
    assert page.errors == [], page.errors


def test_experiments_requires_a_selection(page):
    go(page, "/experiments")
    for e in "ABCD":
        page.get_by_label(re.compile(rf"^{e} - ")).uncheck()
    page.get_by_role("button", name="Run selected experiments").click()
    page.wait_for_selector("text=Select at least one experiment.")


def test_learn_page_calculators(page):
    go(page, "/learn")
    page.wait_for_selector("text=Markov Decision Process")
    page.get_by_role("tab", name="Market & regimes").click()
    page.wait_for_selector("text=Try the regime chain")
    page.wait_for_selector("text=Time spent")
    page.get_by_role("tab", name="States & actions").click()
    page.wait_for_selector("text=State index = 0 x 27")
    page.get_by_label("Regime").select_option("2")
    page.wait_for_selector("text=State index = 2 x 27")
    page.get_by_role("tab", name="Reward").click()
    page.wait_for_selector("text=Total =")
    page.get_by_role("tab", name="Q-learning").click()
    page.wait_for_selector("text=new Q = 2 + 0.1")
    shot(page, "learn")
    assert page.errors == [], page.errors


def test_mobile_layout_has_no_horizontal_scroll(browser):
    ctx = browser.new_context(viewport={"width": 390, "height": 800})
    pg = ctx.new_page()
    pg.set_default_timeout(60000)
    for path in ["/", "/simulation", "/rl-lab", "/analytics", "/experiments", "/learn"]:
        pg.goto(BASE + path)
        pg.wait_for_selector("main .syn-banner")
        time.sleep(1.5)
        overflow = pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        assert overflow <= 1, f"{path} overflows horizontally by {overflow}px"
    pg.goto(BASE + "/")
    pg.wait_for_selector(".kpi")
    time.sleep(1)
    shot(pg, "mobile_dashboard")
    ctx.close()
