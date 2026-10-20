# Multi-Agent Reinforcement Learning for Financial Resource Allocation

A university RL + Multi-Agent Systems mini project with a working web dashboard built for **Vercel**: a Next.js front end
plus a stateless FastAPI (Python) function. Four independent tabular Q-learning agents (Growth, Conservative, Balanced,
RiskLiquidity) split Rs 10,00,000 across **Equity / Bonds / Gold / Cash** inside one shared, simulated market.

> **All data is synthetic.** Prices, returns and market regimes come from a simulated Markov-regime model
> (`environment/market.py`). Nothing here is real market data, investment advice or a prediction, and results are not
> real-world performance. The project has **no** real-money trading and **no** broker connection.

**Status:** deployed on Vercel at <https://multi-agent-finance-eight.vercel.app/> (project linked to this GitHub repository, so every
push to `main` redeploys). The deployment was checked by hand in Chrome page by page (Dashboard, Simulation, RL Lab training and
evaluation, Analytics, Experiments live run, Learn) on the version before the redesign. The redesign described below was tested
locally (see "Tests"), and after it deployed the same pages were checked again by hand on the live URL in Chrome: dashboard (ribbon,
dark theme, scoreboard), simulation (step, play/pause, shortcuts, all four views), RL Lab (train, saved after reload, inspector, policy
maps, walkthrough, evaluation, reset), analytics (all views plus the out-of-sample run), a live experiments run, and the Learn tabs.
The automated browser suite was run against the local build only (the authoring sandbox cannot reach vercel.app), and the phone-width
layout was checked locally, not on the live site. See [DEPLOYMENT.md](DEPLOYMENT.md).

---

## Architecture

```
Browser (Next.js / React)  --/api/*-->  api/index.py  ->  server/app.py (FastAPI, stateless)  ->  engine (numpy / pandas)
```
- The Python function keeps **no state between requests** (serverless). The simulation is rebuilt per request by replaying
  `(seed, step list)`; trained agents travel as an opaque, validated `state` string (base64 of zlib-compressed JSON holding
  Q-tables, visit counts, epsilons and RNG states) kept in the browser. Chunked training is bit-identical to one long run (tested).
- The server returns UI "blocks" (headings, KPIs, tables, Plotly JSON ...) that a generic React renderer draws.
- Training runs in chunks of 50 episodes per request (server cap 200, 20 s time guard) to fit Vercel's function limit.

## Quick start (local)

Requires Python 3.13 (3.11+ should work) and Node 20+.
```bash
git clone https://github.com/vihaang123/multi-agent-finance.git && cd multi-agent-finance
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
npm install
uvicorn api.index:app --port 8000                         # terminal 1: the Python API
LOCAL_API_URL=http://127.0.0.1:8000 npm run dev           # terminal 2: http://localhost:3000
```
On Windows PowerShell set the variable with `$env:LOCAL_API_URL="http://127.0.0.1:8000"; npm run dev`.
Tested with Python 3.13, FastAPI 0.143, Plotly 7.1, NumPy 2.5, pandas 3.0, Node/Next.js 16.4.

## Interface notes
- **Regime ribbon:** a strip on the Dashboard and Simulation pages (and a mini one in the sidebar) shows the market regime of every step
  played (Bear / Neutral / Bull / HighVol) with a playhead; the same colours shade every chart. Colour in the interface is reserved for
  data (agents, regimes, assets); everything else is neutral.
- **Light and dark themes** (Auto follows the system) with charts re-coloured to match; responsive down to phone width.
- **Your session is kept in this browser:** the agents you trained, the episode you were on and (for logs up to ~1 MB) the learning
  curves survive a reload. *Reset session* in the sidebar forgets them. Nothing is stored on the server.
- **Keyboard on the Simulation page:** Space play/pause, Right arrow one step, R reset.
- **Dashboard scoreboard:** "Where the learning ended up" plots return against risk for experiments A-D from the precomputed reference run.

## What is in the dashboard

| Page | What you can do (all backed by the real engine, nothing mocked) |
|---|---|
| **Dashboard** | KPI cards (portfolio value, return, current regime, best agent, drawdown), value chart with regime shading, allocation donut / bars, per-agent performance table. Pick a market seed and run a full episode. |
| **Simulation** | **Start / Pause / Step / Step xN / Run to end / Reset** on the real environment. Per-agent policy (trained Q-greedy, random, or a fixed/manual basket). Views: overview, *How agents decide* (state, Q-values compared, chosen basket, reward), *Money flow* (allocation % -> rupees -> asset return -> profit/loss -> reward breakdown) and a downloadable step log. |
| **RL Lab** | Set episodes, seed, alpha, gamma, epsilon start/floor/decay, initial Q; **Train from scratch / Continue training**. Learning curves (reward, epsilon, TD error, capital), a Q-table inspector for any of the 108 states, policy and visit heat-maps, a step-by-step Q-update walkthrough checked against the formula, and an out-of-sample evaluation button. |
| **Portfolio Analytics** | Growth, cumulative return, volatility, rolling volatility, drawdown, Sharpe, transaction costs, profit by asset, agent comparison - for the episode in the simulator or for many held-out episodes (box plots, mean +/- std curves). CSV export. |
| **Experiments** | A Random, B Fixed basket, C Single-agent Q, D 4-agent Independent Q. Train on one seed, **evaluate out-of-sample** on a disjoint seed block with exploration off and Q-tables frozen. Live runs plus a clearly-labelled precomputed 5000-episode reference. JSON export. |
| **Learn** | Plain-language MDP vs Markov game, the regime Markov chain (with the stationary distribution and a regime simulator), the 108-state encoding calculator, the 7 baskets, the reward formula calculator and a Q-update calculator. |

## Results (precomputed reference, synthetic market)

Produced by `python -m scripts.build_reference` (real run; 5,000 training episodes on seed 42; 50 out-of-sample episodes on
seeds 51-100; epsilon = 0, Q-tables frozen). Metrics are for the combined Rs 10,00,000 portfolio, mean over the 50 held-out
episodes. Sharpe is **episode-scale, not annualised**.

| Experiment | Return (mean) | Return std | Episodes with a loss | Volatility / step | Max drawdown | Sharpe | Trading costs (Rs) | Specialization |
|---|---|---|---|---|---|---|---|---|
| A Random | -3.65% | 5.99% | 78% | 0.572% | 7.86% | -0.97 | 105,155 | 0.024 |
| B Fixed (Balanced basket) | +9.05% | 8.09% | 12% | 0.701% | 5.10% | 0.99 | 3,774 | 0.000 |
| C Single-agent Q | +1.33% | 1.70% | 22% | 0.164% | 1.47% | -0.44 | 30,941 | 0.000 |
| D 4-agent Independent Q | +3.18% | 1.32% | 0% | 0.125% | 0.67% | 0.88 | 877 | 0.000 |

### Honest reading
- Learning **works mechanically**: Q-values move, exploration decays, and the learned policies earn a positive, low-risk
  return where the random policy loses money.
- But the learned agents **do not beat a plain fixed Balanced basket** on return (+3.18% vs +9.05%); they win only on
  risk (volatility, drawdown, 0% losing episodes).
- **All four D agents converge to the same cash-heavy policy** (specialization 0.000) even though their reward weights differ.
  Plain epsilon-greedy tabular Q-learning with transaction costs does not discover the sustained-commitment strategies that
  would favour the risk-seeking role. This is a limitation of the baseline learner, reported rather than hidden; reward-weight
  Monte-Carlo checks in [docs/ENGINE_README.md](docs/ENGINE_README.md) show the reward function itself does encode role
  differences. Because the sleeves behave identically, the four lines on the dashboard overlap - the app says so.
- The market is synthetic, so none of this speaks to real markets.

> Note: the dashboard measures return from the **true initial capital**; the older script printout
> (`python -m experiments.run_experiments`) measures from the first post-step capital, so numbers differ slightly (D: +3.18% vs +3.20%).
> Experiment A uses a fresh, seeded random stream at evaluation, so it differs slightly from the older printout too.

## How it works (short version)
- **Market:** 4 regimes (Bear, Neutral, Bull, HighVol) following a Markov chain; per-regime Normal returns per asset.
- **Agents:** one Q-table (108 states x 7 actions) per agent, updated with
  `Q(s,a) <- Q(s,a) + alpha [ r + gamma max_a' Q(s',a') - Q(s,a) ]`, epsilon-greedy with per-episode decay. No deep RL.
- **State:** regime x equity bucket x bond bucket x gold bucket (Low <20%, Medium 20-45%, High >=45%).
- **Actions:** 7 predefined baskets from AggressiveGrowth (70/20/10/0) to CashHeavy (0/10/10/80).
- **Reward:** `alpha*log-return - beta*risk - lambda*trading cost - delta*new drawdown + mu*cash weight` (per-role weights in
  `config.py`), scaled by 100 and clipped to +/-5.
- **Interaction:** agents are coupled through a one-sided crowding penalty on the equity return when their capital-weighted
  equity weight exceeds 40%. The regime path and base return draws are exogenous (same for every policy on a seed).
- **No look-ahead; train/eval separation:** training uses one seed; evaluation uses fresh environments on disjoint seeds with
  no learning. The experiment engine and the UI both refuse an evaluation block that contains the training seed.

## Repository layout
```
app/ components/ lib/       Next.js front end (pages: Dashboard, Simulation, RL Lab, Analytics, Experiments, Learn; theme, saved session)
api/index.py                Vercel Python entry point (imports the FastAPI app)
server/                     app.py (endpoints), trainstate.py (portable training state), blocks.py, pages.py (UI blocks)
dashboard/                  session.py (replayable simulation), charts.py (Plotly), views.py, theme.py, reference.py, lab_tools.py
config.py                   single source of truth for all parameters
environment/ agents/ rl/    market, portfolio, FinancialEnv, Q-learning agent and update rule (engine)
training/ evaluation/       trainer; metrics.py and analytics.py
experiments/                baseline agents, original runner, and engine.py (reproducible A-D API)
scripts/build_reference.py  regenerates data/reference_results.json + reference_qtables.npz from a real run
data/                       precomputed reference (clearly labelled in the UI)
tests/                      pytest suite (+ tests/e2e Playwright browser tests)
vercel.json  .python-version  requirements.txt  package.json  next.config.mjs
docs/ENGINE_README.md       detailed engine write-up: bug history, analysis, limitations
DEPLOYMENT.md               Vercel deployment steps
```
The earlier Streamlit version was removed; it is recoverable from commit `2452936`.

## Reproducibility
Everything is seeded. `RunConfig` + seeds fully determine a run (tests assert that the same config gives identical
results). Default seeds: train 42, evaluation 51-100. The shipped Q-tables are checked by a test to reproduce the stored
experiment-D results exactly.

## Tests
```bash
pip install -r requirements-dev.txt
pytest -q                                   # engine, analytics, session, charts, reference, API, train-state
npm run typecheck && npm run build          # front end
# browser tests (need the app running, see tests/e2e/test_browser.py):
E2E_BASE_URL=http://localhost:3000 pytest tests/e2e -q
```
What was actually run in the authoring environment for the redesign commit: `pytest -q` -> **138 passed, 15 skipped** (the 15 skipped are the
browser tests when no server is configured); the **15 Playwright browser tests passed** against a local uvicorn + `next start`
(they cover every page, simulation controls and keyboard shortcuts, RL Lab training / continue / inspect / walkthrough / evaluate / restore,
analytics, experiments, learn calculators, theme persistence, reload persistence and a no-horizontal-scroll check at phone width);
`tsc --noEmit` and `next build` passed.

## Known limitations
- Tabular Q-learning, 7 predefined baskets and coarse buckets: simple by design, no generalisation across states.
- No role specialization emerges (see above). Shared / hybrid rewards, a shared capital pool, reward ablations and hyper-parameter
  sweeps are **not** implemented.
- Live training in the app is capped (3,000 episodes in the RL Lab, 50 episodes per request) because Vercel functions are time-limited.
  Trained agents are kept in this browser (localStorage), not on the server: another browser or device starts from the precomputed
  reference agents.
