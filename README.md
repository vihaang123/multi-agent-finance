# Multi-Agent Reinforcement Learning for Financial Resource Allocation

A university RL + Multi-Agent Systems mini project with a working **Streamlit dashboard**: four independent
tabular Q-learning agents (Growth, Conservative, Balanced, RiskLiquidity) split Rs 10,00,000 across
**Equity / Bonds / Gold / Cash** inside one shared, simulated market.

> **All data is synthetic.** Prices, returns and market regimes come from a simulated Markov-regime model
> (`environment/market.py`). Nothing here is real market data, investment advice or a prediction, and results are not
> real-world performance. The project has **no** real-money trading and **no** broker connection.

**Status:** code is in this repository; tests listed below were actually run. The app has **not** been deployed to
Streamlit Community Cloud from the authoring environment (it needs your GitHub login) - see
[DEPLOYMENT.md](DEPLOYMENT.md) for the exact steps. There is no live URL yet.

---

## Quick start

```bash
git clone https://github.com/vihaang123/multi-agent-finance.git && cd multi-agent-finance
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py                            # http://localhost:8501
```
Tested with Python 3.13, Streamlit 1.65, Plotly 7.1, NumPy 2.5, pandas 3.0, matplotlib 3.11.

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
streamlit_app.py            entry point (navigation, theme, synthetic-data banner)
dashboard/                  pages/, charts.py (Plotly), session.py (start/pause/step/reset), views.py, state.py, theme.py, reference.py, lab_tools.py
config.py                   single source of truth for all parameters
environment/ agents/ rl/    market, portfolio, FinancialEnv, Q-learning agent and update rule (engine)
training/ evaluation/       trainer; metrics.py and analytics.py
experiments/                baseline agents, original runner, and engine.py (reproducible A-D API used by the app)
scripts/build_reference.py  regenerates data/reference_results.json + reference_qtables.npz from a real run
data/                       precomputed reference (clearly labelled in the UI)
tests/                      pytest suite incl. Streamlit AppTest UI tests
docs/ENGINE_README.md       detailed engine write-up: bug history, analysis, limitations
DEPLOYMENT.md  Dockerfile  requirements.txt  requirements-dev.txt  .streamlit/config.toml
```

## Reproducibility
Everything is seeded. `RunConfig` + seeds fully determine a run (tests assert that the same config gives identical
results). Default seeds: train 42, evaluation 51-100. The shipped Q-tables are checked by a test to reproduce the stored
experiment-D results exactly.

## Tests
```bash
pip install -r requirements-dev.txt
pytest -q                      # engine, analytics, session, charts, reference integrity, UI (AppTest)
```
At the time of writing all 134 tests passed in a fresh virtual environment built from `requirements-dev.txt`
(Python 3.13; 40 original engine tests + 63 new engine/session/chart/reference tests + 31 UI tests).
The UI tests execute the real app script, click the real buttons and check the resulting state; they do not replace
trying the app yourself in a browser.

## Known limitations
- Tabular Q-learning, 7 predefined baskets and coarse buckets: simple by design, no generalisation across states.
- No role specialization emerges (see above). Shared / hybrid rewards, a shared capital pool, reward ablations and hyper-parameter
  sweeps are **not** implemented.
- Live training in the app is capped (3,000 episodes, 150 s) to suit free hosting.
- Docker image is provided but was not built in the authoring environment; Community Cloud deployment has not been performed.
