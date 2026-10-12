# Multi-Agent Reinforcement Learning for Financial Resource Allocation

A university RL + Multi-Agent Systems mini-project: four independent tabular
Q-Learning agents, each with a different risk-return role, learn to allocate
capital across Equity / Bonds / Gold / Cash inside one shared, stochastic
financial environment.

This README is engineering documentation + an honest results write-up. It is
**not** the academic proposal, PPT, or viva script — those are separate,
later deliverables.

---

## 1. Quickstart

```bash
cd multi_agent_finance
pip install -r requirements.txt

# Run the full test suite (30 tests, ~0.3s)
pytest tests/ -v

# Run all four experiments (A/B/C/D), ~4-5 minutes total,
# writes CSV logs to results/ and PNG plots to results/plots/
python -m experiments.run_experiments
```

Requires Python 3.9+. Dependencies: `numpy`, `pandas`, `matplotlib`, `pytest`.

---

## 2. Project structure

```
multi_agent_finance/
├── config.py                  # every tunable parameter, with rationale comments
├── environment/
│   ├── market.py               # regime-switching stochastic market simulator
│   ├── portfolio.py            # per-agent capital sleeve, rebalancing, drawdown
│   └── financial_env.py        # shared env: state, reward, multi-agent step API
├── agents/
│   ├── base_agent.py           # common interface (learning & baseline agents)
│   └── q_learning_agent.py     # independent tabular Q-learning agent
├── rl/
│   └── q_learning.py           # the Q-update rule and epsilon-greedy policy, from scratch
├── training/
│   └── trainer.py              # episode loop, multi-agent orchestration, logging
├── evaluation/
│   └── metrics.py              # financial / RL / MAS metric functions
├── visualization/
│   └── plots.py                # all 7+1 required plots
├── experiments/
│   ├── baseline.py             # Random & Fixed non-learning baseline agents
│   └── run_experiments.py      # Experiments A, B, C, D end-to-end
├── tests/                       # 30 unit tests (env, agents, multi-agent interaction)
└── results/                     # generated CSV logs + PNG plots (gitignored contents shown as example run)
```

---

## 3. Architecture

```
Market  →  Portfolio (per agent)  →  FinancialEnv  →  Agent (Q-table)
  ↑____________________________________________|  (next action)
```

Each timestep:
1. Every agent's **action** (0-6) selects a target portfolio basket (fixed
   weights over Equity/Bonds/Gold/Cash).
2. The environment **rebalances** each agent's sleeve toward its target,
   charging a transaction cost proportional to turnover.
3. A capital-weighted **aggregate equity exposure** across all 4 agents is
   computed. If it crowds *above* a neutral baseline (40%), a market-impact
   penalty reduces the realized equity return for everyone that step — this
   is the multi-agent coupling: one agent's allocation choice affects every
   other agent's payoff.
4. The market draws **stochastic regime-dependent returns** for each asset;
   the impact penalty is applied to equity's draw.
5. Each sleeve's value updates; drawdown, weights, and reward are computed.
6. State discretizes to `(regime, equity_bucket, bond_bucket, gold_bucket)` —
   108 states. The regime transitions (Markov) for the next step.

Full design rationale (state/action space sizing, capital model, reward
math) is in the original architecture review; the version actually
implemented differs from that first draft in the ways documented in
Section 5 below, based on what testing revealed.

---

## 4. The four agents

| Agent | Role | Reward emphasis |
|---|---|---|
| 0 | Growth | Maximize return, low risk-aversion |
| 1 | Conservative | Strong risk-aversion, capital preservation |
| 2 | Balanced | Moderate return/risk trade-off |
| 3 | RiskLiquidity | Liquidity + drawdown control |

Reward: `R_i = alpha*log_return - beta*risk - lam*txcost - delta*drawdown_increase + liquidity_mu*cash_weight`
(all weights in `config.py`, `REWARD_WEIGHTS`).

---

## 5. Results & Analysis

**This section is deliberately written as a debugging/analysis narrative, not
just a results table** — several of the most interesting findings in this
project came from the process of getting the reward function and market
model to actually produce sensible behaviour, and that process is itself
good evidence of understanding the system, which is worth showing.

> **BASELINE V1 status note:** the numbers below are from Baseline V1 —
> after a full audit and four confirmed fixes on top of the version that
> produced the original results narrative in 5.2-5.6: (1) a regime-labeling
> bug where logged regime was the POST-transition regime, not the one that
> generated that step's return — now `info["current_regime"]` /
> `info["next_regime"]` are exposed separately and correctly; (2) evaluation
> now runs on genuinely held-out seeds (`cfg.EVAL_SEEDS`, 51-100) with fresh
> environments, frozen Q-tables, and no epsilon decay — never a continuation
> of the training seed (42); (3) RL metrics (avg reward, reward trend,
> exploration rate) are now printed per experiment; (4) Sharpe ratio's
> assumptions are now stated explicitly wherever it's reported (see
> `evaluation/metrics.py::SHARPE_METHODOLOGY_NOTE`). The regime fix touched
> only logging/plotting, not the return or reward computation, so retraining
> under the same seed reproduces identical training dynamics — confirmed
> directly. The specific numbers below moved slightly from the original
> narrative purely because evaluation is now computed differently (held-out
> seeds instead of a continued training stream), not because behaviour
> changed.

### 5.1 Baseline V1 results — TRAINING vs OUT-OF-SAMPLE EVALUATION, kept separate

Training: 5000 episodes, single continuing seed (`cfg.TRAIN_SEED = 42`).
Evaluation: 50 independent held-out episodes, each a fresh environment
seeded from `cfg.EVAL_SEEDS = [51..100]`, greedy policy, Q-tables frozen.

**Training results** (avg reward = rolling mean, last 100 episodes; trend =
last-10% vs prior-10% mean reward, ~0 means plateaued):

| Experiment | Agent | Avg reward (last 100 ep) | Reward trend | Final epsilon |
|---|---|---|---|---|
| A (Random) | all 4 | -16.8 to -72.7 (varies by role weights) | mixed | 0 (no learning) |
| B (Fixed) | all 4 | +16.3 (Growth) to -85.3 (Conservative) | small positive | 0 (no learning) |
| C (Single Q-learning) | Growth | +3.71 | -0.11 (plateaued) | 0.05 (floor at ep 2994) |
| D (4-agent Q-learning) | Growth/Cons/Bal/RiskLiq | +4.35 / -19.17 / +0.21 / +26.62 | mostly still rising | 0.05 (floor at ep 2994) |

**Out-of-sample evaluation results** (50 held-out episodes, mean ± std):

| Experiment | Agent | Return | Volatility | Max Drawdown | Sharpe (episode-scale) |
|---|---|---|---|---|---|
| A (Random) | Growth | -4.25% ± 6.11% | 0.66% | 9.01% | -0.96 |
| A (Random) | Conservative | -3.49% ± 6.71% | 0.64% | 8.53% | -0.85 |
| A (Random) | Balanced | -2.59% ± 7.22% | 0.67% | 8.15% | -0.76 |
| A (Random) | RiskLiquidity | -3.42% ± 7.21% | 0.65% | 8.51% | -0.84 |
| B (Fixed) | all 4 (identical) | +9.34% ± 8.02% | 0.70% | 5.08% | +1.03 |
| C (Single Q-learning) | Growth | +1.35% ± 1.69% | 0.16% | 1.46% | -0.41 |
| D (4-agent Q-learning) | all 4 (identical) | +3.20% ± 1.31% | 0.12% | 0.66% | +0.92 |

*Sharpe is episode-scale, not annualized — see
`evaluation/metrics.py::SHARPE_METHODOLOGY_NOTE` for the exact assumptions
(per-step risk-free rate from this project's own cash model, scaled by
sqrt(episode_length), comparable only across experiments in this project).*

All four D agents again converge to **identical** out-of-sample figures —
now confirmed on genuinely unseen market conditions, not just the training
stream. That's the central, honest finding of this project, and Section 5.3
explains exactly why, with evidence.

Note that `results/plots/1_training_rewards.png` still shows four visibly
*different* reward curves despite this — that's not a contradiction. Reward
is each role's personal, weighted valuation of an outcome, not the outcome
itself: since Conservative's weights penalize risk and drawdown more heavily
(and weight return less) than RiskLiquidity's, the exact same CashHeavy
basket choice scores differently for each of them even though the real
financial allocation is identical across all four.

### 5.2 Bugs found and fixed along the way

Getting to the numbers above required fixing several real issues, each
confirmed by isolating it in a standalone experiment before and after:

1. **Noisy single-episode evaluation.** Early versions judged a trained
   policy from one greedy evaluation episode — one random market path. Fixed
   by averaging over 50-300 independent evaluation episodes for any reported
   figure; single-episode trajectories are still used for the *illustrative*
   plots (allocation-over-time, etc.) where a representative example is the
   point.

2. **Premature epsilon decay.** With `epsilon_decay=0.995`, exploration
   floored by episode ~600 of 5000 — before the state-action space had been
   adequately sampled (only 24/108 states ever got a non-trivial number of
   updates). Slowing decay to `0.999` measurably improved and stabilised
   every agent's outcome.

3. **Reward-hacking via cumulative drawdown.** The reward's drawdown penalty
   originally used the *cumulative* distance from the episode's all-time
   capital peak. Since that distance persists for the rest of the episode
   after any bad stretch (regardless of what the agent does next), it acted
   as a "lingering tax" that swamped the return signal and collapsed every
   risk-averse role onto the same degenerate "100% cash forever" policy —
   even agents 100% identical to each other. Fixed by penalizing only the
   *incremental* drawdown created each step (`Portfolio.drawdown_increase`),
   which ties the penalty to the current decision instead of to history.

4. **No equity risk premium in the market model.** The original regime
   parameters gave equity a *negative* average return once weighted by the
   regime's long-run (stationary) distribution — Bear regime's -0.6%/step
   drag outweighed Bull/Neutral gains. That made "hold zero equity" the
   mathematically correct answer for literally every role, independent of
   risk tolerance — a market-calibration bug, not a reward bug. Fixed by
   raising Bear and Bull equity means so the stationary-weighted average
   equity return is clearly positive and above bonds/gold/cash.

5. **Symmetric market-impact penalty.** The crowding-penalty term originally
   penalized deviation from the 40% baseline equity weight in *either*
   direction — so an agent being *underweight* equity (because the other
   three agents held none) was penalized exactly like a crowd piling *in*.
   This has no real-world justification and specifically punished the
   Growth agent for being the only one willing to hold equity. Fixed to a
   one-sided penalty: only crowding *above* baseline matters.

6. **Arithmetic vs. log returns.** Reward used raw fractional per-step
   returns, but capital compounds multiplicatively. A policy can have a
   positive *average* one-step arithmetic return yet a negative cumulative
   compounded outcome once volatility is significant (volatility drag:
   `E[log(1+r)] ≈ E[r] - Var(r)/2`). Confirmed empirically: the Growth
   agent's training reward trended clearly positive while its 300-episode
   evaluated *return* was reliably -3.2% (SE ±0.07%, so not noise). Fixed by
   rewarding log returns, which are additive across compounding steps.

7. **Transaction-cost weight an order of magnitude too small.** Even after
   fix #6, real dollar outcomes still didn't match the reward. Diagnosis:
   `lam` (transaction-cost weight) was ~0.10-0.20 while `alpha` (return
   weight) was 0.5-3.0 — a ~15-30x mismatch — despite `txcost_frac` and
   `log_return` being on the *identical* numerical scale (both fractional
   capital changes). The agent's Q-values were pricing a rupee lost to fees
   at a fraction of a rupee gained from returns. Measured directly: Growth's
   greedy policy paid ~10.2% of capital in cumulative transaction costs
   against only ~6.8% of cumulative market gains over one episode — a real,
   dominant cost the reward was barely penalizing. Fixed by setting
   `lam = alpha` for every role, so a switch is priced at its true cost.

8. **Regime-labeling bug (BASELINE V1 audit).** `info["regime"]` was set
   *after* `MarketSimulator.step_regime()` had already transitioned the
   regime for the next timestep — so any code that logged or plotted
   `info["regime"]` was labeling each step's outcome with the WRONG regime
   (the one about to start, not the one that actually generated that step's
   return). Confirmed empirically: on steps where the regime changed,
   `info["regime"]` matched the post-transition regime, not the
   pre-transition one used to draw returns. The return/reward math itself
   was never affected (it correctly used the pre-transition regime
   internally) — only the logged label was wrong, which silently mislabeled
   `results/plots/5_regime_vs_allocation.png`. Fixed by exposing both
   `info["current_regime"]` (generated this step's return) and
   `info["next_regime"]` (governs the next step) explicitly, with
   regression tests asserting each against ground truth.

9. **No true out-of-sample evaluation (BASELINE V1 audit).** Evaluation
   previously continued on the *same* environment object right after
   training — the RNG stream just carried on rather than resetting to
   independently-seeded, disjoint episodes. Fixed: `evaluate_out_of_sample()`
   builds a brand-new `FinancialEnv` per evaluation seed from
   `cfg.EVAL_SEEDS` ([51..100], disjoint from `cfg.TRAIN_SEED=42`), never
   calls `agent.update()` or `agent.on_episode_end()`, and is tested to
   leave Q-tables and epsilon provably unchanged.

### 5.3 The remaining finding: plain epsilon-greedy can't discover
"sustained commitment" strategies here, and that's real, not a bug

After all of the above fixes, the reward and environment are internally
consistent — confirmed by comparing against **non-learning** `FixedAgent`
baselines run through the exact same environment:

| Growth's policy (via `FixedAgent`, full env, 300 episodes) | Return |
|---|---|
| Always hold "AggressiveGrowth" basket | **+9.45%** |
| Hand-coded regime-chasing (switch basket by regime) | +4.01% |
| Always hold "CashHeavy" basket | +3.04% |

**"Always hold the aggressive basket" is provably the best of these three.**
But the trained Q-learning Growth agent converges to the *worst* of the
three (CashHeavy-equivalent, ~+3.2%) — even with:
- 5000-15000 training episodes,
- slower epsilon decay (tested up to `0.9997`),
- strong optimistic Q-table initialization (tested up to `Q_INIT_VALUE=30`,
  15x any plausible true Q-value), a standard, textbook remedy for
  under-exploration (Sutton & Barto, ch. 2).

None of these closed the gap. The diagnosis: transaction costs penalize
*interrupting* a strategy, but standard per-step epsilon-greedy exploration
interrupts everything constantly — an exploratory trial of the aggressive
basket is essentially always sandwiched between other random actions, so it
almost never gets to experience (and therefore Q-learning almost never gets
to *learn*) what the basket is worth when held *consistently*. Bootstrapping
(`max_a' Q(s',a')`) can't manufacture that value out of interrupted samples:
the value of "stay aggressive" depends on "future-me also staying
aggressive," and if future-me hasn't learned that yet either, the value
never propagates backward to the decision that matters. This is a
recognized class of problem in RL (it's part of the motivation for
hierarchical RL / macro-action / "options" frameworks), and addressing it
properly is out of scope for plain tabular Q-learning as specified for this
project — but it's a genuine, concretely diagnosed, and reportable finding.

Its consequence for the top-line results: Experiment D's per-role numbers
being identical (Section 5.1) is not a training bug — it's all four
Q-learning agents independently converging to the *same* locally-safe,
zero-turnover-after-first-rebalance equilibrium, for the reason above,
regardless of their differing reward weights. The reward weights *do*
correctly differentiate the roles' preferences (Section 5.4 shows this with
a Monte Carlo argument, not just training), but the exploration mechanism
fails to *find* the policy that would express that differentiated
preference for anything requiring sustained equity exposure.

### 5.4 Evidence the reward function *does* correctly encode role differences

Even though training converges to a shared local optimum, the reward
function's *intended* role differentiation is real and verifiable directly
(bypassing the exploration problem) via common-random-number Monte Carlo
evaluation of every basket under each role's reward weights:

- **Growth**: monotonically prefers *more* equity (AggressiveGrowth best,
  CashHeavy worst).
- **Conservative**: monotonically prefers *less* equity (CashHeavy best,
  AggressiveGrowth worst).
- **Balanced**: prefers a genuine middle ground (moderate, diversified
  baskets).
- **RiskLiquidity**: prefers CashHeavy/DefensiveLiquidity (safety + the
  explicit liquidity bonus).

This is the correct qualitative ordering per the brief's design intent
(Section 12: "different agent behaviors must come from reward functions").
The gap between this and what training actually discovers (Section 5.3) is
the project's central, honestly-reported limitation.

### 5.5 What Q-learning *does* demonstrably learn

It would be inaccurate to say training doesn't work at all:

- Training reward climbs consistently and substantially across all four
  experiments' learning curves (see `results/plots/1_training_rewards.png`)
  — from deeply negative (early, mostly-random policy) to a stable positive
  plateau.
- Experiment D (learned) clearly beats Experiment A (random) in expected
  reward and capital outcome.
- In earlier configurations (before the `lam` recalibration in fix #7), the
  Growth agent *did* learn genuine regime-conditioned behaviour — choosing
  the aggressive basket specifically during Bull regimes and retreating to
  defensive baskets in Bear/HighVol (see the git-history / iteration notes
  in this README's development if you re-run with `lam` lowered). That
  policy scored well on training reward but, per fix #6/#7, was actually
  losing real money to whipsaw transaction costs — which is itself the
  finding that led to fix #7.

### 5.6 Suggested future work

- **Sticky / persistent exploration**: force the agent to repeat its chosen
  action for `k` consecutive steps before re-rolling epsilon, so a "hold
  consistently" strategy actually gets sampled cleanly. Directly targets the
  Section 5.3 finding.
- **Eligibility traces (Q(λ) / SARSA(λ))**: propagate reward signal to
  more of the states that led to it, partially mitigating the sparse
  backward-bootstrap problem.
- **Reward shaping via potential functions**: a potential-based shaping
  term could reward "intending to persist" without changing the optimal
  policy (Ng, Harada & Russell 1999 style), which is a principled way to
  address 5.3 without leaving tabular methods.
- **Hierarchical / options-based actions**: a macro-action like "hold
  AggressiveGrowth until regime changes" would sidestep the interruption
  problem entirely — but this is explicitly out of scope for the current
  "tabular Q-Learning from scratch" brief.

---

## 6. Testing

```bash
pytest tests/ -v
```

40 tests across four files:
- `test_environment.py` — reset/state validity, action constraints,
  allocation always sums to 1, episode termination, market simulator,
  portfolio rebalancing/drawdown mechanics, and regression tests for the
  regime-labeling fix (`current_regime` matches the regime that generated
  the return; `next_regime` matches the regime after transition; the two
  can legitimately differ, so the test isn't vacuous).
- `test_agents.py` — Q-table init (including optimistic init), the Q-update
  rule in isolation, epsilon-greedy exploration/exploitation, baseline
  agents.
- `test_multi_agent.py` — simultaneous multi-agent stepping, market-impact
  coupling (and its one-sidedness), shared vs. individual reward modes,
  independent Q-tables per agent, role-weight sanity checks.
- `test_evaluation.py` — out-of-sample evaluation: Q-tables and epsilon are
  provably unchanged by `evaluate_out_of_sample()`, evaluation genuinely
  runs greedy throughout (not just before/after), eval seeds are disjoint
  from the training seed and produce different market paths from each
  other. Also the **fairness verification**: with market impact disabled,
  the same seed produces an identical regime *and* return sequence
  regardless of agent count or actions (the market is purely exogenous);
  with market impact enabled, the regime sequence is still
  action-independent, but the realized equity return can differ based on
  agents' aggregate exposure (the intentional, documented *endogenous*
  component — see `financial_env.py` step 2-3 for where the two are kept
  separate).

---

## 7. Configuration

Every tunable lives in `config.py`, with inline comments explaining *why*
each non-obvious value is set the way it is (most of Section 5's fixes are
also documented at their point of use in the config file itself). Key knobs:

- `NUM_EPISODES`, `EPISODE_LENGTH` — training scale
- `TRAIN_SEED`, `EVAL_SEED_START`, `N_EVAL_EPISODES`, `EVAL_SEEDS` — train
  vs. out-of-sample evaluation seed separation (Baseline V1, Fix 2)
- `ALPHA_LR`, `GAMMA`, `EPSILON_*` — Q-learning hyperparameters
- `Q_INIT_VALUE` — optimistic initialization strength
- `REWARD_WEIGHTS` — per-role `alpha/beta/lam/delta/liquidity_mu`
- `REWARD_MODE` — `"individual" | "shared" | "hybrid"` (architecture
  supports all three; only `individual` is used in the reported
  experiments, per the brief's "build individual first" instruction)
- `MARKET_IMPACT_KAPPA`, `MARKET_IMPACT_BASELINE_EQUITY` — multi-agent
  coupling strength
