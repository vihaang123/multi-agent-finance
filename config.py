"""
config.py

Single source of truth for every tunable parameter in the project.
Nothing below should be hard-coded elsewhere -- import from here instead.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Train / evaluation seed separation (BASELINE V1, Fix 2)
# ---------------------------------------------------------------------------
# TRAIN_SEED seeds the single, continuing-stream training environment used by
# Trainer.train() -- unchanged from before. EVAL_SEEDS is a disjoint block of
# seeds, each used to construct a BRAND NEW, independent FinancialEnv for one
# held-out evaluation episode (see experiments/run_experiments.py ::
# evaluate_out_of_sample()). Because every eval seed starts a fresh
# np.random.default_rng() stream, evaluation episodes never overlap with, or
# continue from, anything the agent saw during training.
#
# Ranges follow the brief's own example (train ~1-50, eval ~51-100): 42 sits
# inside [1, 50] and EVAL_SEEDS = [51..100] is guaranteed disjoint from it.
TRAIN_SEED = SEED  # 42
EVAL_SEED_START = 51
N_EVAL_EPISODES = 50
EVAL_SEEDS = list(range(EVAL_SEED_START, EVAL_SEED_START + N_EVAL_EPISODES))  # [51..100]
assert TRAIN_SEED not in EVAL_SEEDS, "Train and evaluation seeds must be disjoint."

# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------
ASSET_NAMES = ["Equity", "Bonds", "Gold", "Cash"]
N_ASSETS = len(ASSET_NAMES)
EQUITY, BONDS, GOLD, CASH = 0, 1, 2, 3

# ---------------------------------------------------------------------------
# Market regimes
# ---------------------------------------------------------------------------
REGIME_NAMES = ["Bear", "Neutral", "Bull", "HighVol"]
N_REGIMES = len(REGIME_NAMES)
BEAR, NEUTRAL, BULL, HIGHVOL = 0, 1, 2, 3

# Regime transition matrix (row = current regime, col = next regime).
# Regimes are persistent (diagonal-heavy) but do switch.
REGIME_TRANSITION = np.array([
    # Bear   Neutral  Bull   HighVol
    [0.55,   0.25,    0.05,  0.15],   # Bear
    [0.15,   0.55,    0.20,  0.10],   # Neutral
    [0.05,   0.20,    0.65,  0.10],   # Bull
    [0.20,   0.20,    0.20,  0.40],   # HighVol
])

# Per-regime, per-asset (mean daily-equivalent return, volatility(std))
# Cash always has a small near-zero, near-riskless return.
#
# CALIBRATION NOTE: these mu values were originally set with Bear equity mu
# = -0.0060, which -- combined with the transition matrix's stationary
# distribution (Bear occurs ~21% of the time long-run) -- gave equity a
# NEGATIVE average return overall (~-0.07%/step), i.e. no equity risk
# premium at all. That made "hold zero equity" the mathematically correct
# reward-maximizing choice for every role regardless of risk tolerance --
# not a reward-shaping bug, a market-model bug. Bear's equity mu was raised
# (still negative, bear markets should still hurt) and Bull's raised further,
# so the stationary-weighted average equity return is clearly positive and
# above bonds/gold/cash, restoring a genuine risk-return tradeoff for the
# reward weights to differentiate roles on. See README, Results & Analysis.
ASSET_RETURN_PARAMS = {
    BEAR:    {EQUITY: (-0.0020, 0.020), BONDS: (0.0015, 0.006), GOLD: (0.0020, 0.010), CASH: (0.0002, 0.0005)},
    NEUTRAL: {EQUITY: ( 0.0008, 0.010), BONDS: (0.0008, 0.005), GOLD: (0.0003, 0.007), CASH: (0.0002, 0.0005)},
    BULL:    {EQUITY: ( 0.0030, 0.012), BONDS: (0.0005, 0.005), GOLD: (0.0002, 0.007), CASH: (0.0002, 0.0005)},
    HIGHVOL: {EQUITY: ( 0.0010, 0.028), BONDS: (0.0004, 0.012), GOLD: (0.0010, 0.018), CASH: (0.0002, 0.0005)},
}

# NOTE: risk exposure for the reward function is NOT a static per-asset score
# (an earlier version used one and it created a scale mismatch -- see
# REWARD_WEIGHTS note below). It is computed dynamically from the current
# regime's actual sigma via MarketSimulator.sigma_vector(), in financial_env.py.

# ---------------------------------------------------------------------------
# Capital
# ---------------------------------------------------------------------------
TOTAL_INITIAL_CAPITAL = 1_000_000.0  # Rs 10,00,000
N_AGENTS = 4
INITIAL_CAPITAL_PER_AGENT = TOTAL_INITIAL_CAPITAL / N_AGENTS  # Rs 2,50,000 each

# ---------------------------------------------------------------------------
# Action space: predefined portfolio baskets (target weights, sum to 1.0)
# Order: [Equity, Bonds, Gold, Cash]
# ---------------------------------------------------------------------------
ACTION_BASKETS = np.array([
    [0.70, 0.20, 0.10, 0.00],   # 0: Aggressive Growth
    [0.55, 0.25, 0.10, 0.10],   # 1: Growth Tilt
    [0.40, 0.30, 0.20, 0.10],   # 2: Balanced
    [0.25, 0.35, 0.20, 0.20],   # 3: Balanced Defensive
    [0.15, 0.35, 0.15, 0.35],   # 4: Conservative
    [0.05, 0.25, 0.10, 0.60],   # 5: Defensive / Liquidity
    [0.00, 0.10, 0.10, 0.80],   # 6: Cash Heavy
])
ACTION_BASKET_NAMES = [
    "AggressiveGrowth", "GrowthTilt", "Balanced",
    "BalancedDefensive", "Conservative", "DefensiveLiquidity", "CashHeavy",
]
N_ACTIONS = len(ACTION_BASKETS)

assert np.allclose(ACTION_BASKETS.sum(axis=1), 1.0), "Each basket must sum to 1.0"

# ---------------------------------------------------------------------------
# State discretization
# ---------------------------------------------------------------------------
N_ALLOC_BUCKETS = 3  # Low / Medium / High, applied to Equity, Bonds, Gold weights
ALLOC_BUCKET_EDGES = np.array([0.20, 0.45])  # 2 edges -> 3 buckets: [0,0.2) [0.2,0.45) [0.45,1]

N_STATES = N_REGIMES * (N_ALLOC_BUCKETS ** 3)  # 4 * 27 = 108

# ---------------------------------------------------------------------------
# Transaction costs & market impact
# ---------------------------------------------------------------------------
TRANSACTION_COST_RATE = 0.0015          # fraction of turnover value charged as cost
MARKET_IMPACT_KAPPA = 0.05              # strength of aggregate crowding penalty
MARKET_IMPACT_BASELINE_EQUITY = 0.40    # "neutral" aggregate equity weight

# ---------------------------------------------------------------------------
# Reward function weights per agent role
# R_i = alpha*return - beta*risk - lambda*txcost - delta*drawdown (+ liquidity bonus)
# ---------------------------------------------------------------------------
AGENT_ROLES = ["Growth", "Conservative", "Balanced", "RiskLiquidity"]

REWARD_WEIGHTS = {
    "Growth":         {"alpha": 3.00, "beta": 0.05, "lam": 3.00, "delta": 0.10, "liquidity_mu": 0.00},
    "Conservative":   {"alpha": 0.50, "beta": 0.80, "lam": 0.50, "delta": 0.40, "liquidity_mu": 0.00},
    "Balanced":       {"alpha": 2.20, "beta": 0.12, "lam": 2.20, "delta": 0.20, "liquidity_mu": 0.00},
    "RiskLiquidity":  {"alpha": 0.80, "beta": 0.50, "lam": 0.80, "delta": 0.30, "liquidity_mu": 0.50},
}
# NOTE on tuning: alpha and beta must be tuned relative to each other, not just
# in isolation. Per-step equity volatility (sigma ~1-2.8%) is intrinsically much
# larger than per-step equity mean return (mu ~0.02-0.12%) -- that's realistic
# market behaviour, but it means an alpha:beta ratio near 1:1 lets the risk
# penalty structurally dominate the return signal for EVERY role regardless of
# beta's absolute size, collapsing all agents onto the same "hide in cash"
# policy (see README, Results & Analysis, for the empirical failure this
# produced before retuning). Growth's alpha:beta ratio (60:1) is deliberately
# large enough that its average expected reward from equity exposure is
# positive; Conservative's ratio (0.6:1) is deliberately small enough that it
# is not. This is what actually makes "role -> reward function -> behaviour"
# hold in practice, not just in principle.
#
# NOTE on lam (transaction cost weight): lam is set EQUAL to alpha for every
# role, not as an independent small constant. txcost_frac and log_return are
# on the exact same scale (both fractional changes to capital), so a rupee
# lost to fees is just as real as a rupee lost to a bad market move -- there
# is no principled reason to weigh it 30x less. An earlier version used a
# small, role-independent lam (~0.10-0.20) and the agent's Q-values badly
# under-priced the true cost of switching baskets: Growth's regime-adaptive
# policy looked excellent in-reward (high training reward) but lost money in
# practice once real transaction costs were tallied up, because it was
# whipsawing between very different baskets on nearly every regime
# transition (mean regime duration is only ~2-3 steps here). Equalizing
# lam with alpha makes the reward correctly price a full basket switch at
# its real cost, which is what should discourage unnecessary whipsawing.
# See README, Results & Analysis, for the empirical before/after.
REWARD_SCALE = 100.0     # scales fractional returns/penalties into a friendlier reward range
REWARD_CLIP = 5.0        # clip final reward to [-REWARD_CLIP, +REWARD_CLIP]

# Reward mode: "individual" | "shared" | "hybrid"
REWARD_MODE = "individual"
HYBRID_ALPHA = 0.7   # R_total = HYBRID_ALPHA*R_i + (1-HYBRID_ALPHA)*R_shared

# ---------------------------------------------------------------------------
# Q-Learning hyperparameters
# ---------------------------------------------------------------------------
ALPHA_LR = 0.10          # Q-learning step size
GAMMA = 0.95             # discount factor
EPSILON_INITIAL = 1.00
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999    # multiplicative per-episode decay
# NOTE: 0.995 (floor reached by ~episode 600) was tried first and caused
# premature convergence -- agents locked into a policy before adequately
# sampling the state-action space, which visibly hurt the Balanced agent's
# outcome. 0.999 (floor reached by ~episode 3000 of 5000) gives a much
# longer exploration budget and produced materially more stable, sensible
# results across all four roles. See README "Results & Analysis" section.

Q_INIT_VALUE = 2.0       # optimistic Q-table initialization (see note below)
# NOTE on optimistic initialization: with the episode's start state visited
# only ONCE per episode (it's always the very first state), and with a
# transaction-cost structure that penalizes interrupting a sustained
# strategy, standard epsilon-greedy exploration rarely samples a "clean",
# uninterrupted run of any one action long enough to learn its true value --
# every random exploratory action along the way pays an extra switching
# cost that a genuinely persistent policy would not. This biases the agent
# toward whatever safe, zero-turnover-after-first-rebalance policy it
# happens to settle on early, rather than the financially best one. This
# is a well-known, standard remedy (Sutton & Barto, ch. 2): initializing
# Q-values optimistically forces the agent to try EVERY action at least a
# few times before its estimate drops to a realistic level, directly
# countering this bias -- without changing the exploration algorithm
# itself (still plain epsilon-greedy). See README, Results & Analysis.

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
EPISODE_LENGTH = 100
NUM_EPISODES = 5000

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"
