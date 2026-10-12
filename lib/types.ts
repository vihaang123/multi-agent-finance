// Types shared by the pages. Block = the generic view blocks produced by server/blocks.py.

export type Tone = "note" | "warn" | "ok" | "error";

export type Kpi = { label: string; value: string; sub?: string; tone?: "pos" | "neg" | ""; pill?: string };

export type PlotFigure = { data: unknown[]; layout: Record<string, unknown> };

export type Block =
  | { type: "heading"; text: string; level?: number }
  | { type: "markdown"; text: string }
  | { type: "caption"; text: string }
  | { type: "callout"; tone: Tone; text: string }
  | { type: "kpis"; items: Kpi[] }
  | { type: "plot"; figure: PlotFigure }
  | { type: "table"; columns: string[]; rows: (string | number | null)[][]; caption?: string; max_height?: number }
  | { type: "columns"; cols: { weight: number; blocks: Block[] }[] }
  | { type: "tabs"; tabs: { label: string; blocks: Block[] }[] }
  | { type: "switch"; label: string; options: { label: string; blocks: Block[] }[] }
  | { type: "cards"; items: { title: string; tag?: string; color?: string; body?: string; blocks?: Block[] }[] }
  | { type: "download"; label: string; filename: string; mime: string; content: string }
  | { type: "math"; text: string };

export type Basket = { name: string; weights: number[] };

export type Meta = {
  synthetic_text: string;
  agent_ids: number[];
  roles: Record<string, string>;
  agent_colors: Record<string, string>;
  regime_colors: Record<string, string>;
  regimes: string[];
  assets: string[];
  baskets: Basket[];
  bucket_labels: string[];
  policy_labels: Record<string, string>;
  experiment_names: Record<string, string>;
  learning_experiments: string[];
  reward_weights: Record<string, { alpha: number; beta: number; lam: number; delta: number; liquidity_mu: number }>;
  reward_scale: number;
  reward_clip: number;
  asset_sigma: number[][];
  cash_index: number;
  tx_cost_rate: number;
  episode_length: number;
  initial_capital_per_agent: number;
  total_initial_capital: number;
  train_seed: number;
  eval_seed_start: number;
  n_eval_episodes: number;
  hp_defaults: { alpha: number; gamma: number; epsilon_initial: number; epsilon_min: number; epsilon_decay: number; q_init: number };
  limits: { lab_max_episodes: number; ui_max_episodes: number; chunk_episodes: number };
  reference: { available: boolean; label: string; meta: Record<string, unknown> | null };
  sharpe_note: string;
};

export type PolicyKind = "trained" | "random" | "fixed";
export type Policy = { kind: PolicyKind; basket: number };
export type Step = { a: number[]; k: string };

export type SimState = {
  seed: number;
  impact: boolean;
  policies: Policy[];
  steps: Step[] | null; // null = nothing has been run yet (the first page to need it runs an example episode)
};

export type SimResponse = {
  t: number;
  done: boolean;
  episode_length: number;
  steps: Step[];
  seed: number;
  summary_line: string;
  regime_now: string;
  regime_color: string;
  total_value: number;
  total_value_text: string;
  agents_label: string;
  blocks: Block[];
};

export type Log = Record<string, number[]>;

export type AgentsState = {
  source: "reference" | "lab";
  state: string | null; // opaque server-side state string; null = reference agents
  label: string;
  log: Log | null;
  totalEpisodes: number;
};

export type ExpEntry = Record<string, unknown>;
export type ExpResults = { entries: Record<string, ExpEntry>; metaLine: string } | null;
