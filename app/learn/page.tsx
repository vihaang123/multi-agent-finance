"use client";

import { useEffect, useState, type ReactNode } from "react";
import { BlockView } from "@/components/Blocks";
import { Markdown } from "@/components/Markdown";
import { NumberField, QueryBlocks, SelectField, SliderField, TabBar } from "@/components/Controls";
import { PageHeader } from "@/components/PageHeader";
import { api } from "@/lib/api";
import { useBlocksQuery } from "@/lib/hooks";
import { useMeta } from "@/lib/store";
import type { Block, Meta } from "@/lib/types";

type Static = { transition: Block; stationary: Block; params: Block; baskets: Block; reward_weights: Block };

const SHORT_BUCKET = ["Low", "Med", "High"];

export default function LearnPage() {
  const meta = useMeta();
  const [tab, setTab] = useState(0);
  const [stat, setStat] = useState<Static | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api<Static>("/api/learn/static", {}).then(setStat).catch((e) => setErr(String(e.message ?? e)));
  }, []);
  const tabs = ["Big picture", "Market & regimes", "States & actions", "Reward", "Q-learning"];
  return (
    <div className="stack">
      <PageHeader title="Learn: how it works">
        Plain-language explanations of every moving part, with small calculators that use the project&apos;s real configuration.
      </PageHeader>
      <TabBar tabs={tabs} active={tab} onChange={setTab} />
      {err ? <div className="callout error">{err}</div> : null}
      {tab === 0 ? <BigPicture /> : null}
      {tab === 1 ? <MarketTab stat={stat} /> : null}
      {tab === 2 ? <StatesTab meta={meta} stat={stat} /> : null}
      {tab === 3 ? <RewardTab meta={meta} stat={stat} /> : null}
      {tab === 4 ? <QTab /> : null}
    </div>
  );
}

function Two({ left, right }: { left: ReactNode; right: ReactNode }) {
  return <div className="cols" style={{ gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)" }}><div className="col">{left}</div><div className="col">{right}</div></div>;
}

function BigPicture() {
  const steps = [
    ["1. Each agent sees its own state s", "regime + its own weights", "#eef4fd"],
    ["2. Pick an action a", "one of 7 baskets; epsilon-greedy on Q(s, .)", "#eef4fd"],
    ["3. Shared market", "rebalance + trading costs, crowding impact on equity, returns drawn for the regime", "#fff6e0"],
    ["4. Reward and next state", "each agent gets its own reward r and next state s'", "#e8f6f0"],
    ["5. Update own Q-table", "Q(s,a) <- Q + alpha*(r + gamma*max Q(s') - Q)", "#e8f6f0"],
  ];
  return (
    <div className="stack">
      <Markdown text={`**The problem.** You have Rs 10,00,000 to split between **Equity, Bonds, Gold and Cash**. The market changes mood (calm, booming, crashing, jittery) and every trade costs a little. How should the money be allocated over time?

**The twist.** Instead of one decision-maker we have **four agents**, each with its own personality (Growth, Conservative, Balanced, RiskLiquidity) and its own Rs 2,50,000. They all trade in the **same** market and all learn **by trial and error**.`} />
      <div className="flow" aria-label="One step of the learning loop">
        {steps.map(([t, b, bg], i) => (
          <div key={t} style={{ display: "contents" }}>
            <div className="step" style={{ background: bg }}><b>{t}</b><br />{b}</div>
            {i < steps.length - 1 ? <span className="arrow" aria-hidden>→</span> : <span className="arrow" aria-hidden title="next step">↺</span>}
          </div>
        ))}
      </div>
      <Two
        left={<><h4>Markov Decision Process (MDP)</h4><Markdown text={`An MDP is the rulebook for **one** learner acting in a world: *states* it can be in, *actions* it can take, *rewards* it receives, and *transition probabilities* for how the world moves next. The key idea (**Markov property**): the next situation depends only on the *current* state and action, not on the whole history.

Experiment C (a single agent managing all the money) is exactly an MDP.`} /></>}
        right={<><h4>Markov Game (multi-agent)</h4><Markdown text={`With several agents acting **at the same time**, the next state and each reward depend on **everyone's** actions together. That is a *Markov game*. Here the agents are coupled through the market: if they all pile into equity, the combined buying pressure lowers the equity return for **all of them** (market impact).

Each agent learns **independently** (**Independent Q-learning**): it treats the others as part of the environment. Simple, but the environment then keeps shifting as the others learn - that is why multi-agent learning is harder.`} /></>}
      />
      <div className="callout note"><b>Where the simulation is exogenous vs endogenous.</b> The sequence of market regimes and the base random return draws are the same for every policy on a given seed (exogenous). Only the crowding penalty on equity depends on the agents&apos; choices (endogenous). This keeps comparisons fair.</div>
    </div>
  );
}

function MarketTab({ stat }: { stat: Static | null }) {
  const [seed, setSeed] = useState(1);
  const [steps, setSteps] = useState(60);
  const path = useBlocksQuery("/api/learn/regime-path", true, () => ({ seed, steps }), [seed, steps]);
  return (
    <div className="stack">
      <Markdown text="The market is **synthetic**. A hidden **regime** - *Bear, Neutral, Bull or HighVol* - decides how the four assets behave that day. Regimes change according to a **Markov chain**: tomorrow's regime depends only on today's, with the probabilities below." />
      <div className="cols" style={{ gridTemplateColumns: "minmax(0,1.2fr) minmax(0,1fr)" }}>
        <div className="col">{stat ? <BlockView b={stat.transition} /> : null}</div>
        <div className="col stack">
          <Markdown text="**Long-run share of time in each regime** (stationary distribution, computed from the matrix):" />
          {stat ? <BlockView b={stat.stationary} /> : null}
          <p className="caption">Rows add up to 1. E.g. a Bull day is followed by another Bull day 65% of the time - regimes are &apos;sticky&apos;.</p>
        </div>
      </div>
      {stat ? <BlockView b={stat.params} /> : null}
      <p className="caption">Each step, every asset return is drawn from a Normal distribution whose mean and spread depend on the regime. Equity wins in Bull, loses in Bear; Bonds and Gold cushion Bear; HighVol is noisy for everything.</p>
      <h5 className="block-heading">Try the regime chain</h5>
      <div className="grid">
        <NumberField label="Seed" value={seed} min={0} max={100000} onChange={setSeed} />
        <SliderField label="Steps" value={steps} min={10} max={200} step={10} onChange={setSteps} />
      </div>
      <QueryBlocks q={path} />
    </div>
  );
}

function StatesTab({ meta, stat }: { meta: Meta; stat: Static | null }) {
  const [r, setR] = useState(0), [e, setE] = useState(0), [b, setB] = useState(0), [g, setG] = useState(0);
  const idx = r * 27 + e * 9 + b * 3 + g;
  const buckets = meta.bucket_labels.map((l, i) => ({ value: i, label: l }));
  return (
    <div className="stack">
      <Markdown text={`**State space (108 states).** A state is what an agent can 'see' when deciding:

\`state = (market regime) x (equity weight bucket) x (bond weight bucket) x (gold weight bucket)\` = 4 x 3 x 3 x 3 = **108**.

A weight bucket is **Low** (below 20%), **Medium** (20%-45%) or **High** (45%+). Agents see *their own* weights, not other agents'.`} />
      <div className="grid">
        <SelectField label="Regime" value={r} options={meta.regimes.map((n, i) => ({ value: i, label: n }))} onChange={setR} />
        <SelectField label="Equity" value={e} options={buckets} onChange={setE} />
        <SelectField label="Bonds" value={b} options={buckets} onChange={setB} />
        <SelectField label="Gold" value={g} options={buckets} onChange={setG} />
      </div>
      <Markdown text={`State index = ${r} x 27 + ${e} x 9 + ${b} x 3 + ${g} = **${idx}**  (${meta.regimes[r]} regime | Equity ${SHORT_BUCKET[e]}, Bonds ${SHORT_BUCKET[b]}, Gold ${SHORT_BUCKET[g]})`} />
      <Markdown text="**Action space (7 baskets).** Choosing an action means choosing a target mix. Predefined baskets keep the table small (7 actions instead of an infinite number of weight combinations):" />
      {stat ? <BlockView b={stat.baskets} /> : null}
      <p className="caption">After choosing, the portfolio is rebalanced to these weights; the trading cost is {(meta.tx_cost_rate * 100).toFixed(2)}% of the amount moved. Then returns make the weights drift, which is what the next state shows.</p>
    </div>
  );
}

function RewardTab({ meta, stat }: { meta: Meta; stat: Static | null }) {
  const roles = Object.keys(meta.reward_weights);
  const [role, setRole] = useState(roles[0]);
  const [basket, setBasket] = useState(2);
  const [regime, setRegime] = useState(2);
  const [lr, setLr] = useState(0.5), [tc, setTc] = useState(0.05), [dd, setDd] = useState(0);
  const wt = meta.reward_weights[role];
  const w = meta.baskets[basket].weights;
  const sigma = meta.asset_sigma[regime];
  const risk = w.reduce((s, x, i) => s + x * sigma[i], 0);
  const terms: [string, number][] = [
    ["Return", wt.alpha * (lr / 100) * meta.reward_scale],
    ["Risk penalty", -wt.beta * risk * meta.reward_scale],
    ["Trading cost", -wt.lam * (tc / 100) * meta.reward_scale],
    ["Drawdown penalty", -wt.delta * (dd / 100) * meta.reward_scale],
    ["Liquidity bonus", wt.liquidity_mu * w[meta.cash_index]],
  ];
  const total = terms.reduce((s, [, v]) => s + v, 0);
  const clipped = Math.max(-meta.reward_clip, Math.min(meta.reward_clip, total));
  return (
    <div className="stack">
      <p>A good reward tells each agent what &apos;good&apos; means <b>for its personality</b>. Per step:</p>
      <pre className="math">{"r_i = alpha_i*100*(log return) - beta_i*100*(risk) - lambda_i*100*(trading cost) - delta_i*100*(new drawdown) + mu_i*(cash weight)"}</pre>
      <p>then clipped to [-{meta.reward_clip}, +{meta.reward_clip}]. The weights per role (from <code>config.py</code>):</p>
      {stat ? <BlockView b={stat.reward_weights} /> : null}
      <p><b>Reward calculator</b> - plug in a situation and see each term (uses the real config):</p>
      <div className="grid g3">
        <SelectField label="Role" value={role} options={roles.map((x) => ({ value: x, label: x }))} onChange={setRole} />
        <SelectField label="Basket chosen" value={basket} options={meta.baskets.map((x, i) => ({ value: i, label: x.name }))} onChange={setBasket} />
        <SelectField label="Regime" value={regime} options={meta.regimes.map((x, i) => ({ value: i, label: x }))} onChange={setRegime} />
        <SliderField label="Portfolio log return this step (%)" value={lr} min={-3} max={3} step={0.05} onChange={setLr} />
        <SliderField label="Trading cost (% of capital)" value={tc} min={0} max={0.3} step={0.01} onChange={setTc} />
        <SliderField label="New drawdown created this step (%)" value={dd} min={0} max={3} step={0.1} onChange={setDd} />
      </div>
      <div className="table-wrap"><div className="table-scroll"><table><thead><tr><th>Term</th><th>Points</th></tr></thead>
        <tbody>{terms.map(([k, v]) => <tr key={k}><td className="first">{k}</td><td>{v.toFixed(3)}</td></tr>)}</tbody></table></div></div>
      <p>Risk of this basket in this regime = weights . sigma = <b>{risk.toFixed(4)}</b>. Total = <b>{total >= 0 ? "+" : ""}{total.toFixed(3)}</b> → after clipping <b>{clipped >= 0 ? "+" : ""}{clipped.toFixed(3)}</b>.</p>
      <p className="caption">Growth cares mostly about return (alpha 3.0, tiny risk penalty); Conservative weighs risk and drawdown heavily; RiskLiquidity earns a bonus for holding cash. These are fixed project parameters, not tuned to flatter results.</p>
    </div>
  );
}

function QTab() {
  const [q, setQ] = useState(2), [r, setR] = useState(1), [m, setM] = useState(3), [al, setAl] = useState(0.1), [ga, setGa] = useState(0.95);
  const target = r + ga * m;
  const td = target - q;
  return (
    <div className="stack">
      <Markdown text={`**Q-learning** keeps a table \`Q(s, a)\`: *'how good is it, in the long run, to take action a when in state s?'*. Ours has 108 rows (states) x 7 columns (actions). It starts with a generous **optimistic** value, then each real experience nudges one cell:`} />
      <pre className="math">{"Q(s,a)  <-  Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]"}</pre>
      <Markdown text={`- **alpha** (learning rate): how big a step to take toward the new evidence.
- **gamma** (discount): how much the future matters compared with the immediate reward.
- The bracket is the **TD error** = (what happened) - (what we expected).
- **Epsilon-greedy**: with probability epsilon try a random basket (explore), otherwise take the best-known one (exploit). Epsilon starts at 1.0 and shrinks every episode.`} />
      <p><b>Update calculator:</b></p>
      <div className="grid">
        <NumberField label="Q(s,a) now" value={q} step={0.5} onChange={setQ} />
        <NumberField label="Reward r" value={r} step={0.5} onChange={setR} />
        <NumberField label="max Q(s',.)" value={m} step={0.5} onChange={setM} />
        <NumberField label="alpha" value={al} min={0.01} max={1} step={0.01} onChange={setAl} />
        <NumberField label="gamma" value={ga} min={0} max={1} step={0.01} onChange={setGa} />
      </div>
      <p>target = {r} + {ga} x {m} = <b>{target.toFixed(3)}</b>; TD error = {target.toFixed(3)} - {q} = <b>{td >= 0 ? "+" : ""}{td.toFixed(3)}</b>; new Q = {q} + {al} x {td >= 0 ? "+" : ""}{td.toFixed(3)} = <b>{(q + al * td).toFixed(4)}</b></p>
      <Markdown text={`**Independent multi-agent Q-learning** = each agent owns a private table and learns only from its own (state, action, reward). Nothing is shared except the market.

**Honest limitations of this baseline** (also in the README):

- Plain epsilon-greedy rarely produces long runs of the *same* action, but sustained commitment is exactly what pays when trading is costly. In our runs all four learners end up favouring a cash-heavy basket and **do not specialise by role**, and a plain fixed Balanced basket beats them out-of-sample.
- Tabular Q-learning cannot generalise between similar states; a bigger state space would need function approximation (not used here).
- All data is synthetic, so none of this says anything about real markets.`} />
    </div>
  );
}
