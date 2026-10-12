"use client";

import { useRef, useState } from "react";
import { Blocks } from "@/components/Blocks";
import { Button, Checkbox, ErrorBox, NumberField, Panel, Progress, QueryBlocks, SelectField, SliderField, TabBar } from "@/components/Controls";
import { PageHeader } from "@/components/PageHeader";
import { api, errorText } from "@/lib/api";
import { useAction, useBlocksQuery } from "@/lib/hooks";
import { appendLog } from "@/lib/log";
import { useMeta, useStore } from "@/lib/store";
import type { Block, Log } from "@/lib/types";

const CHUNK = 50; // episodes per request (the server also stops a request early if it runs long)

export default function RlLabPage() {
  const meta = useMeta();
  const { agents, setAgents, restoreReference } = useStore();
  const d = meta.hp_defaults;
  const [nEpisodes, setNEpisodes] = useState(500);
  const [seed, setSeed] = useState(meta.train_seed);
  const [impact, setImpact] = useState(true);
  const [alpha, setAlpha] = useState(d.alpha);
  const [gamma, setGamma] = useState(d.gamma);
  const [qInit, setQInit] = useState(d.q_init);
  const [eps0, setEps0] = useState(d.epsilon_initial);
  const [epsMin, setEpsMin] = useState(d.epsilon_min);
  const [decay, setDecay] = useState(d.epsilon_decay);

  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [message, setMessage] = useState<{ tone: "ok" | "warn" | "error"; text: string } | null>(null);
  const cancel = useRef(false);

  const [tab, setTab] = useState(0);
  const [smooth, setSmooth] = useState(50);
  const [inspect, setInspect] = useState({ agent: 0, regime: 0, eq: 0, bond: 0, gold: 0 });
  const [heat, setHeat] = useState({ agent: 0, mode: "greedy" });
  const [trace, setTrace] = useState({ agent: 0, steps: 5, seed: 7 });
  const [evalN, setEvalN] = useState(50);

  const hp = { alpha, gamma, epsilon_initial: eps0, epsilon_min: epsMin, epsilon_decay: decay, q_init: qInit };
  const epsNote =
    decay >= 1 || epsMin >= eps0
      ? "epsilon never decays below its start value."
      : `epsilon reaches its floor after about ${Math.round(Math.log(Math.max(epsMin, 1e-9) / eps0) / Math.log(decay)).toLocaleString()} episodes.`;

  async function train(mode: "fresh" | "continue") {
    if (training) return;
    setMessage(null);
    if (!(nEpisodes >= 10 && nEpisodes <= meta.limits.lab_max_episodes)) {
      setMessage({ tone: "error", text: `Training episodes must be between 10 and ${meta.limits.lab_max_episodes} in the app.` });
      return;
    }
    const lo = meta.eval_seed_start, hi = meta.eval_seed_start + meta.n_eval_episodes - 1;
    if (seed >= lo && seed <= hi) {
      setMessage({ tone: "error", text: `Training seed ${seed} lies inside the held-out evaluation block ${lo}-${hi}; pick another so evaluation stays out-of-sample.` });
      return;
    }
    cancel.current = false;
    setTraining(true);
    const fromLab = mode === "continue" && agents.source === "lab";
    let state: string | null = fromLab ? agents.state : null;
    let log: Log | null = fromLab ? agents.log : null;
    let done = 0;
    let total = fromLab ? agents.totalEpisodes : 0;
    setProgress({ done: 0, total: nEpisodes });
    try {
      while (done < nEpisodes && !cancel.current) {
        const k = Math.min(CHUNK, nEpisodes - done);
        const r = await api<{ state: string; rows: Log; episodes_done: number; total_episodes: number; label: string }>("/api/lab/train", {
          mode, state, hp, seed, impact, episodes: k,
        });
        state = r.state;
        log = appendLog(log, r.rows);
        done += r.episodes_done;
        total = r.total_episodes;
        setAgents({ source: "lab", state, label: r.label, log, totalEpisodes: total });
        setProgress({ done, total: nEpisodes });
      }
      setMessage(
        cancel.current
          ? { tone: "warn", text: `Stopped after ${done} of ${nEpisodes} episodes. The agents keep what they learned; press Continue to train further.` }
          : { tone: "ok", text: `Finished ${done} episodes. The Simulation and Dashboard pages now use these agents.` },
      );
    } catch (e) {
      setMessage({ tone: "error", text: `${errorText(e)}${done ? ` (${done} episodes were completed and kept.)` : ""}` });
    } finally {
      setTraining(false);
      setProgress(null);
    }
  }

  const tabs = ["Learning curves", "Q-table inspector", "Policy maps", "Update walkthrough", "Evaluate (out-of-sample)"];
  const lab = agents.source === "lab";

  const curves = useBlocksQuery("/api/lab/curves", tab === 0, () => ({ agents: agents.state, log: lab ? agents.log : null, smooth }), [tab, agents.state, agents.totalEpisodes, smooth]);
  const insp = useBlocksQuery("/api/lab/inspect", tab === 1, () => ({ agents: agents.state, ...inspect }), [tab, agents.state, inspect.agent, inspect.regime, inspect.eq, inspect.bond, inspect.gold]);
  const heatQ = useBlocksQuery("/api/lab/policy", tab === 2, () => ({ agents: agents.state, ...heat }), [tab, agents.state, heat.agent, heat.mode]);
  const walk = useAction<{ blocks: Block[] }>();
  const evalA = useAction<{ blocks: Block[] }>();
  const roles = meta.agent_ids.map((a) => ({ value: a, label: meta.roles[String(a)] }));
  const bucket = meta.bucket_labels.map((l, i) => ({ value: i, label: l }));

  return (
    <div className="stack">
      <PageHeader title="RL Lab">
        Train real tabular Q-learning agents, watch the learning curves and open the Q-tables. Nothing is pre-baked: pressing Train runs
        the same training loop that produced the reference results.
      </PageHeader>

      <Panel title="1. Configure">
        <div className="grid g3">
          <NumberField label={`Training episodes (max ${meta.limits.lab_max_episodes})`} value={nEpisodes} min={10} max={meta.limits.lab_max_episodes} step={50} onChange={setNEpisodes}
            help="Each episode is 100 market steps with all 4 agents acting and learning. Roughly 30-40 s per 1000 episodes on the server." />
          <NumberField label="Training market seed" value={seed} min={0} max={1_000_000} onChange={setSeed}
            help="Seed of the single continuing market stream used for training. Must stay outside 51-100 (held-out evaluation)." />
          <Checkbox label="Market impact between agents" checked={impact} onChange={setImpact} />
          <SliderField label="Learning rate alpha" value={alpha} min={0.01} max={1} step={0.01} onChange={setAlpha} help="How far each update moves Q(s,a) toward its target." />
          <SliderField label="Discount factor gamma" value={gamma} min={0} max={0.999} step={0.005} onChange={setGamma} help="How much future reward counts relative to immediate reward." />
          <SliderField label="Initial Q-value (optimism)" value={qInit} min={-10} max={30} step={0.5} onChange={setQInit} />
          <SliderField label="Starting exploration epsilon" value={eps0} min={0.05} max={1} step={0.05} onChange={setEps0} />
          <SliderField label="Minimum epsilon" value={epsMin} min={0} max={0.5} step={0.01} onChange={setEpsMin} />
          <SliderField label="Epsilon decay per episode" value={decay} min={0.99} max={1} step={0.0005} format={(v) => v.toFixed(4)} onChange={setDecay} />
        </div>
        <p className="caption" style={{ marginTop: ".6rem" }}>
          {epsNote} Reward weights per role, the 7 baskets, the 108 states and the market model are fixed project configuration (see the Learn page).
        </p>
        <h4>2. Run</h4>
        <div className="toolbar">
          <Button primary disabled={training} onClick={() => train("fresh")} title="Creates 4 brand-new agents (all Q-values = initial value) and trains them.">Train from scratch</Button>
          <Button disabled={training} onClick={() => train("continue")} title="Keeps the current Q-tables and epsilon; alpha, gamma, epsilon floor and decay are updated.">Continue training</Button>
          <Button disabled={training} onClick={() => { restoreReference(); setMessage(null); }} title="Go back to the precomputed 5000-episode agents.">Restore reference agents</Button>
          {training ? <Button onClick={() => { cancel.current = true; }}>Stop after this chunk</Button> : null}
        </div>
        <p className="small" style={{ margin: ".6rem 0 0" }}>Active agents: <b>{agents.label}</b></p>
        {progress ? <div style={{ marginTop: ".6rem" }}><Progress value={progress.done / progress.total} text={`Training episode ${progress.done}/${progress.total}`} /></div> : null}
        {message ? <div className={`callout ${message.tone}`} style={{ marginTop: ".6rem" }}>{message.text}</div> : null}
      </Panel>

      <TabBar tabs={tabs} active={tab} onChange={setTab} />

      {tab === 0 ? (
        <div className="stack">
          {lab && agents.log ? (
            <div style={{ maxWidth: 420 }}><SliderField label="Smoothing window (episodes)" value={smooth} min={1} max={200} step={1} onChange={setSmooth} /></div>
          ) : null}
          <QueryBlocks q={curves} />
        </div>
      ) : null}

      {tab === 1 ? (
        <div className="stack">
          <div className="grid">
            <SelectField label="Agent" value={inspect.agent} options={roles.map((r, i) => ({ value: i, label: r.label }))} onChange={(v) => setInspect({ ...inspect, agent: v })} />
            <SelectField label="Regime" value={inspect.regime} options={meta.regimes.map((r, i) => ({ value: i, label: r }))} onChange={(v) => setInspect({ ...inspect, regime: v })} />
            <SelectField label="Equity weight" value={inspect.eq} options={bucket} onChange={(v) => setInspect({ ...inspect, eq: v })} />
            <SelectField label="Bond weight" value={inspect.bond} options={bucket} onChange={(v) => setInspect({ ...inspect, bond: v })} />
            <SelectField label="Gold weight" value={inspect.gold} options={bucket} onChange={(v) => setInspect({ ...inspect, gold: v })} />
          </div>
          <QueryBlocks q={insp} />
        </div>
      ) : null}

      {tab === 2 ? (
        <div className="stack">
          <div className="grid">
            <SelectField label="Agent" value={heat.agent} options={roles.map((r, i) => ({ value: i, label: r.label }))} onChange={(v) => setHeat({ ...heat, agent: v })} />
            <SelectField label="Map" value={heat.mode} options={[{ value: "greedy", label: "Learned basket" }, { value: "maxq", label: "Best Q-value" }, { value: "visits", label: "Visit counts" }]} onChange={(v) => setHeat({ ...heat, mode: v })} />
          </div>
          <QueryBlocks q={heatQ} />
        </div>
      ) : null}

      {tab === 3 ? (
        <div className="stack">
          <p>Run the first few steps of an episode on a <b>copy</b> of the current agents and see each Q-update&apos;s arithmetic. The result is checked against the real update rule.</p>
          <div className="grid">
            <SelectField label="Agent" value={trace.agent} options={roles.map((r, i) => ({ value: i, label: r.label }))} onChange={(v) => setTrace({ ...trace, agent: v })} />
            <NumberField label="Steps to show" value={trace.steps} min={1} max={20} onChange={(v) => setTrace({ ...trace, steps: v })} />
            <NumberField label="Episode seed" value={trace.seed} min={0} max={1_000_000} onChange={(v) => setTrace({ ...trace, seed: v })} />
            <Button primary disabled={walk.loading} onClick={() => walk.run("/api/lab/walkthrough", { agents: agents.state, ...trace })}>Show the updates</Button>
          </div>
          <ErrorBox message={walk.error} />
          {walk.data ? <Blocks blocks={walk.data.blocks} /> : null}
        </div>
      ) : null}

      {tab === 4 ? (
        <div className="stack">
          <p>Freeze the Q-tables (epsilon = 0, no learning) and play brand-new markets the agents have never seen.</p>
          <div className="grid">
            <NumberField label="Held-out episodes" value={evalN} min={5} max={200} step={5} onChange={setEvalN} />
            <p className="caption" style={{ margin: 0 }}>Evaluation seeds {meta.eval_seed_start} ... {meta.eval_seed_start} + n - 1 (disjoint from the training seed).</p>
            <Button primary disabled={evalA.loading} onClick={() => evalA.run("/api/lab/evaluate", { agents: agents.state, n: evalN, start: meta.eval_seed_start, impact })}>
              {evalA.loading ? "Evaluating..." : "Evaluate current agents"}
            </Button>
          </div>
          <ErrorBox message={evalA.error} />
          {evalA.data ? <Blocks blocks={evalA.data.blocks} /> : <p className="caption">Press Evaluate to run the out-of-sample test.</p>}
        </div>
      ) : null}
    </div>
  );
}
