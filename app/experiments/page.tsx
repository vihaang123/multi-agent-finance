"use client";

import { useRef, useState } from "react";
import { Button, Checkbox, ErrorBox, NumberField, Panel, Progress, QueryBlocks, Radio, SelectField, SliderField } from "@/components/Controls";
import { PageHeader } from "@/components/PageHeader";
import { api, errorText } from "@/lib/api";
import { useBlocksQuery } from "@/lib/hooks";
import { appendLog } from "@/lib/log";
import { useMeta, useStore } from "@/lib/store";
import type { ExpEntry, Log } from "@/lib/types";

const CHUNK = 50;
type View = "ref" | "live";

export default function ExperimentsPage() {
  const meta = useMeta();
  const { expResults, setExpResults } = useStore();
  const d = meta.hp_defaults;
  const [sel, setSel] = useState<string[]>(["A", "B", "C", "D"]);
  const [episodes, setEpisodes] = useState(1000);
  const [trainSeed, setTrainSeed] = useState(meta.train_seed);
  const [evalStart, setEvalStart] = useState(meta.eval_seed_start);
  const [evalN, setEvalN] = useState(meta.n_eval_episodes);
  const [fixed, setFixed] = useState(2);
  const [alpha, setAlpha] = useState(d.alpha);
  const [gamma, setGamma] = useState(d.gamma);
  const [decay, setDecay] = useState(d.epsilon_decay);
  const [epLen, setEpLen] = useState(meta.episode_length);
  const [impact, setImpact] = useState(true);
  const [view, setView] = useState<View>(expResults ? "live" : "ref");
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<{ value: number; text: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const cancel = useRef(false);

  const config = {
    n_train_episodes: episodes, train_seed: trainSeed, eval_seed_start: evalStart, n_eval_episodes: evalN,
    episode_length: epLen, fixed_action: fixed, enable_market_impact: impact, alpha, gamma, epsilon_decay: decay,
  };
  const short = (e: string) => meta.experiment_names[e];

  async function run() {
    const selected = ["A", "B", "C", "D"].filter((e) => sel.includes(e));
    if (!selected.length) return setError("Select at least one experiment.");
    setError(null);
    cancel.current = false;
    setRunning(true);
    const entries: Record<string, ExpEntry> = {};
    try {
      for (let i = 0; i < selected.length && !cancel.current; i++) {
        const e = selected[i];
        const base = i / selected.length, span = 1 / selected.length;
        if (e === "A" || e === "B") {
          setProgress({ value: base, text: `${short(e)} (${i + 1}/${selected.length}): evaluating (no training phase)...` });
          entries[e] = (await api<{ entry: ExpEntry }>("/api/experiments/baseline", { experiment: e, config })).entry;
        } else {
          let state: string | null = null;
          let log: Log | null = null;
          let done = 0;
          while (done < episodes && !cancel.current) {
            const k = Math.min(CHUNK, episodes - done);
            const r: { state: string; rows: Log; episodes_done: number } = await api("/api/experiments/train", { experiment: e, config, state, episodes: k });
            state = r.state;
            log = appendLog(log, r.rows);
            done += r.episodes_done;
            setProgress({ value: base + (span * 0.9 * done) / episodes, text: `${short(e)} (${i + 1}/${selected.length}): training episode ${done}/${episodes}` });
          }
          if (cancel.current || !state || !log) break;
          setProgress({ value: base + span * 0.92, text: `${short(e)} (${i + 1}/${selected.length}): evaluating on held-out seeds...` });
          entries[e] = (await api<{ entry: ExpEntry }>("/api/experiments/evaluate", { experiment: e, config, state, log })).entry;
        }
      }
      if (cancel.current) {
        setError("Stopped before all experiments finished; nothing was saved.");
      } else {
        setExpResults({ entries, metaLine: `Live run | training ${episodes} episodes | evaluation seeds ${evalStart}-${evalStart + evalN - 1}` });
        setView("live");
      }
    } catch (e) {
      setError(errorText(e));
    } finally {
      setRunning(false);
      setProgress(null);
    }
  }

  const liveQ = useBlocksQuery("/api/experiments/report", view === "live" && !!expResults, () => ({ entries: expResults?.entries, meta_line: expResults?.metaLine ?? "" }), [view, expResults]);
  const refQ = useBlocksQuery("/api/experiments/reference", view === "ref", () => ({}), [view]);

  return (
    <div className="stack">
      <PageHeader title="Experiments">
        Compare four ways of allocating the same Rs 10,00,000 in the same synthetic market. Learning agents TRAIN on one seed, then are
        EVALUATED out-of-sample on a different block of seeds with exploration off and Q-tables frozen. Training and evaluation numbers are
        never mixed. Same configuration + same seeds = same results.
      </PageHeader>
      <Panel title="Configure">
        <div className="toolbar" role="group" aria-label="Experiments to run">
          {["A", "B", "C", "D"].map((e) => (
            <Checkbox key={e} label={short(e)} checked={sel.includes(e)} onChange={(v) => setSel((s) => (v ? [...s, e] : s.filter((x) => x !== e)))} />
          ))}
        </div>
        <div className="grid" style={{ marginTop: ".6rem" }}>
          <NumberField label={`Training episodes (C, D; max ${meta.limits.ui_max_episodes})`} value={episodes} min={10} max={meta.limits.ui_max_episodes} step={100} onChange={setEpisodes} />
          <NumberField label="Training seed" value={trainSeed} min={0} max={1_000_000} onChange={setTrainSeed} />
          <NumberField label="First evaluation seed" value={evalStart} min={0} max={1_000_000} onChange={setEvalStart} />
          <NumberField label="Evaluation episodes" value={evalN} min={5} max={200} step={5} onChange={setEvalN} />
          <SelectField label="Fixed basket for B" value={fixed} options={meta.baskets.map((b, i) => ({ value: i, label: `${i}: ${b.name}` }))} onChange={setFixed} />
          <SliderField label="alpha (C, D)" value={alpha} min={0.01} max={1} step={0.01} onChange={setAlpha} />
          <SliderField label="gamma (C, D)" value={gamma} min={0} max={0.999} step={0.005} onChange={setGamma} />
          <SliderField label="epsilon decay (C, D)" value={decay} min={0.99} max={1} step={0.0005} format={(v) => v.toFixed(4)} onChange={setDecay} />
          <NumberField label="Episode length (steps)" value={epLen} min={20} max={300} step={10} onChange={setEpLen} />
          <Checkbox label="Market impact between agents" checked={impact} onChange={setImpact} />
        </div>
        <div className="toolbar" style={{ marginTop: ".8rem" }}>
          <Button primary disabled={running} onClick={run} title="Runs the real training + evaluation pipeline. About 30-40 s per 1000 training episodes for D.">Run selected experiments</Button>
          {running ? <Button onClick={() => { cancel.current = true; }}>Cancel</Button> : null}
        </div>
        {progress ? <div style={{ marginTop: ".6rem" }}><Progress value={progress.value} text={progress.text} /></div> : null}
        <ErrorBox message={error} />
      </Panel>

      <Radio<View> name="expview" value={view} onChange={setView}
        options={[{ value: "ref", label: "Precomputed reference (5000 episodes)" }, { value: "live", label: "Your live run" }]} />
      {view === "live" ? (
        expResults ? <QueryBlocks q={liveQ} /> : <p className="caption">Configure above and press <b>Run selected experiments</b>; your results will appear here.</p>
      ) : (
        <QueryBlocks q={refQ} />
      )}
    </div>
  );
}
