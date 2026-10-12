"use client";

import { useEffect, useState } from "react";
import { Blocks } from "@/components/Blocks";
import { Button, ErrorBox, NumberField, QueryBlocks, Radio, SelectField, SliderField, Spinner } from "@/components/Controls";
import { PageHeader } from "@/components/PageHeader";
import { useAction, useBlocksQuery, useSimulation } from "@/lib/hooks";
import { useMeta, useStore } from "@/lib/store";
import type { Block } from "@/lib/types";

type Source = "episode" | "oos";

export default function AnalyticsPage() {
  const meta = useMeta();
  const { agents } = useStore();
  const [source, setSource] = useState<Source>("episode");
  return (
    <div className="stack">
      <PageHeader title="Portfolio Analytics">
        Every number and chart here is computed from actual simulation output (synthetic market) - either the episode currently in the
        Simulation page, or many held-out episodes.
      </PageHeader>
      <Radio<Source> name="source" value={source} onChange={setSource}
        options={[{ value: "episode", label: "The episode in the Simulation page" }, { value: "oos", label: "Out-of-sample evaluation of the active agents" }]} />
      {source === "episode" ? <EpisodeAnalytics /> : <OosAnalytics key={agents.state ?? "ref"} start={meta.eval_seed_start} n={meta.n_eval_episodes} />}
    </div>
  );
}

function EpisodeAnalytics() {
  const { agents } = useStore();
  const [vol, setVol] = useState(10);
  const [debounced, setDebounced] = useState(10);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(vol), 300);
    return () => clearTimeout(id);
  }, [vol]);
  const { resp, blocks, busy, error } = useSimulation("analytics", debounced);
  return (
    <div className="stack">
      <p className="small muted">Agents: <b>{agents.label}</b> | market seed <b>{resp?.seed ?? "..."}</b> | {resp?.summary_line ?? "..."}</p>
      <div style={{ maxWidth: 420 }}><SliderField label="Rolling-volatility window (steps)" value={vol} min={3} max={30} step={1} onChange={setVol} /></div>
      <ErrorBox message={error} />
      {!resp && busy ? <Spinner text="Computing analytics..." /> : null}
      <div style={{ opacity: busy && resp ? 0.6 : 1, transition: "opacity .15s" }}><Blocks blocks={blocks} /></div>
    </div>
  );
}

function OosAnalytics({ start: defStart, n: defN }: { start: number; n: number }) {
  const { agents } = useStore();
  const [start, setStart] = useState(defStart);
  const [n, setN] = useState(defN);
  const run = useAction<{ blocks: Block[]; seeds: number[] }>();
  const [pick, setPick] = useState<number | null>(null);
  const [vol, setVol] = useState(10);
  const seeds = run.data?.seeds ?? [];
  const ep = useBlocksQuery("/api/analytics/oos-episode", !!run.data && pick !== null, () => ({ agents: agents.state, seed: pick, vol_window: vol }), [run.data, pick, vol, agents.state], 350);

  return (
    <div className="stack">
      <div className="grid">
        <NumberField label="First evaluation seed" value={start} min={0} max={1_000_000} onChange={setStart} />
        <NumberField label="Number of episodes" value={n} min={5} max={200} step={5} onChange={setN} />
        <Button primary disabled={run.loading} onClick={async () => { const r = await run.run("/api/analytics/oos", { agents: agents.state, start, n }); if (r) setPick(r.seeds[0]); }}>
          {run.loading ? "Evaluating..." : "Run out-of-sample evaluation"}
        </Button>
      </div>
      <ErrorBox message={run.error} />
      {run.loading ? <Spinner text="Playing the frozen agents on fresh markets..." /> : null}
      {!run.data && !run.loading ? <p className="caption">Press the button to play the active (frozen, greedy) agents on fresh markets they were not trained on.</p> : null}
      {run.data ? (
        <>
          <Blocks blocks={run.data.blocks} />
          <hr style={{ border: 0, borderTop: "1px solid var(--line)", width: "100%" }} />
          <h5 className="block-heading">Inspect one held-out episode in detail</h5>
          <div className="grid">
            <SelectField label="Held-out seed" value={pick ?? seeds[0]} options={seeds.map((s) => ({ value: s, label: String(s) }))} onChange={setPick} />
            <SliderField label="Rolling-volatility window (steps)" value={vol} min={3} max={30} step={1} onChange={setVol} />
          </div>
          <QueryBlocks q={ep} />
        </>
      ) : null}
    </div>
  );
}
