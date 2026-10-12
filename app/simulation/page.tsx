"use client";

import { useEffect, useState } from "react";
import { Blocks } from "@/components/Blocks";
import { Button, Checkbox, ErrorBox, NumberField, Progress, Radio, SelectField, SliderField, Spinner } from "@/components/Controls";
import { PageHeader } from "@/components/PageHeader";
import { useSimulation } from "@/lib/hooks";
import { inr } from "@/lib/log";
import { useMeta, useStore } from "@/lib/store";
import type { PolicyKind } from "@/lib/types";

type View = "overview" | "decisions" | "money" | "log";
const VIEWS: { value: View; label: string }[] = [
  { value: "overview", label: "Overview" },
  { value: "decisions", label: "How agents decide" },
  { value: "money", label: "Money flow (Rs)" },
  { value: "log", label: "Step log" },
];

export default function SimulationPage() {
  const meta = useMeta();
  const { sim, setSim } = useStore();
  const [view, setView] = useState<View>("overview");
  const { resp, blocks, busy, error, call, latency } = useSimulation(view);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(2);
  const [stepsN, setStepsN] = useState(10);
  const [seedDraft, setSeedDraft] = useState(sim.seed);
  const [impactDraft, setImpactDraft] = useState(sim.impact);
  useEffect(() => setSeedDraft(sim.seed), [sim.seed]);
  useEffect(() => setImpactDraft(sim.impact), [sim.impact]);

  const done = !!resp?.done;
  const t = resp?.t ?? 0;

  // Auto-play: one step per tick, waiting for each answer (the server rebuilds the episode every time).
  useEffect(() => {
    if (!playing) return;
    if (done) {
      setPlaying(false);
      return;
    }
    if (!resp || busy) return;
    const id = setTimeout(() => void call({ advance: 1 }), Math.max(0, 1000 / speed - latency));
    return () => clearTimeout(id);
  }, [playing, done, resp, busy, speed, latency, call]);

  const stop = () => setPlaying(false);
  const roleName = (i: number) => meta.roles[String(meta.agent_ids[i])];

  return (
    <div className="stack">
      <PageHeader title="Interactive Simulation">
        Step the real environment forward one decision at a time. Watching never trains the agents - training lives in the RL Lab.
      </PageHeader>

      <details className="settings" open={t === 0}>
        <summary>Settings: market seed, market impact and each agent&apos;s policy</summary>
        <div className="stack" style={{ marginTop: ".8rem" }}>
          <div className="grid">
            <NumberField label="Market seed" value={seedDraft} min={0} max={1_000_000} onChange={setSeedDraft} help="Same seed => the same synthetic market path, so you can compare policies fairly." />
            <Checkbox label="Agents' trades move the equity price (market impact)" checked={impactDraft} onChange={setImpactDraft}
              help="When on, the capital-weighted aggregate equity weight above 40% lowers the equity return for EVERYONE - the interaction channel between agents." />
            <Button disabled={busy} onClick={() => { stop(); void call({ sim: { ...sim, seed: seedDraft, impact: impactDraft, steps: [] } }); }}>
              Apply seed &amp; impact setting (restarts at step 0)
            </Button>
          </div>
          <p className="small"><b>Policy for each agent</b> (can be changed between steps - e.g. take manual control with <i>Fixed / manual basket</i>)</p>
          <div className="grid g3" style={{ alignItems: "start" }}>
            {sim.policies.map((p, i) => (
              <div key={i} className="stack" style={{ gap: ".4rem" }}>
                <b><span className="agent-dot" style={{ background: meta.agent_colors[roleName(i)] }} />{roleName(i)}</b>
                <SelectField<PolicyKind>
                  label="Policy" value={p.kind}
                  options={(["trained", "random", "fixed"] as PolicyKind[]).map((k) => ({ value: k, label: meta.policy_labels[k] }))}
                  onChange={(kind) => setSim((s) => ({ ...s, policies: s.policies.map((q, j) => (j === i ? { ...q, kind } : q)) }))}
                />
                {p.kind === "fixed" ? (
                  <SelectField<number>
                    label="Basket" value={p.basket}
                    options={meta.baskets.map((b, bi) => ({ value: bi, label: `${bi}: ${b.name}` }))}
                    onChange={(basket) => setSim((s) => ({ ...s, policies: s.policies.map((q, j) => (j === i ? { ...q, basket } : q)) }))}
                  />
                ) : null}
              </div>
            ))}
          </div>
        </div>
      </details>

      <div className="toolbar">
        <Button primary disabled={done || playing || busy || !resp} onClick={() => setPlaying(true)}>Start</Button>
        <Button disabled={!playing} onClick={stop}>Pause</Button>
        <Button disabled={done || busy || !resp} onClick={() => { stop(); void call({ advance: 1 }); }}>Step</Button>
        <span style={{ display: "inline-flex", gap: ".4rem", alignItems: "end" }}>
          <div style={{ width: 86 }}><NumberField label="n" value={stepsN} min={1} max={100} onChange={setStepsN} /></div>
          <Button disabled={done || busy || !resp} onClick={() => { stop(); void call({ advance: stepsN }); }}>Step xN</Button>
        </span>
        <Button disabled={done || busy || !resp} onClick={() => { stop(); void call({ toEnd: true }); }}>Run to end</Button>
        <Button disabled={busy} onClick={() => { stop(); void call({ sim: { ...sim, steps: [] } }); }}>Reset</Button>
      </div>
      <div style={{ maxWidth: 420 }}>
        <SliderField label="Auto-play speed (steps per second)" value={speed} min={0.5} max={5} step={0.5} onChange={setSpeed} />
      </div>

      <ErrorBox message={error} />
      {resp ? (
        <div className="grid g3" style={{ alignItems: "center" }}>
          <Progress value={t / resp.episode_length} text={resp.summary_line} />
          <div>
            {done ? "Episode complete" : (<>Next step&apos;s regime: <span className="pill" style={{ background: resp.regime_color }}>{resp.regime_now}</span></>)}
          </div>
          <div>Status: <b>{playing ? "RUNNING" : done ? "DONE" : "PAUSED / READY"}</b> &nbsp; Total: <b>{inr(resp.total_value)}</b></div>
        </div>
      ) : (
        <Spinner text="Starting the simulation..." />
      )}
      <Radio<View> name="simview" value={view} options={VIEWS} onChange={setView} />
      <div style={{ opacity: busy && resp && !playing ? 0.6 : 1, transition: "opacity .15s" }}>
        <Blocks blocks={blocks} />
      </div>
    </div>
  );
}
