"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Blocks } from "@/components/Blocks";
import { Button, ErrorBox, NumberField, Spinner } from "@/components/Controls";
import { PageHeader } from "@/components/PageHeader";
import { useSimulation } from "@/lib/hooks";
import { useMeta, useStore } from "@/lib/store";

export default function DashboardPage() {
  const meta = useMeta();
  const { agents, sim } = useStore();
  const { resp, blocks, busy, error, call } = useSimulation("dashboard");
  const [seed, setSeed] = useState(sim.seed);
  useEffect(() => setSeed(sim.seed), [sim.seed]);

  return (
    <div className="stack">
      <PageHeader title="Multi-Agent Financial Resource Allocation">
        Four independent Q-learning agents share one synthetic market and split Rs 10,00,000 into four sleeves of Rs 2,50,000. This page
        summarises the episode currently loaded in the simulator.
      </PageHeader>
      <div className="grid">
        <NumberField
          label="Market seed" value={seed} min={0} max={1_000_000} onChange={setSeed}
          help={`Same seed = same synthetic market path. Seeds ${meta.eval_seed_start}-${meta.eval_seed_start + meta.n_eval_episodes - 1} are the held-out evaluation block used on the Experiments page; the agents were trained on seed ${meta.train_seed}.`}
        />
        <Button primary disabled={busy} onClick={() => call({ sim: { ...sim, seed, steps: [] }, toEnd: true })} title="Simulates all steps of one episode on this seed with the current policies.">
          Run a full episode
        </Button>
        <Button disabled={busy} onClick={() => call({ sim: { ...sim, seed, steps: [] } })}>Clear (step 0)</Button>
        <Link className="btn" href="/simulation" style={{ textDecoration: "none", textAlign: "center" }}>Step through it in the Simulation page</Link>
      </div>
      <p className="small muted">
        Agents: <b>{agents.label}</b> &nbsp;|&nbsp; Market seed <b>{resp?.seed ?? sim.seed}</b> &nbsp;|&nbsp; {resp?.summary_line ?? "..."}
      </p>
      <ErrorBox message={error} />
      {!resp && busy ? <Spinner text="Simulating an episode..." /> : null}
      <div style={{ opacity: busy && resp ? 0.6 : 1, transition: "opacity .15s" }}>
        <Blocks blocks={blocks} />
      </div>
    </div>
  );
}
