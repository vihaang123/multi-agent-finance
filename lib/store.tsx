"use client";

// App-wide state kept in the browser (the server is stateless): which agents are active, the simulation in
// progress, and the latest live experiment results. Reloading the page returns to the reference agents.

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api, errorText } from "./api";
import type { AgentsState, ExpResults, Meta, SimState } from "./types";

type Store = {
  meta: Meta | null;
  metaError: string | null;
  retryMeta: () => void;
  agents: AgentsState;
  setAgents: (a: AgentsState) => void;
  restoreReference: () => void;
  sim: SimState;
  setSim: (s: SimState | ((prev: SimState) => SimState)) => void;
  expResults: ExpResults;
  setExpResults: (r: ExpResults) => void;
};

const Ctx = createContext<Store | null>(null);

const defaultPolicies = () => Array.from({ length: 4 }, () => ({ kind: "trained" as const, basket: 2 }));

export function StoreProvider({ children }: { children: ReactNode }) {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [agents, setAgentsRaw] = useState<AgentsState>({ source: "reference", state: null, label: "Loading agents...", log: null, totalEpisodes: 0 });
  const [sim, setSimRaw] = useState<SimState>({ seed: 51, impact: true, policies: defaultPolicies(), steps: null });
  const [expResults, setExpResults] = useState<ExpResults>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setMetaError(null);
    api<Meta>("/api/meta")
      .then((m) => {
        if (cancelled) return;
        setMeta(m);
        setSimRaw((s) => ({ ...s, seed: m.eval_seed_start }));
        setAgentsRaw((a) => (a.source === "reference" ? { ...a, label: m.reference.label } : a));
      })
      .catch((e) => !cancelled && setMetaError(errorText(e)));
    return () => {
      cancelled = true;
    };
  }, [attempt]);

  const setAgents = useCallback((a: AgentsState) => {
    setAgentsRaw(a);
    // Different agents -> the episode on screen is stale: the next page that needs it plays a fresh example episode.
    setSimRaw((s) => ({ ...s, steps: null }));
  }, []);

  const restoreReference = useCallback(() => {
    setAgentsRaw({ source: "reference", state: null, label: meta?.reference.label ?? "Reference agents", log: null, totalEpisodes: 0 });
    setSimRaw((s) => ({ ...s, steps: null }));
  }, [meta]);

  const setSim = useCallback((s: SimState | ((prev: SimState) => SimState)) => setSimRaw(s), []);

  const value = useMemo<Store>(
    () => ({ meta, metaError, retryMeta: () => setAttempt((n) => n + 1), agents, setAgents, restoreReference, sim, setSim, expResults, setExpResults }),
    [meta, metaError, agents, setAgents, restoreReference, sim, setSim, expResults],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useStore(): Store {
  const s = useContext(Ctx);
  if (!s) throw new Error("StoreProvider missing");
  return s;
}

export function useMeta(): Meta {
  const { meta } = useStore();
  if (!meta) throw new Error("Configuration not loaded yet");
  return meta;
}
