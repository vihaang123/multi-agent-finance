"use client";

// App-wide state kept in the browser (the server is stateless): which agents are active, the simulation in
// progress, and the latest live experiment results. The agents you trained and the episode you were on are
// saved in this browser (localStorage), so reloading the page no longer throws them away.

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, errorText } from "./api";
import type { AgentsState, ExpResults, Meta, SimState } from "./types";

export type LiveSim = { t: number; total: number; regimes: number[]; regimeNow: string };

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
  live: LiveSim | null;
  setLive: (l: LiveSim | null) => void;
  resetSession: () => void;
};

const Ctx = createContext<Store | null>(null);

const STORAGE_KEY = "marl.session.v2";
const MAX_LOG_CHARS = 900_000; // the learning-curve history is dropped from the saved copy beyond this size

const defaultPolicies = () => Array.from({ length: 4 }, () => ({ kind: "trained" as const, basket: 2 }));

type Saved = { v: 2; stateVersion: number; agents: AgentsState | null; sim: SimState };

function readSaved(stateVersion: number): Saved | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const s = JSON.parse(raw) as Saved;
    if (!s || s.v !== 2 || s.stateVersion !== stateVersion || !s.sim || !Array.isArray(s.sim.policies) || s.sim.policies.length !== 4) return null;
    if (s.agents && (s.agents.source !== "lab" || typeof s.agents.state !== "string")) return { ...s, agents: null };
    return s;
  } catch {
    return null;
  }
}

function writeSaved(s: Saved) {
  const attempt = (payload: Saved) => localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  try {
    const logSize = s.agents?.log ? JSON.stringify(s.agents.log).length : 0;
    attempt(logSize > MAX_LOG_CHARS && s.agents ? { ...s, agents: { ...s.agents, log: null } } : s);
  } catch {
    try {
      attempt({ ...s, agents: s.agents ? { ...s.agents, log: null } : null });
    } catch {
      try {
        localStorage.removeItem(STORAGE_KEY);
      } catch {
        /* storage unavailable: the session just lasts for this page view */
      }
    }
  }
}

export function StoreProvider({ children }: { children: ReactNode }) {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [agents, setAgentsRaw] = useState<AgentsState>({ source: "reference", state: null, label: "Loading agents...", log: null, totalEpisodes: 0 });
  const [sim, setSimRaw] = useState<SimState>({ seed: 51, impact: true, policies: defaultPolicies(), steps: null });
  const [expResults, setExpResults] = useState<ExpResults>(null);
  const [live, setLive] = useState<LiveSim | null>(null);
  const [attempt, setAttempt] = useState(0);
  const hydrated = useRef(false);

  useEffect(() => {
    let cancelled = false;
    setMetaError(null);
    api<Meta>("/api/meta")
      .then((m) => {
        if (cancelled) return;
        setMeta(m);
        const saved = hydrated.current ? null : readSaved(m.state_version);
        if (saved) {
          setSimRaw(saved.sim);
          setAgentsRaw(saved.agents ?? { source: "reference", state: null, label: m.reference.label, log: null, totalEpisodes: 0 });
        } else {
          setSimRaw((s) => ({ ...s, seed: m.eval_seed_start }));
          setAgentsRaw((a) => (a.source === "reference" ? { ...a, label: m.reference.label } : a));
        }
        hydrated.current = true;
      })
      .catch((e) => !cancelled && setMetaError(errorText(e)));
    return () => {
      cancelled = true;
    };
  }, [attempt]);

  // Save the session whenever it changes (after the first load, so an empty state never overwrites a saved one).
  useEffect(() => {
    if (!meta || !hydrated.current) return;
    const id = setTimeout(
      () => writeSaved({ v: 2, stateVersion: meta.state_version, agents: agents.source === "lab" ? agents : null, sim }),
      400,
    );
    return () => clearTimeout(id);
  }, [meta, agents, sim]);

  const setAgents = useCallback((a: AgentsState) => {
    setAgentsRaw(a);
    // Different agents -> the episode on screen is stale: the next page that needs it plays a fresh example episode.
    setSimRaw((s) => ({ ...s, steps: null }));
  }, []);

  const restoreReference = useCallback(() => {
    setAgentsRaw({ source: "reference", state: null, label: meta?.reference.label ?? "Reference agents", log: null, totalEpisodes: 0 });
    setSimRaw((s) => ({ ...s, steps: null }));
  }, [meta]);

  const resetSession = useCallback(() => {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* nothing saved to remove */
    }
    setAgentsRaw({ source: "reference", state: null, label: meta?.reference.label ?? "Reference agents", log: null, totalEpisodes: 0 });
    setSimRaw({ seed: meta?.eval_seed_start ?? 51, impact: true, policies: defaultPolicies(), steps: null });
    setExpResults(null);
    setLive(null);
  }, [meta]);

  const setSim = useCallback((s: SimState | ((prev: SimState) => SimState)) => setSimRaw(s), []);

  const value = useMemo<Store>(
    () => ({
      meta, metaError, retryMeta: () => setAttempt((n) => n + 1), agents, setAgents, restoreReference, sim, setSim,
      expResults, setExpResults, live, setLive, resetSession,
    }),
    [meta, metaError, agents, setAgents, restoreReference, sim, setSim, expResults, live, resetSession],
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
