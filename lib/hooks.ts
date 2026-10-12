"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, errorText } from "./api";
import { useStore } from "./store";
import type { Block, SimResponse } from "./types";

export type SimView = "dashboard" | "overview" | "decisions" | "money" | "log" | "analytics";

/** Rebuild-and-advance the simulation on the server and keep the page's blocks in sync. */
export function useSimulation(view: SimView, volWindow = 10) {
  const { sim, setSim, agents, meta } = useStore();
  const [resp, setResp] = useState<SimResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [latency, setLatency] = useState(300);
  const seq = useRef(0);
  const simRef = useRef(sim);
  simRef.current = sim;

  const call = useCallback(
    async (opts: { sim?: typeof sim; advance?: number; toEnd?: boolean } = {}) => {
      const s = opts.sim ?? simRef.current;
      const id = ++seq.current;
      const t0 = performance.now();
      setBusy(true);
      setError(null);
      try {
        const r = await api<SimResponse>("/api/simulate", {
          seed: s.seed,
          impact: s.impact,
          policies: s.policies,
          steps: s.steps ?? [],
          advance: opts.advance ?? 0,
          to_end: !!opts.toEnd,
          view,
          vol_window: volWindow,
          agents: agents.state,
        });
        if (id !== seq.current) return null;
        setResp(r);
        setLatency(performance.now() - t0);
        setSim((prev) => ({ ...prev, seed: s.seed, impact: s.impact, steps: r.steps }));
        return r;
      } catch (e) {
        if (id === seq.current) setError(errorText(e));
        return null;
      } finally {
        if (id === seq.current) setBusy(false);
      }
    },
    [view, volWindow, agents.state, setSim],
  );

  // (Re)load whenever the view, the active agents or the analytics window change. A missing episode is played to the end once.
  const ready = !!meta;
  useEffect(() => {
    if (!ready) return;
    void call(simRef.current.steps === null ? { toEnd: true } : {});
  }, [call, ready]);

  const blocks: Block[] = resp?.blocks ?? [];
  return { resp, blocks, busy, error, call, latency };
}

/** Fetch `path` with `build()` as the body whenever `deps` change (debounced). */
export function useBlocksQuery(path: string, enabled: boolean, build: () => unknown, deps: unknown[], delay = 250) {
  const [blocks, setBlocks] = useState<Block[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [extra, setExtra] = useState<Record<string, unknown>>({});
  const buildRef = useRef(build);
  buildRef.current = build;

  useEffect(() => {
    if (!enabled) return;
    const ctl = new AbortController();
    const timer = setTimeout(async () => {
      setLoading(true);
      setError(null);
      try {
        const r = await api<{ blocks: Block[] } & Record<string, unknown>>(path, buildRef.current(), ctl.signal);
        setBlocks(r.blocks);
        setExtra(r);
      } catch (e) {
        if ((e as Error).name !== "AbortError") setError(errorText(e));
      } finally {
        if (!ctl.signal.aborted) setLoading(false);
      }
    }, delay);
    return () => {
      clearTimeout(timer);
      ctl.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, enabled, delay, ...deps]);

  return { blocks, loading, error, extra };
}

/** A button-triggered request. */
export function useAction<T>() {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = useCallback(async (path: string, body: unknown) => {
    setLoading(true);
    setError(null);
    try {
      const r = await api<T>(path, body);
      setData(r);
      return r;
    } catch (e) {
      setError(errorText(e));
      return null;
    } finally {
      setLoading(false);
    }
  }, []);
  return { data, loading, error, run, reset: () => setData(null) };
}
