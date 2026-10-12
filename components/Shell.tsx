"use client";

import type { ReactNode } from "react";
import { useStore } from "@/lib/store";
import { Nav } from "./Nav";
import { Spinner } from "./Controls";

export function Shell({ children }: { children: ReactNode }) {
  const { meta, metaError, retryMeta } = useStore();
  return (
    <>
      <Nav />
      <main className="page">
        <div className="syn-banner" role="note">
          <b>Synthetic data.</b> Every price, return and market regime in this app comes from a <b>simulated</b> Markov-regime market
          model (see the Learn page) - it is <b>not real market data</b>, and results are <b>not</b> real investment performance, advice
          or predictions. No real-money trading or broker connection exists in this project.
        </div>
        {metaError ? (
          <div className="callout error" role="alert">
            Could not load the app configuration: {metaError}{" "}
            <button type="button" className="btn" onClick={retryMeta}>Retry</button>
          </div>
        ) : !meta ? (
          <Spinner text="Starting the simulation server..." />
        ) : (
          children
        )}
      </main>
    </>
  );
}
