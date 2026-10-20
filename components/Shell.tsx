"use client";

import Link from "next/link";
import type { ReactNode } from "react";
import { useStore } from "@/lib/store";
import { MobileBar, Rail } from "./Nav";
import { Skeleton } from "./Controls";

export function Shell({ children }: { children: ReactNode }) {
  const { meta, metaError, retryMeta } = useStore();
  return (
    <div className="app">
      <Rail />
      <div className="content">
        <MobileBar />
        <main className="page">
          <div className="syn-banner" role="note">
            <b>Synthetic data.</b> Every price, return and market regime in this app comes from a <b>simulated</b> Markov-regime market
            model (see the <Link href="/learn">Learn page</Link>) - it is <b>not real market data</b>, and results are <b>not</b> real investment
            performance, advice or predictions. No real-money trading or broker connection exists in this project.
          </div>
          {metaError ? (
            <div className="callout error" role="alert">
              Could not load the app configuration: {metaError}{" "}
              <button type="button" className="btn" onClick={retryMeta}>Try again</button>
            </div>
          ) : !meta ? (
            <Skeleton text="Starting the simulation server. The first request after a quiet spell can take a few seconds." />
          ) : (
            children
          )}
        </main>
      </div>
    </div>
  );
}
