"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { applyPref, readPref, type ThemePref } from "@/lib/theme";
import { useStore } from "@/lib/store";
import { RegimeRibbon } from "./RegimeRibbon";

const PAGES = [
  { href: "/", label: "Dashboard", hint: "Overview of the current episode" },
  { href: "/simulation", label: "Simulation", hint: "Step through decisions one at a time" },
  { href: "/rl-lab", label: "RL Lab", hint: "Train agents and open their Q-tables" },
  { href: "/analytics", label: "Portfolio Analytics", hint: "Returns, risk and costs" },
  { href: "/experiments", label: "Experiments", hint: "Compare four ways to allocate" },
  { href: "/learn", label: "Learn", hint: "How the market, states and rewards work" },
];

export function BrandMark() {
  return (
    <svg className="brand-mark" width="28" height="28" viewBox="0 0 28 28" aria-hidden>
      <rect x="1" y="12" width="5" height="15" rx="1.6" fill="#2a78d6" />
      <rect x="8" y="5" width="5" height="22" rx="1.6" fill="#eb6834" />
      <rect x="15" y="9" width="5" height="18" rx="1.6" fill="#1baf7a" />
      <rect x="22" y="2" width="5" height="25" rx="1.6" fill="#6250d6" />
    </svg>
  );
}

function Brand() {
  return (
    <Link href="/" className="brand" aria-label="Allocation Lab, home">
      <BrandMark />
      <span className="brand-name">
        Allocation Lab
        <span className="brand-sub">Four Q-learning agents, one synthetic market</span>
      </span>
    </Link>
  );
}

function ThemeToggle() {
  const [pref, setPref] = useState<ThemePref>("system");
  useEffect(() => setPref(readPref()), []);
  const choose = (p: ThemePref) => {
    setPref(p);
    applyPref(p);
  };
  return (
    <div className="radio" role="group" aria-label="Colour theme">
      {(["system", "light", "dark"] as ThemePref[]).map((p) => (
        <button
          key={p}
          type="button"
          className={pref === p ? "radio-item active" : "radio-item"}
          aria-pressed={pref === p}
          onClick={() => choose(p)}
          style={{ border: 0, background: pref === p ? undefined : "none", font: "inherit", fontSize: ".78rem", padding: ".2rem .6rem" }}
        >
          {p === "system" ? "Auto" : p === "light" ? "Light" : "Dark"}
        </button>
      ))}
    </div>
  );
}

function Links() {
  const path = usePathname();
  return (
    <nav aria-label="Pages">
      {PAGES.map((p) => (
        <Link key={p.href} href={p.href} title={p.hint} className={path === p.href ? "navlink active" : "navlink"} aria-current={path === p.href ? "page" : undefined}>
          {p.label}
        </Link>
      ))}
    </nav>
  );
}

export function Rail() {
  const { meta, agents, live, resetSession } = useStore();
  return (
    <aside className="rail" aria-label="Sidebar">
      <Brand />
      <Links />
      <div className="session">
        <div>
          <span className="muted">Active agents</span>
          <div className="agents-label"><b>{agents.label}</b></div>
        </div>
        {meta && live ? (
          <div>
            <span className="muted">Episode</span> <b>step {live.t}/{meta.episode_length}</b>
            <div style={{ marginTop: ".35rem" }}><RegimeRibbon regimes={live.regimes} total={meta.episode_length} mini /></div>
          </div>
        ) : null}
        <div className="rail-actions">
          <button type="button" className="linkbtn" onClick={resetSession} title="Forget the trained agents and the episode saved in this browser">Reset session</button>
        </div>
        <ThemeToggle />
      </div>
    </aside>
  );
}

export function MobileBar() {
  return (
    <header className="mobilebar">
      <div className="row">
        <Brand />
        <ThemeToggle />
      </div>
      <Links />
    </header>
  );
}
