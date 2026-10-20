"use client";

import { useState } from "react";
import type { Block, Kpi } from "@/lib/types";
import { useStore } from "@/lib/store";
import { Inline, Markdown } from "./Markdown";
import { Plot } from "./Plot";

function KpiCard({ k }: { k: Kpi }) {
  return (
    <div className="kpi">
      <div className="lbl">{k.label}</div>
      <div className={`val ${k.tone === "pos" ? "pos" : k.tone === "neg" ? "neg" : ""}`}>
        {k.pill ? (
          <span className="pill" style={{ background: k.pill }}>
            {k.value}
          </span>
        ) : (
          k.value
        )}
      </div>
      {k.sub ? <div className="sub">{k.sub}</div> : null}
    </div>
  );
}

function TableView({ b }: { b: Extract<Block, { type: "table" }> }) {
  const { meta } = useStore();
  const colors = meta?.agent_colors ?? {};
  return (
    <div className="table-wrap">
      <div className="table-scroll" style={b.max_height ? { maxHeight: b.max_height } : undefined}>
        <table>
          <thead>
            <tr>
              {b.columns.map((c, i) => (
                <th key={i}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {b.rows.map((r, i) => (
              <tr key={i}>
                {r.map((v, j) => {
                  const dot = j === 0 && typeof v === "string" ? colors[v] : undefined;
                  return (
                    <td key={j} className={j === 0 ? "first" : undefined}>
                      {dot ? <span className="agent-dot" style={{ background: dot }} /> : null}
                      {fmt(v)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {b.caption ? <p className="caption" style={{ marginTop: ".4rem" }}>{b.caption}</p> : null}
    </div>
  );
}

function fmt(v: string | number | null): string {
  if (v === null || v === undefined) return "–";
  return String(v);
}

function Download({ b }: { b: Extract<Block, { type: "download" }> }) {
  const onClick = () => {
    const url = URL.createObjectURL(new Blob([b.content], { type: b.mime }));
    const a = document.createElement("a");
    a.href = url;
    a.download = b.filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };
  return (
    <button type="button" className="btn" onClick={onClick}>
      {b.label}
    </button>
  );
}

function Tabs({ tabs }: { tabs: { label: string; blocks: Block[] }[] }) {
  const [i, setI] = useState(0);
  const idx = Math.min(i, tabs.length - 1);
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    e.preventDefault();
    const next = (idx + (e.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    setI(next);
    (e.currentTarget as HTMLElement).querySelectorAll<HTMLElement>('[role="tab"]')[next]?.focus();
  };
  return (
    <div className="tabs">
      <div className="tablist" role="tablist" onKeyDown={onKey}>
        {tabs.map((t, j) => (
          <button key={t.label} role="tab" aria-selected={j === idx} tabIndex={j === idx ? 0 : -1} className={j === idx ? "tab active" : "tab"} onClick={() => setI(j)}>
            {t.label}
          </button>
        ))}
      </div>
      <div className="tabpanel" role="tabpanel">
        <Blocks blocks={tabs[idx].blocks} />
      </div>
    </div>
  );
}

function Switch({ label, options }: { label: string; options: { label: string; blocks: Block[] }[] }) {
  const [i, setI] = useState(0);
  const idx = Math.min(i, options.length - 1);
  return (
    <div className="switch">
      <label className="field inline">
        <span>{label}</span>
        <select value={idx} onChange={(e) => setI(Number(e.target.value))}>
          {options.map((o, j) => (
            <option key={o.label} value={j}>
              {o.label}
            </option>
          ))}
        </select>
      </label>
      <Blocks blocks={options[idx].blocks} />
    </div>
  );
}

export function BlockView({ b }: { b: Block }) {
  switch (b.type) {
    case "heading": {
      const L = b.level ?? 3;
      const Tag = (L <= 2 ? "h2" : L === 3 ? "h3" : L === 4 ? "h4" : "h5") as "h2" | "h3" | "h4" | "h5";
      return <Tag className="block-heading">{b.text}</Tag>;
    }
    case "markdown":
      return <Markdown text={b.text} />;
    case "caption":
      return (
        <p className="caption">
          <Inline text={b.text} />
        </p>
      );
    case "callout":
      return (
        <div className={`callout ${b.tone}`} role={b.tone === "error" ? "alert" : "note"}>
          <Inline text={b.text} />
        </div>
      );
    case "kpis":
      return (
        <div className="kpis">
          {b.items.map((k) => (
            <KpiCard key={k.label} k={k} />
          ))}
        </div>
      );
    case "plot":
      return <Plot figure={b.figure} />;
    case "table":
      return <TableView b={b} />;
    case "columns":
      return (
        <div className="cols" style={{ gridTemplateColumns: b.cols.map((c) => `minmax(0, ${c.weight}fr)`).join(" ") }}>
          {b.cols.map((c, i) => (
            <div key={i} className="col">
              <Blocks blocks={c.blocks} />
            </div>
          ))}
        </div>
      );
    case "tabs":
      return <Tabs tabs={b.tabs} />;
    case "switch":
      return <Switch label={b.label} options={b.options} />;
    case "cards":
      return (
        <div className="card-grid">
          {b.items.map((c) => (
            <div key={c.title} className="card" style={{ ["--dot" as string]: c.color ?? "var(--line2)" }}>
              <h4>
                <span className="agent-dot" style={{ background: c.color ?? "#2a78d6" }} />
                {c.title} {c.tag ? <span className="small muted">- {c.tag}</span> : null}
              </h4>
              {c.body ? <Markdown text={c.body} className="small" /> : null}
              {c.blocks ? <Blocks blocks={c.blocks} /> : null}
            </div>
          ))}
        </div>
      );
    case "download":
      return <Download b={b} />;
    case "math":
      return <pre className="math">{b.text}</pre>;
    default:
      return null;
  }
}

export function Blocks({ blocks }: { blocks: Block[] | null | undefined }) {
  if (!blocks) return null;
  return (
    <div className="blocks">
      {blocks.map((b, i) => (
        <BlockView key={i} b={b} />
      ))}
    </div>
  );
}
