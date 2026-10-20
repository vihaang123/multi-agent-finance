"use client";

import { useEffect, useState, type ReactNode } from "react";
import { useStore } from "@/lib/store";
import { Blocks as BlocksLazy } from "./Blocks";

export function Field({ label, help, children }: { label: string; help?: string; children: ReactNode }) {
  return (
    <label className="field" title={help}>
      <span>{label}</span>
      {children}
    </label>
  );
}

/** Number input that lets you type freely and commits a clamped value on blur/Enter. */
export function NumberField({
  label, value, onChange, min, max, step = 1, help, disabled,
}: { label: string; value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number; help?: string; disabled?: boolean }) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  const commit = () => {
    let v = Number(text);
    if (!Number.isFinite(v) || text.trim() === "") v = value;
    if (min !== undefined) v = Math.max(min, v);
    if (max !== undefined) v = Math.min(max, v);
    if (step >= 1) v = Math.round(v);
    setText(String(v));
    if (v !== value) onChange(v);
  };
  return (
    <Field label={label} help={help}>
      <input
        type="number" inputMode="decimal" value={text} min={min} max={max} step={step} disabled={disabled}
        onChange={(e) => setText(e.target.value)} onBlur={commit} onKeyDown={(e) => e.key === "Enter" && commit()}
      />
    </Field>
  );
}

export function SliderField({
  label, value, onChange, min, max, step, format, help, disabled,
}: { label: string; value: number; onChange: (v: number) => void; min: number; max: number; step: number; format?: (v: number) => string; help?: string; disabled?: boolean }) {
  return (
    <Field label={`${label}: ${format ? format(value) : value}`} help={help}>
      <input type="range" min={min} max={max} step={step} value={value} disabled={disabled} onChange={(e) => onChange(Number(e.target.value))} />
    </Field>
  );
}

export function SelectField<T extends string | number>({
  label, value, options, onChange, help, disabled,
}: { label: string; value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; help?: string; disabled?: boolean }) {
  return (
    <Field label={label} help={help}>
      <select
        value={String(value)} disabled={disabled}
        onChange={(e) => {
          const o = options.find((x) => String(x.value) === e.target.value);
          if (o) onChange(o.value);
        }}
      >
        {options.map((o) => (
          <option key={String(o.value)} value={String(o.value)}>{o.label}</option>
        ))}
      </select>
    </Field>
  );
}

export function Checkbox({ label, checked, onChange, help, disabled }: { label: string; checked: boolean; onChange: (v: boolean) => void; help?: string; disabled?: boolean }) {
  return (
    <label className="check" title={help}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

export function Radio<T extends string>({ value, options, onChange, name }: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; name: string }) {
  return (
    <div className="radio" role="radiogroup">
      {options.map((o) => (
        <label key={o.value} className={o.value === value ? "radio-item active" : "radio-item"}>
          <input type="radio" name={name} checked={o.value === value} onChange={() => onChange(o.value)} />
          <span>{o.label}</span>
        </label>
      ))}
    </div>
  );
}

export function Button({
  children, onClick, primary, disabled, title,
}: { children: ReactNode; onClick?: () => void; primary?: boolean; disabled?: boolean; title?: string }) {
  return (
    <button type="button" className={primary ? "btn primary" : "btn"} onClick={onClick} disabled={disabled} title={title}>
      {children}
    </button>
  );
}

export function Progress({ value, text }: { value: number; text?: string }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="progress-wrap">
      {text ? <div className="progress-text">{text}</div> : null}
      <div className="progress" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100} aria-label={text ?? "Progress"}>
        <div className="bar" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function Panel({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <section className="panel">
      {title ? <h3>{title}</h3> : null}
      {children}
    </section>
  );
}

export function Spinner({ text = "Working..." }: { text?: string }) {
  return (
    <div className="spinner" role="status">
      <span className="dot" /> {text}
    </div>
  );
}

/** Placeholder shown while a page's first answer is on its way. */
export function Skeleton({ text = "Loading..." }: { text?: string }) {
  return (
    <div className="stack" role="status" aria-label={text}>
      <div className="skel skel-kpis" />
      <div className="skel skel-plot" />
      <p className="caption">{text}</p>
    </div>
  );
}

/** Error message. When the problem looks like a bad saved training state it offers a one-click way back. */
export function ErrorBox({ message }: { message: string | null }) {
  const { agents, restoreReference } = useStore();
  if (!message) return null;
  const stateProblem = agents.source === "lab" && /state|agents|layout|version|shape|truncated|decode/i.test(message);
  return (
    <div className="callout error" role="alert">
      {message}
      {stateProblem ? (
        <button type="button" className="btn" onClick={restoreReference}>Switch back to the reference agents</button>
      ) : null}
    </div>
  );
}

export function TabBar({ tabs, active, onChange }: { tabs: string[]; active: number; onChange: (i: number) => void }) {
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    e.preventDefault();
    const next = (active + (e.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    onChange(next);
    const buttons = (e.currentTarget as HTMLElement).querySelectorAll<HTMLElement>('[role="tab"]');
    buttons[next]?.focus();
  };
  return (
    <div className="tablist" role="tablist" onKeyDown={onKey}>
      {tabs.map((t, i) => (
        <button key={t} role="tab" aria-selected={i === active} tabIndex={i === active ? 0 : -1} className={i === active ? "tab active" : "tab"} onClick={() => onChange(i)}>
          {t}
        </button>
      ))}
    </div>
  );
}

export function QueryBlocks({ q, empty }: { q: { blocks: import("@/lib/types").Block[] | null; loading: boolean; error: string | null }; empty?: string }) {
  // Renders the result of useBlocksQuery with loading / error states.
  return (
    <div className="stack">
      <ErrorBox message={q.error} />
      {q.loading && !q.blocks ? <Skeleton text="Computing..." /> : null}
      {q.blocks ? <div style={{ opacity: q.loading ? 0.55 : 1, transition: "opacity .15s" }}><BlocksLazy blocks={q.blocks} /></div> : !q.loading && !q.error && empty ? <p className="caption">{empty}</p> : null}
    </div>
  );
}

