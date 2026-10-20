"use client";

import { useMeta } from "@/lib/store";

/**
 * The market's mood across one episode, one cell per step, coloured by regime. Steps that have not been played
 * yet stay blank, and a playhead marks where the simulation is. Colours match the shaded bands in every chart.
 */
export function RegimeRibbon({ regimes, total, mini = false }: { regimes: number[]; total: number; mini?: boolean }) {
  const meta = useMeta();
  const shown = Math.min(regimes.length, total);
  const counts = meta.regimes.map((_, r) => regimes.slice(0, shown).filter((x) => x === r).length);
  const summary = shown
    ? `Market regimes over the ${shown} steps played so far: ${meta.regimes.map((n, r) => `${counts[r]} ${n}`).join(", ")}.`
    : "No steps played yet.";
  return (
    <div className={mini ? undefined : "ribbon-wrap"}>
      <div className={mini ? "ribbon mini" : "ribbon"} style={{ gridTemplateColumns: `repeat(${total}, minmax(0, 1fr))` }} role="img" aria-label={summary}>
        {Array.from({ length: total }, (_, i) => {
          const r = i < shown ? regimes[i] : -1;
          return (
            <span
              key={i}
              className="cell"
              style={r >= 0 ? { background: meta.regime_colors[meta.regimes[r]] } : undefined}
              title={r >= 0 ? `Step ${i + 1}: ${meta.regimes[r]}` : undefined}
            />
          );
        })}
        {shown > 0 && shown < total ? <span className="playhead" style={{ left: `calc(${(shown / total) * 100}% - 1px)` }} /> : null}
      </div>
      {mini ? null : (
        <div className="ribbon-legend">
          {meta.regimes.map((n, r) => (
            <span key={n}>
              <span className="sw" style={{ background: meta.regime_colors[n] }} />
              {n} {shown ? <b style={{ color: "var(--ink)", fontWeight: 600 }}>{counts[r]}</b> : null}
            </span>
          ))}
          <span className="grow">{shown ? `Step ${shown} of ${total}` : `${total} steps, none played yet`}</span>
        </div>
      )}
    </div>
  );
}
