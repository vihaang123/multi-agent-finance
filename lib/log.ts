import type { Log } from "./types";

/** Append newly trained episode rows (column lists) to the training log held in the browser. */
export function appendLog(prev: Log | null, rows: Log): Log {
  if (!prev) return rows;
  const out: Log = {};
  for (const k of Object.keys(rows)) out[k] = (prev[k] ?? []).concat(rows[k]);
  return out;
}

export const inr = (x: number): string => {
  const neg = x < 0;
  const s = Math.round(Math.abs(x)).toString();
  let out = s;
  if (s.length > 3) {
    let head = s.slice(0, -3);
    const tail = s.slice(-3);
    const groups: string[] = [];
    while (head.length > 2) {
      groups.unshift(head.slice(-2));
      head = head.slice(0, -2);
    }
    if (head) groups.unshift(head);
    out = [...groups, tail].join(",");
  }
  return `${neg ? "-" : ""}₹${out}`;
};
