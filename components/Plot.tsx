"use client";

import { useEffect, useRef, useState } from "react";
import type { PlotFigure } from "@/lib/types";

// Plotly is loaded on demand in the browser only (it is large and needs the DOM).
type PlotlyLib = {
  react: (el: HTMLElement, data: unknown[], layout: unknown, config: unknown) => Promise<unknown>;
  purge: (el: HTMLElement) => void;
};
let plotlyPromise: Promise<PlotlyLib> | null = null;
let plotlyLib: PlotlyLib | null = null;
function loadPlotly(): Promise<PlotlyLib> {
  if (!plotlyPromise) {
    plotlyPromise = import("plotly.js-dist-min").then((m) => {
      plotlyLib = (m as unknown as { default: PlotlyLib }).default ?? (m as unknown as PlotlyLib);
      return plotlyLib;
    });
  }
  return plotlyPromise;
}

type Layout = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

/** On narrow screens: wrap the title, give the (wrapping) bottom legend more room and shrink the type. */
function adaptLayout(layout: Layout, width: number): Layout {
  if (width >= 640) return layout;
  const out: Layout = { ...layout };
  const margin = { ...(layout.margin ?? {}) };
  const title = layout.title && typeof layout.title === "object" ? { ...layout.title } : null;
  if (title && typeof title.text === "string" && title.text) {
    const perLine = Math.max(18, Math.floor(width / 7.2));
    const words: string[] = title.text.split(" ");
    const lines: string[] = [];
    let cur = "";
    for (const w of words) {
      if ((cur + " " + w).trim().length > perLine && cur) { lines.push(cur); cur = w; } else cur = (cur + " " + w).trim();
    }
    if (cur) lines.push(cur);
    title.text = lines.join("<br>");
    title.font = { ...(title.font ?? {}), size: 12 };
    margin.t = (margin.t ?? 46) + 18 * (lines.length - 1) + (lines.length > 1 ? 8 : 0);
    out.title = title;
  }
  if (layout.showlegend !== false && typeof margin.b === "number" && margin.b > 40) margin.b += 44;
  out.margin = margin;
  if (layout.height) out.height = (layout.height as number) + (margin.b - (layout.margin?.b ?? 0)) + (margin.t - (layout.margin?.t ?? 0));
  return out;
}

export function Plot({ figure }: { figure: PlotFigure }) {
  const ref = useRef<HTMLDivElement>(null);
  const [failed, setFailed] = useState(false);
  const [width, setWidth] = useState(1000);
  useEffect(() => {
    const update = () => setWidth(ref.current?.clientWidth || window.innerWidth);
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);
  const adapted = adaptLayout(figure.layout as Layout, width);
  const height = typeof adapted.height === "number" ? (adapted.height as number) : 340;

  useEffect(() => {
    let cancelled = false;
    loadPlotly()
      .then((Plotly) => {
        if (cancelled || !ref.current) return;
        const layout = { ...adapted, autosize: true, width: undefined };
        return Plotly.react(ref.current, figure.data, layout, {
          responsive: true,
          displaylogo: false,
          modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
        });
      })
      .catch(() => !cancelled && setFailed(true));
    return () => {
      cancelled = true;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [figure, width < 640]);

  // Free Plotly's resources when the chart leaves the page (synchronously: never races with a re-draw).
  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el && plotlyLib) plotlyLib.purge(el);
    };
  }, []);

  if (failed) return <div className="callout warn">The chart could not be drawn in this browser.</div>;
  return <div ref={ref} className="plot" style={{ height }} role="img" aria-label="Chart" />;
}
