"use client";

import { useEffect, useState } from "react";

export type Theme = "light" | "dark";
export type ThemePref = "system" | "light" | "dark";
const KEY = "marl.theme";

export function readPref(): ThemePref {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

export function applyPref(p: ThemePref) {
  const root = document.documentElement;
  if (p === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", p);
  try {
    if (p === "system") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, p);
  } catch {
    /* storage unavailable: the choice just lasts for this page view */
  }
}

export function resolveTheme(): Theme {
  const a = document.documentElement.getAttribute("data-theme");
  if (a === "dark" || a === "light") return a;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** The theme currently in effect (follows the toggle and the operating system). */
export function useResolvedTheme(): Theme {
  const [theme, setTheme] = useState<Theme>("light");
  useEffect(() => {
    const update = () => setTheme(resolveTheme());
    update();
    const mo = new MutationObserver(update);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", update);
    return () => {
      mo.disconnect();
      mq.removeEventListener("change", update);
    };
  }, []);
  return theme;
}

// The Python figures are drawn with the light palette; these maps re-colour them for the active theme.
// (the replacement is spliced into JSON text, so it must not contain double quotes)
const COMMON_FONT: [string, string] = ["Inter, system-ui, -apple-system, Segoe UI, sans-serif", "IBM Plex Sans, system-ui, sans-serif"];
const MAPS: Record<Theme, [string, string][]> = {
  light: [
    ["#0b0b0b", "#1a2530"], ["#52514e", "#4c5b68"], ["#8a8985", "#7c8a96"], ["#e6e5e1", "#d6dde3"], ["#fcfcfb", "#eaeef0"], COMMON_FONT,
  ],
  dark: [
    ["#0b0b0b", "#e7edf2"], ["#52514e", "#a0aeba"], ["#8a8985", "#6f7e8b"], ["#e6e5e1", "#25313b"], ["#fcfcfb", "#0e141a"],
    ["#6250d6", "#8f82f5"], ["#2a78d6", "#4d97ee"], ["#eb6834", "#f2835a"], ["#9a9994", "#8d9aa6"], COMMON_FONT,
  ],
};

export function recolour<T>(value: T, theme: Theme): T {
  let s = JSON.stringify(value);
  for (const [from, to] of MAPS[theme]) s = s.split(from).join(to);
  return JSON.parse(s) as T;
}
