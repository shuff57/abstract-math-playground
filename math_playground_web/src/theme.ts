// One place owns the theme: CSS variables hang off <html data-theme>, the engine is told via setTheme.
import type { Engine } from "./engine";

const KEY = "gp-theme";
let engine: Engine | null = null;
let stored: "light" | "dark" | null = null;
const mq = matchMedia("(prefers-color-scheme: dark)");

function readStored(): "light" | "dark" | null {
  try { const v = localStorage.getItem(KEY); return v === "dark" || v === "light" ? v : null; } catch { return null; }
}

export function isDark(): boolean { return document.documentElement.dataset.theme === "dark"; }

function apply(dark: boolean): void {
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  const meta = document.querySelector('meta[name="theme-color"]');
  const bg = getComputedStyle(document.documentElement).getPropertyValue("--panel").trim();
  if (meta && bg) meta.setAttribute("content", bg);
  const btn = document.getElementById("theme-btn");
  btn?.setAttribute("aria-pressed", String(dark));
  btn?.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  engine?.send({ t: "setTheme", dark });
}

export function setTheme(dark: boolean, persist = true): void {
  if (persist) {
    stored = dark ? "dark" : "light";
    try { localStorage.setItem(KEY, stored); } catch { /* storage unavailable */ }
  }
  apply(dark);
}

export function initTheme(e: Engine): void {
  engine = e;
  stored = readStored();
  apply(stored ? stored === "dark" : mq.matches);
  mq.addEventListener("change", () => { if (!stored) apply(mq.matches); });
  document.getElementById("theme-btn")!.addEventListener("click", () => setTheme(!isDark()));
}

export function initThemeButtonOnly(): void {
  // Before the engine exists, still reflect the pre-paint theme on the button.
  apply(isDark());
}
