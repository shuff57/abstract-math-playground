import "./style.css";
import { Engine, type EngineEvent, type Kind } from "./engine";
import { ExpressionList } from "./list";
import { CanvasController } from "./canvas";
import { LabelOverlay } from "./overlay";
import { Layout } from "./layout";
import { initTheme, isDark, setTheme } from "./theme";
import { announce, toast } from "./announce";
import { SliceControl } from "./slice";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

function showFailure(err: unknown): void {
  const el = $("fail");
  const msg = err instanceof Error ? err.message : String(err);
  el.hidden = false;
  el.replaceChildren();
  const h = document.createElement("h2"); h.textContent = "The graph could not start";
  const p = document.createElement("p");
  p.textContent = "This app draws with WebGL (via WebAssembly). Your browser or device did not provide it, or it was blocked.";
  const d = document.createElement("p"); d.style.cssText = "font:12px var(--mono);color:var(--muted)"; d.textContent = msg;
  el.append(h, p, d);
}

async function main(): Promise<void> {
  const layout = new Layout();
  const canvas = $<HTMLCanvasElement>("canvas");
  const stage = $("stage");

  // The canvas needs its real size before the GPU surface is created.
  const r = stage.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(r.width * dpr));
  canvas.height = Math.max(1, Math.round(r.height * dpr));

  let engine: Engine;
  try {
    engine = await Engine.create(canvas);
  } catch (err) {
    showFailure(err);
    return;
  }

  const list = new ExpressionList(engine, $("rows"));
  const controller = new CanvasController(engine, canvas, stage);
  const overlay = new LabelOverlay(engine, $("labels"), canvas);
  const slice = new SliceControl(engine, canvas, () => overlay.update());
  overlay.sliceRect = () => slice.rectCss();
  controller.onResize = () => overlay.update();
  void layout;

  // ---- engine events ------------------------------------------------------------------------
  let mode = "2d";
  let ortho = false;
  let descTimer = 0;
  let lastView = { mode: "2d", min: [-10, -10, -10], max: [10, 10, 10] };
  const fmt = (n: number) => String(Math.round(n * 100) / 100);
  const describe = () => {
    const v = lastView;
    const axes = v.mode === "1d" ? [0] : v.mode === "2d" ? [0, 1] : [0, 1, 2];
    const names = ["x", "y", "z"];
    const win = axes.map((a) => `${names[a]} from ${fmt(v.min[a])} to ${fmt(v.max[a])}`).join(", ");
    return `${v.mode.toUpperCase()} graph, ${win}. ${list.count} expression${list.count === 1 ? "" : "s"}.`;
  };
  const updateDescription = () => {
    clearTimeout(descTimer);
    descTimer = window.setTimeout(() => {
      const d = describe();
      canvas.setAttribute("aria-label", d);
      announce(d);
    }, 700);
  };
  const setModeUi = (m: string) => {
    mode = m;
    document.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === m)));
    $("three-tools").hidden = m !== "3d";
  };
  // ---- ticker bar ---------------------------------------------------------------------------
  const tBar = $("ticker-bar"), tToggle = $<HTMLButtonElement>("ticker-toggle"), tRate = $<HTMLInputElement>("ticker-rate");
  let tickerRunning = false;
  let tickerAction: string | null = null;
  const onTickerState = (running: boolean, action: string | null) => {
    tickerRunning = running; tickerAction = action;
    list.onTickerState(running, action);
    tBar.hidden = !action;
    tToggle.setAttribute("aria-pressed", String(running));
    tToggle.setAttribute("aria-label", `${running ? "Stop" : "Start"} ticker (Space)`);
    $("ticker-lbl").textContent = running ? "Pause" : "Play";
  };
  const toggleTicker = () => engine.send({ t: "setTicker", running: !tickerRunning });
  tToggle.addEventListener("click", toggleTicker);
  tRate.addEventListener("change", () => {
    const v = Number(tRate.value);
    if (!Number.isFinite(v) || v <= 0) { tRate.value = "50"; return; }
    engine.send({ t: "setTicker", rateMs: v });
  });
  engine.on((e: EngineEvent) => {
    switch (e.t) {
      case "diagnostics": list.onDiagnostics(e.items); break;
      case "info": list.onInfo(e.items); break;
      case "colors": list.onColors(e.items); break;
      case "view": {
        const modeChanged = e.mode !== lastView.mode;
        lastView = { mode: e.mode, min: e.min, max: e.max };
        if (e.mode !== mode) setModeUi(e.mode);
        void modeChanged;
        slice.onView(e.mode, e.min, e.max);
        updateDescription();
        break;
      }
      case "slice": slice.onSlice(e); break;
      case "table": list.onTable(e); break;
      case "itemEdited": list.onItemEdited(e.id, e.latex); break;
      case "sliderValue": list.onSliderValue(e.name, e.value); break;
      case "tickerState": onTickerState(e.running, e.action); break;
      case "error":
        // Ticker errors show on the action row (and are announced there); others as a toast.
        if (e.message.startsWith("ticker: ") && list.onTickerError(e.message)) break;
        toast(e.message, 4000);
        break;
      default: break;
    }
  });

  // ---- theme (single place) -----------------------------------------------------------------
  initTheme(engine);

  // ---- reduced motion: mode switches jump instead of animating -------------------------------
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const sendReducedMotion = () => engine.send({ t: "setReducedMotion", on: reduceMotion.matches });
  sendReducedMotion();
  reduceMotion.addEventListener("change", sendReducedMotion);

  // ---- initial document ---------------------------------------------------------------------
  let loaded = false;
  if (location.hash.startsWith("#v1.")) {
    const evs = engine.send({ t: "loadHash", hash: location.hash.slice(1) });
    if (!evs.some((e) => e.t === "error")) {
      const { doc } = engine.exportDoc();
      if (doc) { list.loadFromDoc(doc); loaded = true; slice.syncFromDoc(doc); }
    } else {
      toast("That share link could not be opened. Starting with a blank graph.", 5000);
    }
  }
  if (!loaded || list.count === 0) list.addRow("equation", "", false);
  announce(`Graphing Playground ready. ${engine.backend} renderer.`);

  // ---- toolbar ------------------------------------------------------------------------------
  document.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach((b) =>
    b.addEventListener("click", () => { slice.beforeMode(b.dataset.mode!); engine.send({ t: "setMode", mode: b.dataset.mode }); }));
  document.querySelectorAll<HTMLButtonElement>("[data-ortho]").forEach((b) =>
    b.addEventListener("click", () => setOrtho(b.dataset.ortho === "true")));
  const setOrtho = (o: boolean) => {
    ortho = o;
    engine.send({ t: "setOrtho", ortho: o });
    document.querySelectorAll<HTMLButtonElement>("[data-ortho]").forEach((x) => x.setAttribute("aria-pressed", String((x.dataset.ortho === "true") === o)));
    announce(o ? "Orthographic projection" : "Perspective projection", { immediate: true });
  };
  $("reset-btn").addEventListener("click", () => { engine.send({ t: "reset" }); announce("View reset", { immediate: true }); });

  $("share-btn").addEventListener("click", async () => {
    const { hash } = engine.exportDoc();
    if (!hash) { toast("Nothing to share yet."); return; }
    const url = `${location.origin}${location.pathname}#${hash}`;
    history.replaceState(null, "", `#${hash}`);
    try {
      await navigator.clipboard.writeText(url);
      toast("Share link copied to clipboard");
      announce("Share link copied to clipboard", { immediate: true });
    } catch {
      // Clipboard blocked: show the link so it can be copied by hand.
      const inp = document.createElement("input");
      inp.readOnly = true; inp.value = url; inp.setAttribute("aria-label", "Share link");
      toast("Copy this link: ", 12000, inp);
      inp.focus(); inp.select();
    }
  });

  // ---- add menu -----------------------------------------------------------------------------
  const addBtn = $("add-btn"), menu = $("add-menu");
  const closeMenu = (refocus = false) => { menu.hidden = true; addBtn.setAttribute("aria-expanded", "false"); if (refocus) addBtn.focus(); };
  addBtn.addEventListener("click", () => {
    const open = menu.hidden;
    menu.hidden = !open; addBtn.setAttribute("aria-expanded", String(open));
    if (open) (menu.querySelector("button") as HTMLElement).focus();
  });
  menu.addEventListener("click", (e) => {
    const b = (e.target as HTMLElement).closest("button[data-kind]") as HTMLElement | null;
    if (!b) return;
    closeMenu();
    list.addRow(b.dataset.kind as Kind, b.dataset.latex ?? "", true);
  });
  menu.addEventListener("keydown", (e) => {
    const items = [...menu.querySelectorAll<HTMLElement>("button")];
    const i = items.indexOf(document.activeElement as HTMLElement);
    if (e.key === "ArrowDown") { e.preventDefault(); items[(i + 1) % items.length].focus(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); items[(i - 1 + items.length) % items.length].focus(); }
    else if (e.key === "Escape") { e.preventDefault(); closeMenu(true); }
  });
  document.addEventListener("pointerdown", (e) => {
    if (!menu.hidden && !(e.target as HTMLElement).closest(".add-wrap")) closeMenu();
  });

  // ---- global shortcuts (never while typing) -------------------------------------------------
  window.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || e.defaultPrevented) return;
    const t = e.target as HTMLElement;
    if (t.closest("input, textarea, select, math-field, [contenteditable='true']")) return;
    if (e.key === " " && !t.closest("button, a, [role='button'], [role='menuitem']")) {
      if (tickerAction) { e.preventDefault(); toggleTicker(); }
      return;
    }
    if (e.key === "1") { slice.beforeMode("1d"); engine.send({ t: "setMode", mode: "1d" }); }
    else if (e.key === "2") { slice.beforeMode("2d"); engine.send({ t: "setMode", mode: "2d" }); }
    else if (e.key === "3") { slice.beforeMode("3d"); engine.send({ t: "setMode", mode: "3d" }); }
    else if (e.key === "r" || e.key === "R") engine.send({ t: "reset" });
    else if (e.key === "d" || e.key === "D") setTheme(!isDark());
    else return;
  });
  void ortho;

  // ---- render loop --------------------------------------------------------------------------
  let failed = false;
  const tick = (now: number) => {
    requestAnimationFrame(tick);
    if (failed) return;
    try {
      list.tick(now);
      if (engine.frame(now)) overlay.update();
    } catch (err) {
      failed = true;
      showFailure(err);
    }
  };
  requestAnimationFrame(tick);
  document.documentElement.dataset.ready = "1";
}

main().catch(showFailure);
