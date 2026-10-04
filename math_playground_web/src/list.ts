// Expression list: rows, diagnostics, inline sliders, reordering. Talks to the engine only via commands.
import type { Engine, DocJson, ItemInfo, Kind, SliderCfg, TableState } from "./engine";
import { TableGrid } from "./tablegrid";
import { MathInput, renderStaticMath } from "./mathinput";
import { announce } from "./announce";
import { isDark } from "./theme";

const SUPPORTED: Kind[] = ["equation", "expression", "complex", "points", "note", "table", "action"];
const KIND_LABEL: Record<string, string> = { equation: "Equation", expression: "Expression", complex: "Complex", points: "Points", note: "Note", table: "Table", action: "Action" };
const PAL_LIGHT = ["#c72e2e", "#2b61b8", "#2e8c4d", "#7345a6", "#e68019", "#1a1a1a"];
const PAL_DARK = ["#fa6b6b", "#73a6ff", "#66d18c", "#bf8cf2", "#ffb34d", "#ebebeb"];
const UNDEF = /undefined variable '([A-Za-z][A-Za-z0-9_]*)'/;

const ICON_GRIP = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="9" cy="6" r="1.4"/><circle cx="15" cy="6" r="1.4"/><circle cx="9" cy="12" r="1.4"/><circle cx="15" cy="12" r="1.4"/><circle cx="9" cy="18" r="1.4"/><circle cx="15" cy="18" r="1.4"/></svg>';
const ICON_PALETTE = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3a9 9 0 1 0 0 18c1.5 0 2-1 1.5-2-.6-1.2.1-2.5 1.6-2.5H17a4 4 0 0 0 4-4C21 6.6 17 3 12 3z"/></svg>';
const ICON_X = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>';
const ICON_PLAY = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5l12 7-12 7z" fill="currentColor"/></svg>';
const ICON_STEP = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5l9 7-9 7z" fill="currentColor"/><path d="M19 5v14" stroke-width="2.5"/></svg>';
const ICON_PAUSE = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14M16 5v14" stroke-width="3"/></svg>';

interface Row {
  id: string; kind: Kind; hidden: boolean; color: string | null; latex: string;
  li: HTMLLIElement; input: MathInput; badge: HTMLButtonElement; diag: HTMLElement; info: HTMLElement; palette: HTMLElement;
  sliderHost: HTMLElement; grid?: TableGrid; handle: HTMLButtonElement; timer: number; message: string; textBtn: HTMLButtonElement;
  runBtn?: HTMLButtonElement; tickBtn?: HTMLButtonElement; engineKind: Kind; palBtn: HTMLButtonElement; tools: HTMLElement; actionError: string;
}
interface SliderState {
  name: string; cfg: SliderCfg; step: number; playing: boolean; dir: 1 | -1; el: HTMLElement;
  range: HTMLInputElement; val: HTMLInputElement; play: HTMLButtonElement;
}

export class ExpressionList {
  private rows: Row[] = [];
  private sliders = new Map<string, SliderState>();
  private textModes = new Set<string>();
  private counter = 0;
  private lastTick = 0;
  private diagSig = "";
  private infoTimer = 0;
  private infoSig = "";
  private residuals = new Set<string>();
  private tables = new Map<string, TableState>();
  private tickerAction: string | null = null;
  private tickerRunning = false;

  constructor(private engine: Engine, private ol: HTMLElement) {
    new MutationObserver(() => this.refreshColors()).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    // The add menu (index.html) predates complex items: add its entry here. main.ts handles the
    // click for any `button[data-kind]`, so this sends `addItem` with kind "complex".
    const menu = document.getElementById("add-menu");
    if (menu && !menu.querySelector('[data-kind="complex"]')) {
      const li = document.createElement("li"); li.setAttribute("role", "none");
      const b = document.createElement("button");
      b.type = "button"; b.setAttribute("role", "menuitem"); b.dataset.kind = "complex"; b.textContent = "Complex";
      li.append(b);
      menu.querySelector('[data-kind="points"]')?.parentElement?.before(li) ?? menu.append(li);
    }
    if (menu && !menu.querySelector('[data-kind="action"]')) {
      const li = document.createElement("li"); li.setAttribute("role", "none");
      const b = document.createElement("button");
      b.type = "button"; b.setAttribute("role", "menuitem"); b.dataset.kind = "action"; b.dataset.latex = "a \\to a+1"; b.textContent = "Action (example)";
      li.append(b);
      menu.querySelector('[data-kind="note"]')?.parentElement?.before(li) ?? menu.append(li);
    }
  }

  get count(): number { return this.rows.length; }

  private nextId(): string {
    let id: string;
    do { id = `r${++this.counter}`; } while (this.rows.some((r) => r.id === id));
    return id;
  }

  // ---- document sync --------------------------------------------------------------------------
  loadFromDoc(doc: DocJson): void {
    for (const r of this.rows) { r.input.destroy(); r.grid?.destroy(); clearTimeout(r.timer); }
    this.rows = [];
    this.ol.replaceChildren();
    for (const s of this.sliders.values()) s.el.remove();
    this.sliders.clear();
    for (const it of doc.items) {
      if (it.kind === "table" && it.table && !this.tables.has(it.id)) {
        const cols = it.table.columns;
        this.tables.set(it.id, { id: it.id, columns: cols, rows: Math.max(0, ...cols.map((c) => c.cells.length)), style: it.table.style ?? "points" });
      }
      this.buildRow({ id: it.id, kind: it.kind, latex: it.latex, hidden: !!it.hidden, color: it.color ?? null });
      const n = /^r(\d+)$/.exec(it.id);
      if (n) this.counter = Math.max(this.counter, Number(n[1]));
    }
    for (const [name, cfg] of Object.entries(doc.sliders ?? {})) this.makeSlider(name, cfg);
    this.placeSliders();
    this.refreshColors();
    this.renumber();
  }

  // ---- rows -----------------------------------------------------------------------------------
  addRow(kind: Kind, latex = "", focus = true, afterIndex = this.rows.length - 1): Row | null {
    const id = this.nextId();
    const evs = this.engine.send({ t: "addItem", id, kind, latex });
    if (evs.some((e) => e.t === "error")) return null;
    const row = this.buildRow({ id, kind, latex, hidden: false, color: null }, afterIndex + 1);
    this.placeSliders();
    this.refreshColors();
    this.renumber();
    if (focus) { row.li.scrollIntoView({ block: "nearest" }); this.focusRow(row); }
    return row;
  }

  /** True when the text has a top-level action arrow (mirrors the engine's `looks_like_action`). */
  private static hasArrow(src: string): boolean {
    let depth = 0;
    for (let i = 0; i < src.length; i++) {
      const rest = src.slice(i);
      if (rest.startsWith("->") || rest.startsWith("\u2192") || rest.startsWith("\u21A6")) { if (depth === 0) return true; continue; }
      if (rest[0] === "\\") {
        const m = /^\\(to|rightarrow|mapsto)(?![a-zA-Z])/.exec(rest);
        if (m) { if (depth === 0) return true; i += m[0].length - 1; continue; }
        const n = /^\\[a-zA-Z]+/.exec(rest);
        if (n) i += n[0].length - 1;
        continue;
      }
      if ("([{".includes(rest[0])) depth++;
      else if (")]}".includes(rest[0])) depth--;
    }
    return false;
  }

  /** Switches an equation row to the action UI when its text gains an arrow, and back when it loses it. */
  private syncKind(row: Row): void {
    if (row.kind !== "equation" && row.kind !== "action") return;
    const want: Kind = ExpressionList.hasArrow(row.latex) ? "action" : "equation";
    if (want === row.kind) return;
    row.kind = want;
    if (want === "equation" && row.engineKind === "action") {
      // An engine item of kind action rejects arrow-less text: re-create it as an equation in place.
      const at = this.rows.indexOf(row);
      this.engine.send({ t: "removeItem", id: row.id });
      this.engine.send({ t: "addItem", id: row.id, kind: "equation", latex: row.latex });
      this.engine.send({ t: "moveItem", id: row.id, to: at });
      if (row.hidden) this.engine.send({ t: "setHidden", id: row.id, hidden: true });
      if (row.color) this.engine.send({ t: "setColor", id: row.id, color: row.color });
      row.engineKind = "equation";
    }
    // (An equation item with an arrow is already run as an action by the engine.)
    this.applyActionUi(row);
    this.renumber();
    announce(want === "action" ? `${this.label(row)} is now an action` : `${this.label(row)} is now an equation`);
  }

  private applyActionUi(row: Row): void {
    const isAction = row.kind === "action";
    if (isAction && !row.runBtn) {
      const run = document.createElement("button");
      run.type = "button"; run.className = "tool run"; run.innerHTML = ICON_STEP; run.title = "Run action once";
      const tick = document.createElement("button");
      tick.type = "button"; tick.className = "tool tick"; tick.innerHTML = ICON_PLAY; tick.title = "Run on ticker";
      run.addEventListener("click", () => this.runAction(row));
      tick.addEventListener("click", () => this.setTickerFor(row, !(this.tickerRunning && this.tickerAction === row.id)));
      row.tools.prepend(run, tick);
      row.runBtn = run; row.tickBtn = tick;
    } else if (!isAction && row.runBtn) {
      if (this.tickerAction === row.id) this.setTickerFor(row, false);
      row.runBtn.remove(); row.tickBtn?.remove(); row.runBtn = undefined; row.tickBtn = undefined;
    }
    row.input.el.querySelector("input, math-field")?.setAttribute("aria-label", `${KIND_LABEL[row.kind] ?? row.kind} input`);
    row.palBtn.hidden = isAction || row.kind === "note";
    this.syncTickerUi();
  }

  private focusRow(row: Row): void { if (row.grid) row.grid.focus(); else row.input.focus(); }

  private buildRow(init: { id: string; kind: Kind; latex: string; hidden: boolean; color: string | null }, at = this.rows.length): Row {
    const li = document.createElement("li");
    li.className = "row";
    li.dataset.id = init.id;
    const main = document.createElement("div");
    main.className = "row-main";

    const handle = document.createElement("button");
    handle.type = "button"; handle.className = "handle"; handle.innerHTML = ICON_GRIP;

    const badge = document.createElement("button");
    badge.type = "button"; badge.className = "badge"; badge.innerHTML = "<span></span>";
    badge.classList.toggle("note", init.kind === "note");

    const supported = SUPPORTED.includes(init.kind);
    const input = new MathInput({
      latex: init.latex, textMode: this.textModes.has(init.id) || !supported || init.kind === "note",
      label: `${KIND_LABEL[init.kind] ?? init.kind} input`,
      onEnter: () => this.enter(row),
    });

    const tools = document.createElement("div");
    tools.className = "tools";
    const textBtn = document.createElement("button");
    textBtn.type = "button"; textBtn.className = "tool"; textBtn.textContent = "Aa";
    textBtn.title = "Toggle plain-text input"; textBtn.setAttribute("aria-pressed", String(input.isTextMode()));
    const palBtn = document.createElement("button");
    palBtn.type = "button"; palBtn.className = "tool"; palBtn.innerHTML = ICON_PALETTE;
    palBtn.title = "Colour"; palBtn.setAttribute("aria-expanded", "false");
    const del = document.createElement("button");
    del.type = "button"; del.className = "tool"; del.innerHTML = ICON_X; del.title = "Delete";
    if (init.kind === "note") palBtn.hidden = true;
    tools.append(textBtn, palBtn, del);

    if (init.kind === "complex") input.el.querySelector("input")?.setAttribute("placeholder", "z^2-1");
    const isTable = init.kind === "table";
    let grid: TableGrid | undefined;
    if (isTable) {
      // A table has no expression input (`input` stays unmounted); its grid goes below the header line.
      const title = document.createElement("span");
      title.className = "row-title"; title.textContent = "Table";
      textBtn.hidden = true;
      main.append(handle, badge, title, tools);
    } else main.append(handle, badge, input.el, tools);

    const diag = document.createElement("div");
    diag.className = "diag"; diag.id = `diag-${init.id}`; diag.hidden = true;
    diag.setAttribute("role", "note");

    const palette = document.createElement("div");
    palette.className = "palette"; palette.hidden = true;

    const sliderHost = document.createElement("div");
    sliderHost.className = "slider-host";
    li.append(main);
    if (isTable) {
      grid = new TableGrid(this.engine, init.id, () => this.label(row));
      li.append(grid.el);
    }
    const info = document.createElement("div");
    info.className = "info"; info.hidden = true;
    li.append(diag, info, palette, sliderHost);

    const row: Row = {
      ...init, engineKind: init.kind, palBtn, tools, li, input, badge, diag, info, palette, sliderHost, handle, timer: 0, message: "", textBtn, grid, actionError: "",
    };
    const known = this.tables.get(init.id);
    if (grid && known) grid.update(known);
    this.rows.splice(at, 0, row);
    this.ol.insertBefore(li, this.ol.children[at] ?? null);

    // Colour palette
    const mk = (color: string | null, label: string): HTMLButtonElement => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "swatch" + (color ? "" : " auto");
      if (color) b.style.setProperty("--c", color);
      b.setAttribute("aria-label", label); b.setAttribute("aria-pressed", String(row.color === color));
      b.addEventListener("click", () => {
        row.color = color;
        this.engine.send({ t: "setColor", id: row.id, color });
        palette.querySelectorAll(".swatch").forEach((s, i) => s.setAttribute("aria-pressed", String(i === (color ? [...PAL_LIGHT].indexOf(color) + 1 : 0))));
        this.refreshColors();
      });
      return b;
    };
    palette.append(mk(null, "Automatic colour"), ...PAL_LIGHT.map((c, i) => mk(c, `Colour ${i + 1}`)));
    if (row.color && PAL_LIGHT.includes(row.color)) palette.querySelectorAll(".swatch")[PAL_LIGHT.indexOf(row.color) + 1]?.setAttribute("aria-pressed", "true");
    else if (!row.color) palette.querySelector(".swatch")?.setAttribute("aria-pressed", "true");
    palBtn.addEventListener("click", () => {
      palette.hidden = !palette.hidden;
      palBtn.setAttribute("aria-expanded", String(!palette.hidden));
    });

    this.applyActionUi(row);
    badge.addEventListener("click", () => {
      row.hidden = !row.hidden;
      this.engine.send({ t: "setHidden", id: row.id, hidden: row.hidden });
      this.refreshColors();
      announce(`${this.label(row)} ${row.hidden ? "hidden" : "shown"}`, { immediate: true });
    });
    del.addEventListener("click", () => this.removeRow(row.id));
    textBtn.addEventListener("click", async () => {
      const text = !input.isTextMode();
      if (text) this.textModes.add(row.id); else this.textModes.delete(row.id);
      await input.setTextMode(text);
      textBtn.setAttribute("aria-pressed", String(input.isTextMode()));
      announce(text ? "Plain text input" : "Math input");
    });
    input.onChange((v) => {
      row.latex = v;
      row.actionError = "";
      this.syncKind(row);
      clearTimeout(row.timer);
      row.timer = window.setTimeout(() => {
        this.engine.send({ t: "setExpr", id: row.id, latex: v });
        this.placeSliders();
        this.renderDiag(row);
      }, 60);
      this.renderDiag(row);
    });

    // Keyboard reorder and drag reorder
    li.addEventListener("keydown", (e) => {
      if (e.altKey && (e.key === "ArrowUp" || e.key === "ArrowDown")) {
        e.preventDefault(); e.stopPropagation();
        this.move(row, this.rows.indexOf(row) + (e.key === "ArrowUp" ? -1 : 1), true);
      }
    }, true);
    handle.addEventListener("keydown", (e) => {
      if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        e.preventDefault();
        this.move(row, this.rows.indexOf(row) + (e.key === "ArrowUp" ? -1 : 1), true, handle);
      }
    });
    this.wireDrag(row);
    input.describedBy("");
    this.renderDiag(row);
    return row;
  }

  private label(row: Row): string { return `Row ${this.rows.indexOf(row) + 1}`; }

  private renumber(): void {
    this.rows.forEach((r, i) => {
      r.handle.setAttribute("aria-label", `Reorder row ${i + 1}. Use arrow keys, or Alt plus arrow keys from the input.`);
      r.badge.setAttribute("aria-label", `${r.hidden ? "Show" : "Hide"} row ${i + 1}`);
      r.badge.setAttribute("aria-pressed", String(!r.hidden));
      r.li.setAttribute("aria-label", r.kind === "table" ? `Row ${i + 1}, table` : r.kind === "action" ? `Row ${i + 1}, action` : `Row ${i + 1}`);
    });
    for (const r of this.rows) r.grid?.relabel();
    this.syncTickerUi();
  }

  private enter(row: Row): void {
    const i = this.rows.indexOf(row);
    if (i === this.rows.length - 1) this.addRow("equation", "", true);
    else this.focusRow(this.rows[i + 1]);
  }

  removeRow(id: string): void {
    const i = this.rows.findIndex((r) => r.id === id);
    if (i < 0) return;
    const row = this.rows[i];
    this.engine.send({ t: "removeItem", id });
    row.input.destroy(); row.grid?.destroy(); clearTimeout(row.timer);
    this.tables.delete(id);
    // Park any slider hosted here before the row disappears.
    for (const s of this.sliders.values()) if (row.sliderHost.contains(s.el)) s.el.remove();
    row.li.remove();
    this.rows.splice(i, 1);
    this.textModes.delete(id); this.residuals.delete(id);
    this.placeSliders(); this.renumber();
    const next = this.rows[Math.min(i, this.rows.length - 1)];
    if (next) this.focusRow(next);
    announce(`Row deleted. ${this.rows.length} rows.`, { immediate: true });
  }

  private move(row: Row, to: number, announceIt: boolean, refocus?: HTMLElement): void {
    const from = this.rows.indexOf(row);
    to = Math.max(0, Math.min(this.rows.length - 1, to));
    if (to === from) return;
    const r = this.engine.send({ t: "moveItem", id: row.id, to });
    if (r.some((e) => e.t === "error")) return;
    this.rows.splice(from, 1);
    this.rows.splice(to, 0, row);
    this.ol.insertBefore(row.li, this.ol.children[to + (to > from ? 1 : 0)] ?? null);
    this.renumber(); this.refreshColors();
    if (refocus) refocus.focus(); else this.focusRow(row);
    if (announceIt) announce(`Row moved to position ${to + 1} of ${this.rows.length}`, { immediate: true });
  }

  private wireDrag(row: Row): void {
    let startY = 0; let target = -1; let active = false;
    const clear = () => this.rows.forEach((r) => r.li.classList.remove("drop-before", "drop-after"));
    row.handle.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      active = true; startY = e.clientY; target = this.rows.indexOf(row);
      row.handle.setPointerCapture(e.pointerId);
      row.li.classList.add("dragging");
    });
    row.handle.addEventListener("pointermove", (e) => {
      if (!active) return;
      row.li.style.transform = `translateY(${e.clientY - startY}px)`;
      clear();
      let t = this.rows.length - 1;
      for (let i = 0; i < this.rows.length; i++) {
        const r = this.rows[i]; if (r === row) continue;
        const b = r.li.getBoundingClientRect();
        if (e.clientY < b.top + b.height / 2) { t = i - (i > this.rows.indexOf(row) ? 1 : 0); break; }
      }
      target = t;
      const ref = this.rows.filter((r) => r !== row)[t];
      if (ref) ref.li.classList.add(t >= this.rows.indexOf(row) ? "drop-after" : "drop-before");
    });
    const end = () => {
      if (!active) return;
      active = false;
      row.li.classList.remove("dragging"); row.li.style.transform = "";
      clear();
      this.move(row, target, true, row.handle);
    };
    row.handle.addEventListener("pointerup", end);
    row.handle.addEventListener("pointercancel", () => { active = false; row.li.classList.remove("dragging"); row.li.style.transform = ""; clear(); });
  }

  // ---- engine-driven updates ------------------------------------------------------------------
  /** A `table` event: the engine's table is the truth; cache it (events can precede the row) and render. */
  onTable(t: TableState): void {
    this.tables.set(t.id, t);
    this.rows.find((r) => r.id === t.id)?.grid?.update(t);
  }

  /** A drag rewrote an item's text: show it without sending `setExpr` back (no feedback loop). */
  onItemEdited(id: string, latex: string): void {
    const row = this.rows.find((r) => r.id === id);
    if (!row || row.kind === "table") return;
    clearTimeout(row.timer);
    row.latex = latex;
    row.input.setLatex(latex);
    this.placeSliders(); this.renderDiag(row); this.refreshColors();
  }

  /** A drag moved a slider: reflect it in the range, number box and aria-valuetext. */
  onSliderValue(name: string, value: number): void {
    const s = this.sliders.get(name);
    if (!s) return;
    s.cfg.value = value;
    if (value < s.cfg.min || value > s.cfg.max) {
      s.cfg.min = Math.min(s.cfg.min, value); s.cfg.max = Math.max(s.cfg.max, value);
      s.range.min = String(s.cfg.min); s.range.max = String(s.cfg.max);
    }
    s.range.value = String(value);
    if (document.activeElement !== s.val) s.val.value = String(round(value));
    s.range.setAttribute("aria-valuetext", String(round(value)));
  }

  // ---- colours --------------------------------------------------------------------------------
  /** Resolved colours from the engine's `colors` event (source of truth once received). */
  private engineColors: Map<string, string> | null = null;
  onColors(items: { id: string; color: string }[]): void {
    this.engineColors = new Map(items.map((i) => [i.id, i.color]));
    this.refreshColors();
  }

  refreshColors(): void {
    const ec = this.engineColors;
    const fb = ec ? null : this.fallbackColors();
    for (const r of this.rows) {
      let c: string | null | undefined;
      if (ec) {
        c = ec.get(r.id);
        if (!c && r.color) c = isDark() && PAL_LIGHT.includes(r.color) ? PAL_DARK[PAL_LIGHT.indexOf(r.color)] : r.color;
      } else c = fb!.get(r);
      r.badge.style.setProperty("--c", c ?? "var(--muted)");
      r.badge.setAttribute("aria-pressed", String(!r.hidden));
      r.li.classList.toggle("hidden-row", r.hidden);
    }
    this.renumber();
  }

  /** Local guess of the engine's automatic colouring; used only before the first `colors` event. */
  private fallbackColors(): Map<Row, string | null> {
    const pal = isDark() ? PAL_DARK : PAL_LIGHT;
    const isDef = (r: Row) => {
      const m = /^\s*([A-Za-z]\w*)\s*(\([^)]*\))?\s*=/.exec(r.latex);
      return !!m && (!!m[2] || !["x", "y", "z", "r"].includes(m[1]));
    };
    const drawsInOrder = (r: Row) => !["note", "action", "folder", "table"].includes(r.kind) && r.latex.trim() !== "";
    const auto = new Map<Row, number>();
    let idx = 0;
    for (const r of this.rows) if (!r.hidden && drawsInOrder(r) && !isDef(r)) auto.set(r, idx++);
    for (const r of this.rows) if (!r.hidden && r.kind === "table") auto.set(r, idx++);
    const out = new Map<Row, string | null>();
    for (const r of this.rows) {
      let c = r.color;
      const i = auto.get(r);
      if (!c && i !== undefined) c = pal[i % pal.length];
      if (r.color && isDark()) { const k = PAL_LIGHT.indexOf(r.color); if (k >= 0) c = PAL_DARK[k]; }
      out.set(r, c);
    }
    return out;
  }

  // ---- actions and ticker ---------------------------------------------------------------------
  /** Runs an action row once; an engine error is shown on that row. */
  private runAction(row: Row): void {
    const evs = this.engine.send({ t: "runAction", id: row.id });
    const err = evs.find((e) => e.t === "error");
    row.actionError = err && err.t === "error" ? err.message : "";
    this.renderDiag(row);
    if (row.actionError) announce(`${this.label(row)}: ${row.actionError}`, { immediate: true });
  }

  private setTickerFor(row: Row, running: boolean): void {
    row.actionError = "";
    const evs = this.engine.send({ t: "setTicker", action: row.id, running });
    const err = evs.find((e) => e.t === "error");
    if (err && err.t === "error") { row.actionError = err.message; this.renderDiag(row); }
    else this.renderDiag(row);
  }

  /** `tickerState` event: reflect running/paused on the action rows and announce changes. */
  onTickerState(running: boolean, action: string | null): void {
    const changed = running !== this.tickerRunning || action !== this.tickerAction;
    this.tickerRunning = running; this.tickerAction = action;
    this.syncTickerUi();
    if (changed) {
      const row = this.rows.find((r) => r.id === action);
      announce(running ? `Ticker started${row ? ": " + this.label(row) : ""}` : "Ticker stopped", { immediate: true });
    }
  }

  /** A `ticker: ...` error belongs to the ticker's action row. Returns false when there is none. */
  onTickerError(message: string): boolean {
    const row = this.rows.find((r) => r.id === this.tickerAction);
    if (!row) return false;
    row.actionError = message.replace(/^ticker:\s*/, "Ticker: ");
    this.renderDiag(row);
    announce(`${this.label(row)}: ${row.actionError}`, { immediate: true });
    return true;
  }

  private syncTickerUi(): void {
    this.rows.forEach((r, i) => {
      if (!r.tickBtn || !r.runBtn) return;
      const on = this.tickerRunning && this.tickerAction === r.id;
      r.tickBtn.innerHTML = on ? ICON_PAUSE : ICON_PLAY;
      r.tickBtn.setAttribute("aria-pressed", String(on));
      r.tickBtn.setAttribute("aria-label", `Run row ${i + 1} on ticker`);
      r.tickBtn.title = on ? "Stop ticker" : "Run on ticker";
      r.runBtn.setAttribute("aria-label", `Run row ${i + 1} once`);
    });
  }

  // ---- info read-outs -------------------------------------------------------------------------
  /** `items` is the full current list; rows not mentioned are cleared. */
  onInfo(items: ItemInfo[]): void {
    const by = new Map(items.map((i) => [i.id, i]));
    for (const r of this.rows) this.renderInfo(r, by.get(r.id));
    const f = (n: number | undefined) => (n === undefined ? "" : String(Math.round(n * 1e4) / 1e4));
    const summary = (i: ItemInfo) => [i.text ?? i.latex ?? i.kind, i.kind === "regression" && i.r2 !== undefined ? `r squared ${f(i.r2)}` : ""].filter(Boolean).join(", ");
    const sig = items.map((i) => `${i.id}:${summary(i)}`).join("|");
    // Debounced so a slider drag announces once, after it settles.
    clearTimeout(this.infoTimer);
    this.infoTimer = window.setTimeout(() => {
      if (sig === this.infoSig) return;
      const had = this.infoSig !== "";
      this.infoSig = sig;
      if (items.length) announce(items.map((i) => { const r = this.rows.find((x) => x.id === i.id); return `${r ? this.label(r) : i.id}: ${summary(i)}`; }).join(". "));
      else if (had) announce("Read-outs cleared");
    }, 400);
  }

  private renderInfo(row: Row, info: ItemInfo | undefined): void {
    const host = row.info;
    if (!info) { host.hidden = true; host.replaceChildren(); return; }
    host.hidden = false;
    host.dataset.kind = info.kind;
    let math = host.querySelector<HTMLElement>(".info-math");
    if (!math) { math = document.createElement("div"); math.className = "info-math"; host.prepend(math); }
    if (math.dataset.latex !== (info.latex ?? "") || math.dataset.text !== (info.text ?? "")) {
      math.dataset.latex = info.latex ?? ""; math.dataset.text = info.text ?? "";
      renderStaticMath(math, info.latex ?? "", info.text ?? "");
    }
    const f = (n: number) => String(Math.round(n * 1e4) / 1e4);
    let stats = host.querySelector<HTMLElement>(".info-stats");
    let box = host.querySelector<HTMLInputElement>("input.info-resid");
    if (info.kind !== "regression") { stats?.remove(); host.querySelector(".info-resid-label")?.remove(); return; }
    if (!stats) { stats = document.createElement("div"); stats.className = "info-stats"; host.append(stats); }
    const bits: string[] = [];
    if (info.r2 !== undefined) bits.push(`r\u00b2 = ${f(info.r2)}`);
    if (info.rmse !== undefined) bits.push(`rmse = ${f(info.rmse)}`);
    if (info.n !== undefined) bits.push(`n = ${info.n}`);
    stats.textContent = bits.join("  ");
    if (!box) {
      const label = document.createElement("label"); label.className = "info-resid-label";
      box = document.createElement("input"); box.type = "checkbox"; box.className = "info-resid";
      label.append(box, " residuals");
      host.append(label);
      const id = row.id;
      box.addEventListener("change", () => {
        if (box!.checked) this.residuals.add(id); else this.residuals.delete(id);
        this.engine.send({ t: "setRegressionResiduals", id, on: box!.checked });
      });
    }
    box.checked = this.residuals.has(row.id);
    box.setAttribute("aria-label", `Show residuals for ${this.label(row)}`);
  }

  // ---- diagnostics ----------------------------------------------------------------------------
  onDiagnostics(items: { id: string; message: string }[]): void {
    const by = new Map(items.map((i) => [i.id, i.message]));
    for (const r of this.rows) { r.message = by.get(r.id) ?? ""; this.renderDiag(r); }
    const sig = items.map((i) => `${i.id}:${i.message}`).join("|");
    if (sig !== this.diagSig) {
      this.diagSig = sig;
      const shown = this.rows.filter((r) => r.message && (r.latex.trim() || r.kind === "table"));
      announce(shown.length ? shown.map((r) => `${this.label(r)}: ${r.message}`).join(". ") : "No expression errors");
    }
  }

  private renderDiag(row: Row): void {
    const empty = !row.latex.trim() && row.kind !== "table";
    const msg = empty ? "" : (row.actionError || row.message);
    const unsupported = !SUPPORTED.includes(row.kind);
    row.diag.replaceChildren();
    row.diag.classList.toggle("note", !msg);
    if (msg) {
      const t = document.createElement("span"); t.textContent = msg; row.diag.append(t);
      const m = UNDEF.exec(msg);
      if (m && !this.sliders.has(m[1])) {
        const chip = document.createElement("button");
        chip.type = "button"; chip.className = "chip"; chip.textContent = `Make slider for ${m[1]}`;
        chip.addEventListener("click", () => this.createSlider(m[1]));
        row.diag.append(chip);
      }
    } else if (unsupported) {
      row.diag.textContent = `${row.kind} items are not supported yet.`;
    }
    const show = !!msg || unsupported;
    row.diag.hidden = !show;
    row.input.describedBy(show ? row.diag.id : "");
  }

  // ---- sliders --------------------------------------------------------------------------------
  createSlider(name: string): void {
    const cfg: SliderCfg = { min: -10, max: 10, value: 1 };
    this.engine.send({ t: "setSlider", name, value: 1, min: -10, max: 10, step: 0.1 });
    this.makeSlider(name, cfg);
    this.placeSliders();
    announce(`Slider for ${name} added`, { immediate: true });
    this.sliders.get(name)?.range.focus();
  }

  private makeSlider(name: string, cfg: SliderCfg): void {
    const el = document.createElement("div");
    el.className = "slider-box";
    el.setAttribute("role", "group"); el.setAttribute("aria-label", `Slider ${name}`);
    const top = document.createElement("div"); top.className = "slider-top";
    const lab = document.createElement("label"); lab.textContent = name; lab.htmlFor = `sl-${name}`;
    const range = document.createElement("input");
    range.type = "range"; range.id = `sl-${name}`;
    const val = document.createElement("input"); val.type = "number"; val.className = "slider-val"; val.setAttribute("aria-label", `${name} value`);
    const play = document.createElement("button");
    play.type = "button"; play.className = "tool play"; play.innerHTML = ICON_PLAY; play.setAttribute("aria-label", `Play ${name}`);
    const rm = document.createElement("button");
    rm.type = "button"; rm.className = "tool"; rm.innerHTML = ICON_X; rm.title = "Remove slider"; rm.setAttribute("aria-label", `Remove slider ${name}`);
    top.append(lab, range, val, play, rm);
    const bounds = document.createElement("div"); bounds.className = "slider-bounds";
    const num = (label: string, v: number) => {
      const l = document.createElement("label"); l.textContent = label + " ";
      const i = document.createElement("input"); i.type = "number"; i.value = String(v); i.step = "any";
      i.setAttribute("aria-label", `${name} ${label}`); l.append(i); bounds.append(l); return i;
    };
    const step = cfg.step ?? 0.1;
    const minI = num("min", cfg.min), maxI = num("max", cfg.max), stepI = num("step", step);
    el.append(top, bounds);
    const s: SliderState = { name, cfg: { ...cfg }, step, playing: false, dir: 1, el, range, val, play };
    this.sliders.set(name, s);
    const syncRange = () => {
      range.min = String(s.cfg.min); range.max = String(s.cfg.max); range.step = String(s.step || "any");
      range.value = String(s.cfg.value); val.value = String(round(s.cfg.value));
      range.setAttribute("aria-valuetext", String(round(s.cfg.value)));
    };
    syncRange();
    const push = (withBounds: boolean) => this.engine.send(withBounds
      ? { t: "setSlider", name, value: s.cfg.value, min: s.cfg.min, max: s.cfg.max, step: s.step }
      : { t: "setSlider", name, value: s.cfg.value });
    range.addEventListener("input", () => { s.cfg.value = Number(range.value); val.value = String(round(s.cfg.value)); push(false); });
    val.addEventListener("change", () => {
      const v = Number(val.value); if (!Number.isFinite(v)) return syncRange();
      s.cfg.value = v;
      if (v < s.cfg.min) { s.cfg.min = v; minI.value = String(v); }
      if (v > s.cfg.max) { s.cfg.max = v; maxI.value = String(v); }
      syncRange(); push(true);
    });
    const bound = () => {
      const lo = Number(minI.value), hi = Number(maxI.value);
      if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo >= hi) { minI.value = String(s.cfg.min); maxI.value = String(s.cfg.max); return; }
      s.cfg.min = lo; s.cfg.max = hi; s.cfg.value = Math.min(hi, Math.max(lo, s.cfg.value));
      syncRange(); push(true);
    };
    minI.addEventListener("change", bound); maxI.addEventListener("change", bound);
    stepI.addEventListener("change", () => { const v = Number(stepI.value); s.step = v > 0 ? v : 0.1; syncRange(); push(true); });
    rm.addEventListener("click", () => {
      this.engine.send({ t: "removeSlider", name });
      this.sliders.delete(name); el.remove();
      announce(`Slider for ${name} removed`, { immediate: true });
    });
    play.addEventListener("click", () => {
      s.playing = !s.playing;
      play.innerHTML = s.playing ? ICON_PAUSE : ICON_PLAY;
      play.setAttribute("aria-label", `${s.playing ? "Pause" : "Play"} ${name}`);
    });
  }

  /** Puts each slider under the first row that mentions it, else under the last row. Moves nodes, never rebuilds. */
  private placeSliders(): void {
    for (const s of this.sliders.values()) {
      const re = new RegExp(`(?<![A-Za-z\\\\])${s.name}(?![A-Za-z0-9_])`);
      const owner = this.rows.find((r) => re.test(r.latex)) ?? this.rows[this.rows.length - 1];
      if (owner && s.el.parentElement !== owner.sliderHost) owner.sliderHost.append(s.el);
    }
  }

  /** Called every animation frame; advances playing sliders. */
  tick(now: number): void {
    const dt = this.lastTick ? Math.min(now - this.lastTick, 100) : 0;
    this.lastTick = now;
    for (const s of this.sliders.values()) {
      if (!s.playing || !dt) continue;
      const span = s.cfg.max - s.cfg.min;
      let v = s.cfg.value + s.dir * (span / 4000) * dt;
      if (v >= s.cfg.max) { v = s.cfg.max; s.dir = -1; } else if (v <= s.cfg.min) { v = s.cfg.min; s.dir = 1; }
      s.cfg.value = v;
      s.range.value = String(v); s.val.value = String(round(v));
      this.engine.send({ t: "setSlider", name: s.name, value: v });
    }
  }
}

function round(v: number): number { return Math.round(v * 1000) / 1000; }
