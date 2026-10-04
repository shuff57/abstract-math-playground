// Slice control (toggle, axis, value, range) and the inset frame drawn over the canvas.
// The engine is the source of truth: every "slice" event re-syncs the controls.
import type { Engine, EngineEvent, DocJson } from "./engine";
import { announce } from "./announce";

export interface SliceEvent {
  t: "slice"; active: boolean; dim: number; fixed: Record<string, number>; free: string[];
  rect: [number, number, number, number] | null; curves: number; points: number; error: string | null;
}

const AXES = ["x", "y", "z"];
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const fmt = (n: number) => String(Math.round(n * 1000) / 1000);

export class SliceControl {
  private mode = "2d";
  private prevMode = "2d";
  private switching = false; // a mode switch was requested: the engine's re-derived slice is not the user's intent
  private win = { min: [-10, -10, -10], max: [10, 10, 10] };
  private enabled = false;
  private resuming = false;
  private staleErr = false; // the mode-switch error still queued behind a resume
  private want = false; // the user's intent; survives a mode the engine cannot slice in (1D)
  private axis = "y";
  private text = "0";
  // Last axis/value the user had in each slicing mode, restored when that mode comes back (a 2D x/y
  // choice and a 3D plane choice are separate intents; the engine's mode-forced axis never overwrites them).
  private mem: Record<string, { axis: string; text: string }> = {};
  private custom = false; // a multi-axis slice from a loaded doc, edited as raw "y=1,z=0.5"
  private sendTimer = 0;
  private annTimer = 0;
  private lastAnn = "";
  private last: SliceEvent | null = null;

  private toggle = $<HTMLButtonElement>("slice-toggle");
  private panel = $("slice-panel");
  private axisBtns = $("slice-axes");
  private input = $<HTMLInputElement>("slice-value");
  private range = $<HTMLInputElement>("slice-range");
  private status = $("slice-status");
  private frame = $("slice-frame");
  private title = $("slice-title");
  private close = $<HTMLButtonElement>("slice-close");
  private dpr = () => this.canvas.width / Math.max(this.canvas.clientWidth, 1);

  constructor(private engine: Engine, private canvas: HTMLCanvasElement, private onRect: () => void) {
    this.toggle.addEventListener("click", () => {
      this.enabled = !this.enabled; this.want = this.enabled;
      if (this.enabled && !this.axisValid()) this.axis = this.defaultAxis();
      this.refresh();
      this.queue(true);
    });
    this.axisBtns.addEventListener("click", (e) => {
      const b = (e.target as HTMLElement).closest("button[data-axis]") as HTMLElement | null;
      if (!b) return;
      this.custom = false;
      this.axis = b.dataset.axis!;
      this.enabled = true; this.want = true;
      this.refresh();
      this.queue();
    });
    this.input.addEventListener("input", () => {
      this.text = this.input.value;
      const n = Number(this.text);
      if (this.text.trim() !== "" && Number.isFinite(n)) this.range.value = String(n);
      if (this.text.trim() === "") return;
      this.enabled = true; this.want = true;
      this.refresh();
      this.queue();
    });
    this.range.addEventListener("input", () => {
      this.text = fmt(Number(this.range.value));
      this.input.value = this.text;
      this.enabled = true; this.want = true;
      this.refresh();
      this.queue();
    });
    this.close.addEventListener("click", () => { this.enabled = false; this.want = false; this.refresh(); this.queue(true); this.toggle.focus(); });
    // The inset is read-only: nothing pressed, dragged or scrolled over it may reach the canvas.
    for (const ev of ["pointerdown", "pointermove", "pointerup", "pointercancel", "wheel", "contextmenu", "mousedown", "touchstart"]) {
      this.frame.addEventListener(ev, (e) => e.stopPropagation(), { passive: true });
    }
    this.refresh();
  }

  private defaultAxis(): string { return this.mode === "3d" ? "z" : "y"; }
  private axisValid(): boolean { return this.custom || (this.mode === "3d" ? AXES.includes(this.axis) : this.mode === "2d" ? this.axis === "x" || this.axis === "y" : false); }
  private axisIdx(): number { return Math.max(0, AXES.indexOf(this.axis)); }

  /** Sends the current state to the engine (debounced 50 ms), or clears when off. */
  private queue(now = false): void {
    clearTimeout(this.sendTimer);
    const go = () => {
      if (!this.enabled) { this.engine.send({ t: "clearSlice" }); return; }
      const txt = this.text.trim();
      if (!txt) return;
      if (this.custom) { this.engine.send({ t: "setSlice", fixed: txt }); return; }
      const n = Number(txt);
      this.engine.send({ t: "setSlice", fixed: { [this.axis]: Number.isFinite(n) && txt !== "" ? n : txt } });
    };
    if (now) go(); else this.sendTimer = window.setTimeout(go, 50);
  }

  private refresh(): void {
    const m = this.mode;
    const unavailable = m === "1d";
    this.toggle.disabled = unavailable;
    this.toggle.setAttribute("aria-pressed", String(this.enabled && !unavailable));
    this.toggle.title = unavailable ? "Slices need a 2D or 3D graph" : "Show a cross-section of the graph";
    $("slice-reason").hidden = !unavailable;
    this.panel.hidden = !this.enabled || unavailable;
    const axes = m === "3d" ? ["x", "y", "z"] : ["x", "y"];
    this.axisBtns.querySelectorAll<HTMLButtonElement>("button[data-axis]").forEach((b) => {
      const a = b.dataset.axis!;
      b.hidden = !axes.includes(a);
      b.setAttribute("aria-pressed", String(!this.custom && a === this.axis));
      b.setAttribute("aria-label", m === "3d" ? `Slice the ${a} plane: ${a} equals a value` : `Slice along the line ${a} equals a value`);
    });
    this.axisBtns.hidden = this.custom;
    const i = this.axisIdx();
    const lo = this.win.min[i], hi = this.win.max[i];
    this.range.min = String(lo); this.range.max = String(hi);
    this.range.step = String(Math.max((hi - lo) / 400, 0.001));
    const n = Number(this.text);
    if (Number.isFinite(n) && this.text.trim() !== "") this.range.value = String(n);
    this.range.disabled = this.custom;
    this.range.setAttribute("aria-label", `${this.custom ? "Slice" : this.axis} value, from ${fmt(lo)} to ${fmt(hi)}`);
    if (document.activeElement !== this.input) this.input.value = this.text;
    this.input.setAttribute("aria-label", this.custom ? "Slice constants, like y=1,z=0.5" : `Value of ${this.axis}: a number or a slider expression like a or 2a+1`);
    $("slice-eq").textContent = this.custom ? "" : `${this.axis} =`;
  }

  /** Call just before sending `setMode`: the slice event that precedes the `view` event is mode-forced. */
  beforeMode(next: string): void {
    if (next === this.mode) return;
    this.switching = true;
    setTimeout(() => { this.switching = false; }, 100);
  }

  onView(mode: string, min: number[], max: number[]): void {
    const changed = mode !== this.mode;
    if (changed) this.switching = false;
    this.mode = mode; this.win = { min, max };
    if (changed) {
      const old = this.prevMode;
      if ((old === "2d" || old === "3d") && !this.custom && this.want) this.mem[old] = { axis: this.axis, text: this.text };
      this.prevMode = mode;
      const m = this.mem[mode];
      if (m && !this.custom && (mode === "2d" || mode === "3d")) { this.axis = m.axis; this.text = m.text; }
      else if (!this.axisValid() && !this.custom) this.axis = this.defaultAxis();
    }
    // The engine drops a slice it cannot keep across a mode switch (and a 2D slice does not carry to 3D):
    // bring it back, recomputed for the new mode, once the mode can slice again.
    if (changed && this.want && mode !== "1d") {
      this.enabled = true; this.refresh();
      // Clear first: the engine keeps the old slice's dimension until it is cleared (Rust-side stale `dim`).
      this.resuming = true;
      try { this.engine.send({ t: "clearSlice" }); this.queue(true); } finally { this.resuming = false; }
      this.staleErr = true; setTimeout(() => { this.staleErr = false; }, 0);
      return;
    }
    // The range follows the window only while it is not being dragged.
    if (document.activeElement !== this.range || changed) this.refresh();
  }

  /** Re-syncs from a saved document (text of the constants, which the event only has resolved). */
  syncFromDoc(doc?: DocJson): boolean {
    const s = (doc as unknown as { slice?: { fixed?: Record<string, number | string | { Num?: number; Expr?: string }> } } | undefined)?.slice;
    if (!s?.fixed) return false;
    const fx = s.fixed;
    const keys = Object.keys(fx);
    const val = (v: unknown): string => typeof v === "object" && v ? String((v as { Num?: number; Expr?: string }).Num ?? (v as { Expr?: string }).Expr ?? "") : String(v);
    if (keys.length === 1) {
      this.custom = false; this.axis = keys[0]; this.text = val(fx[keys[0]]);
    } else if (keys.length > 1) {
      this.custom = true; this.text = keys.map((k) => `${k}=${val(fx[k])}`).join(",");
    } else return false;
    this.enabled = true; this.want = true;
    this.refresh();
    return true;
  }

  onSlice(e: SliceEvent): void {
    const prev = this.last;
    this.last = e;
    const pendingEdit = this.sendTimer !== 0 && false;
    void pendingEdit;
    if (e.active) {
      const keys = Object.keys(e.fixed);
      const external = !this.enabled || (keys.length === 1 && (keys[0] !== this.axis || this.custom)) || keys.length > 1 && !this.custom;
      if (external && this.switching) {
        // Mode-forced re-derivation (e.g. z=1 becoming y=0 in 2D): keep the remembered intent.
      } else if (external) {
        this.enabled = true; this.want = true;
        const { doc } = this.engine.exportDoc();
        if (!this.syncFromDoc(doc) && keys.length === 1) { this.axis = keys[0]; this.custom = false; this.text = fmt(e.fixed[keys[0]]); }
        // The engine's document holds a different slice than this event reports: the event is a stale
        // re-sample from before the mode switch (Rust emits it late), so do not draw it.
        if (keys.length === 1 && !this.custom && this.axis !== keys[0]) { this.refresh(); return; }
      } else if (keys.length === 1) {
        // A plain number that disagrees with the engine was changed elsewhere (e.g. a loaded doc).
        const n = Number(this.text);
        if (this.text.trim() !== "" && Number.isFinite(n) && Math.abs(n - e.fixed[keys[0]]) > 1e-9) this.text = fmt(e.fixed[keys[0]]);
      }
      this.status.textContent = this.summary(e);
      this.status.classList.remove("err");
    } else if (e.error && this.staleErr) {
      return; // belongs to the mode we just left; the resumed slice already replaced it
    } else if (e.error) {
      // The engine could not use the slice: turn the control off and say why.
      this.enabled = false;
      this.status.textContent = e.error;
      this.status.classList.add("err");
      this.panel.hidden = this.mode === "1d";
      this.statusShown(e.error);
    } else {
      this.enabled = this.resuming; if (!this.resuming) this.want = false;
      this.status.textContent = "";
      this.status.classList.remove("err");
    }
    this.refresh();
    if (!e.active && e.error && this.mode !== "1d") this.panel.hidden = false;
    this.drawFrame(e);
    this.announceSlice(prev, e);
  }

  private statusShown(msg: string): void { this.scheduleAnn(`Slice not shown: ${msg}`); }

  private summary(e: SliceEvent): string {
    const where = Object.entries(e.fixed).map(([k, v]) => `${k} = ${fmt(v)}`).join(", ");
    if (e.dim === 1) return `${where}: ${e.points} crossing point${e.points === 1 ? "" : "s"} marked.`;
    return `${where}: ${e.curves} curve${e.curves === 1 ? "" : "s"} in the inset.`;
  }

  private drawFrame(e: SliceEvent): void {
    if (!e.active || !e.rect) { this.frame.hidden = true; this.onRect(); return; }
    const d = this.dpr();
    const [x, y, w, h] = e.rect;
    Object.assign(this.frame.style, { left: `${x / d}px`, top: `${y / d}px`, width: `${w / d}px`, height: `${h / d}px` });
    this.frame.hidden = false;
    const where = Object.entries(e.fixed).map(([k, v]) => `${k} = ${fmt(v)}`).join(", ");
    this.title.textContent = where;
    this.frame.setAttribute("aria-label", `Slice inset, ${where}`);
    this.onRect();
  }

  /** Inset rectangle in CSS pixels, for clipping its labels. */
  rectCss(): [number, number, number, number] | null {
    if (!this.last?.active || !this.last.rect) return null;
    const d = this.dpr(), r = this.last.rect;
    return [r[0] / d, r[1] / d, r[2] / d, r[3] / d];
  }

  private announceSlice(prev: SliceEvent | null, e: SliceEvent): void {
    if (e.error && !e.active) return; // already announced
    const was = !!prev?.active;
    if (!e.active) { if (was) this.scheduleAnn("Slice off"); return; }
    const where = Object.entries(e.fixed).map(([k, v]) => `${k} equals ${fmt(v)}`).join(", ");
    this.scheduleAnn(was ? `Slice ${where}` : `Slice on, ${where}. ${this.summary(e)}`);
  }

  private scheduleAnn(msg: string): void {
    clearTimeout(this.annTimer);
    this.annTimer = window.setTimeout(() => {
      if (msg === this.lastAnn && msg.startsWith("Slice ") && !msg.startsWith("Slice off") && !msg.startsWith("Slice not")) return;
      this.lastAnn = msg;
      announce(msg, { immediate: true });
    }, 600);
  }

  handle(e: EngineEvent): void {
    if ((e as { t: string }).t === "slice") this.onSlice(e as unknown as SliceEvent);
  }
}
