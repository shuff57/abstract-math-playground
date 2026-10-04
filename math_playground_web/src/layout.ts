// Responsive shell: desktop resizer, tablet drawer, phone bottom sheet with three snap points.
const q = (id: string) => document.getElementById(id) as HTMLElement;
const phoneMQ = matchMedia("(max-width: 599px)");
const tabletMQ = matchMedia("(max-width: 1023px)");

type Snap = "collapsed" | "half" | "full";
const SNAPS: Snap[] = ["collapsed", "half", "full"];
const HANDLE_H = 56, TOP_H = 48;

export class Layout {
  private snap: Snap = "half";
  private sheetFull = 0;
  onStageChange: () => void = () => {};
  private sidebar = q("sidebar");
  private handle = q("sheet-handle");

  constructor() {
    this.initResizer();
    this.initDrawer();
    this.initSheet();
    const update = () => this.apply();
    phoneMQ.addEventListener("change", update);
    tabletMQ.addEventListener("change", update);
    window.addEventListener("resize", update);
    this.apply();
  }

  get isPhone(): boolean { return phoneMQ.matches; }
  currentSnap(): Snap { return this.snap; }
  setSnap(s: Snap): void { this.snap = s; this.apply(); }

  private visible(s: Snap): number {
    if (s === "collapsed") return HANDLE_H;
    if (s === "half") return Math.round(window.innerHeight * 0.45);
    return this.sheetFull;
  }

  private apply(): void {
    const root = document.documentElement.style;
    const sb = this.sidebar;
    if (phoneMQ.matches) {
      this.sheetFull = window.innerHeight - TOP_H - 8;
      root.setProperty("--sheet-h", `${this.sheetFull}px`);
      root.setProperty("--sheet-y", `${this.sheetFull - this.visible(this.snap)}px`);
      // The graph keeps the area above the sheet; at "full" it stays at the half-size so it doesn't thrash.
      const inset = this.snap === "collapsed" ? HANDLE_H : this.visible("half");
      root.setProperty("--stage-inset", `${inset}px`);
      this.handle.setAttribute("aria-expanded", String(this.snap !== "collapsed"));
      this.handle.setAttribute("aria-valuetext", this.snap);
      sb.toggleAttribute("inert", false);
      for (const el of sb.querySelectorAll<HTMLElement>(".side-head, .rows, .hint")) el.toggleAttribute("inert", this.snap === "collapsed");
      document.body.classList.remove("drawer-open");
    } else {
      root.removeProperty("--sheet-h"); root.removeProperty("--sheet-y"); root.removeProperty("--stage-inset");
      for (const el of sb.querySelectorAll<HTMLElement>(".side-head, .rows, .hint")) el.removeAttribute("inert");
      const closedDrawer = tabletMQ.matches && !document.body.classList.contains("drawer-open");
      sb.toggleAttribute("inert", closedDrawer);
    }
  }

  // ---- phone sheet ----------------------------------------------------------------------------
  private initSheet(): void {
    const h = this.handle;
    let y0 = 0, v0 = 0, moved = false, lastY = 0, lastT = 0, vel = 0, active = false;
    h.addEventListener("pointerdown", (e) => {
      active = true; moved = false; y0 = lastY = e.clientY; lastT = e.timeStamp; vel = 0;
      v0 = this.visible(this.snap);
      h.setPointerCapture(e.pointerId);
    });
    h.addEventListener("pointermove", (e) => {
      if (!active) return;
      const dy = e.clientY - y0;
      if (!moved && Math.abs(dy) < 5) return;
      moved = true;
      this.sidebar.classList.add("dragging");
      const vis = Math.max(HANDLE_H, Math.min(this.sheetFull, v0 - dy));
      document.documentElement.style.setProperty("--sheet-y", `${this.sheetFull - vis}px`);
      const dt = e.timeStamp - lastT;
      if (dt > 0) vel = (e.clientY - lastY) / dt; // px/ms, positive = moving down
      lastY = e.clientY; lastT = e.timeStamp;
    });
    const end = (e: PointerEvent) => {
      if (!active) return;
      active = false;
      this.sidebar.classList.remove("dragging");
      if (!moved) { this.cycle(); return; }
      const cur = Math.max(HANDLE_H, Math.min(this.sheetFull, v0 - (e.clientY - y0)));
      const projected = cur - vel * 160;
      let best: Snap = "half", bd = Infinity;
      for (const s of SNAPS) { const d = Math.abs(this.visible(s) - projected); if (d < bd) { bd = d; best = s; } }
      this.setSnap(best);
    };
    h.addEventListener("pointerup", end);
    h.addEventListener("pointercancel", () => { active = false; this.sidebar.classList.remove("dragging"); this.apply(); });
    h.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); this.cycle(); }
      else if (e.key === "ArrowUp") { e.preventDefault(); this.setSnap(SNAPS[Math.min(2, SNAPS.indexOf(this.snap) + 1)]); }
      else if (e.key === "ArrowDown") { e.preventDefault(); this.setSnap(SNAPS[Math.max(0, SNAPS.indexOf(this.snap) - 1)]); }
    });
    // The stage only resizes when the snap settles (the ResizeObserver on the stage handles it).
  }

  private cycle(): void {
    this.setSnap(this.snap === "collapsed" ? "half" : this.snap === "half" ? "full" : "collapsed");
  }

  // ---- tablet drawer --------------------------------------------------------------------------
  private initDrawer(): void {
    const btn = q("drawer-btn"), scrim = q("scrim");
    const set = (open: boolean) => {
      document.body.classList.toggle("drawer-open", open);
      btn.setAttribute("aria-expanded", String(open));
      btn.setAttribute("aria-label", open ? "Hide expressions" : "Show expressions");
      scrim.hidden = !(open && tabletMQ.matches && !phoneMQ.matches);
      this.apply();
      if (open) q("add-btn").focus(); else btn.focus();
    };
    btn.addEventListener("click", () => set(!document.body.classList.contains("drawer-open")));
    scrim.addEventListener("click", () => set(false));
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && document.body.classList.contains("drawer-open")) set(false);
    });
  }

  // ---- desktop resizer ------------------------------------------------------------------------
  private initResizer(): void {
    const r = q("sidebar-resizer");
    const setW = (w: number) => {
      w = Math.max(240, Math.min(560, Math.round(w)));
      document.documentElement.style.setProperty("--side-w", `${w}px`);
      r.setAttribute("aria-valuenow", String(w));
    };
    let drag = false;
    r.addEventListener("pointerdown", (e) => { drag = true; r.classList.add("drag"); r.setPointerCapture(e.pointerId); });
    r.addEventListener("pointermove", (e) => { if (drag) setW(e.clientX); });
    const end = () => { drag = false; r.classList.remove("drag"); };
    r.addEventListener("pointerup", end); r.addEventListener("pointercancel", end);
    r.addEventListener("keydown", (e) => {
      const w = this.sidebar.getBoundingClientRect().width;
      if (e.key === "ArrowLeft") { e.preventDefault(); setW(w - 16); }
      if (e.key === "ArrowRight") { e.preventDefault(); setW(w + 16); }
    });
    r.setAttribute("aria-valuemin", "240"); r.setAttribute("aria-valuemax", "560"); r.setAttribute("aria-valuenow", "320");
  }
}
