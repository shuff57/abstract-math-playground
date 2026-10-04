// DOM tick-label layer. Pool of elements reused every update; only touched when a frame was rendered.
import type { Engine } from "./engine";

export class LabelOverlay {
  private pool: HTMLElement[] = [];
  private shown = 0;
  private cache: { text: string; x: number; y: number; vis: boolean; axis: number }[] = [];
  private lastWasThree = false;
  /** Inset rectangle in CSS pixels, or null. Inset labels are clipped to it. */
  sliceRect: () => [number, number, number, number] | null = () => null;

  constructor(private engine: Engine, private layer: HTMLElement, private canvas: HTMLCanvasElement) {}

  update(): void {
    const labels = this.engine.screenLabels();
    const dpr = this.canvas.width / Math.max(this.canvas.clientWidth, 1);
    const n = labels.length;
    const ir = this.sliceRect();
    while (this.pool.length < n) {
      const el = document.createElement("span");
      el.className = "lbl-tick";
      el.style.display = "none";
      this.layer.append(el);
      this.pool.push(el);
      this.cache.push({ text: "", x: NaN, y: NaN, vis: false, axis: -1 });
    }
    for (let i = 0; i < n; i++) {
      const l = labels[i], el = this.pool[i], c = this.cache[i];
      const x = Math.round((l.x / dpr) * 10) / 10, y = Math.round((l.y / dpr) * 10) / 10;
      if (c.text !== l.text) { el.textContent = l.text; c.text = l.text; }
      if (c.axis !== l.axis) { el.dataset.axis = String(l.axis); c.axis = l.axis; }
      const w = this.layer.clientWidth, h = this.layer.clientHeight;
      // Hide labels whose text would be cut off by the canvas edge.
      let vis = l.visible && (l.axis === 0 ? x > 16 && x < w - 16 : l.axis === 1 ? y > 9 && y < h - 9 : true);
      if (l.inset) {
        // Inset labels live inside the inset frame: drop any that fall outside it (or without one).
        const m = 2;
        vis = l.visible && !!ir && x >= ir[0] - m && x <= ir[0] + ir[2] + m && y >= ir[1] - m && y <= ir[1] + ir[3] + m;
        el.classList.toggle("inset", true);
      } else {
        el.classList.toggle("inset", false);
        // The inset covers the main view: a main-scene label that pokes into it would overprint the inset's own labels.
        if (vis && ir && x >= ir[0] - 16 && x <= ir[0] + ir[2] + 16 && y >= ir[1] - 22 && y <= ir[1] + ir[3] + 12) vis = false;
      }
      if (c.vis !== vis) { el.style.display = vis ? "" : "none"; c.vis = vis; }
      if (vis && (c.x !== x || c.y !== y)) {
        // x ticks sit just below their line, y ticks just left of theirs, z ticks centred.
        const t = l.axis === 3 ? "0, 0" : l.axis === 0 ? "-50%, 5px" : l.axis === 1 ? "calc(-100% - 6px), -50%" : "-50%, -50%";
        el.style.transform = `translate(${x}px, ${y}px) translate(${t})`;
        c.x = x; c.y = y;
      }
    }
    for (let i = n; i < this.shown; i++) { this.pool[i].style.display = "none"; this.cache[i].vis = false; }
    this.shown = n;
    void this.lastWasThree;
  }
}
