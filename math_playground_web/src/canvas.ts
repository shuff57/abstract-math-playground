// Canvas sizing and input: pointer, wheel, pinch, keyboard. All coordinates sent are physical pixels.
import type { Engine } from "./engine";

export class CanvasController {
  private pointers = new Map<number, { x: number; y: number; type: string }>();
  private pinchDist = 0;
  private pinching = false;
  private btn = 0; // button that started the current one-pointer drag (move events report -1)
  private ro: ResizeObserver;
  onResize: () => void = () => {};

  constructor(private engine: Engine, private canvas: HTMLCanvasElement, private stage: HTMLElement) {
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(stage);
    window.addEventListener("resize", () => this.resize());
    this.resize();
    canvas.addEventListener("pointerdown", (e) => this.down(e));
    canvas.addEventListener("pointermove", (e) => this.move(e));
    canvas.addEventListener("pointerup", (e) => this.up(e, "up"));
    canvas.addEventListener("pointercancel", (e) => this.up(e, "cancel"));
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener("wheel", (e) => this.wheel(e), { passive: false });
    canvas.addEventListener("keydown", (e) => this.key(e));
  }

  get dpr(): number { return this.canvas.width / Math.max(this.canvas.clientWidth, 1); }

  resize(): void {
    const r = this.stage.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width * dpr));
    const h = Math.max(1, Math.round(r.height * dpr));
    if (w === this.canvas.width && h === this.canvas.height) return;
    this.canvas.width = w; this.canvas.height = h;
    this.engine.resize(w, h);
    // Render straight away so the canvas never shows a cleared (blank) frame between resize and next tick.
    this.engine.frame(performance.now());
    this.onResize();
  }

  private pos(e: { clientX: number; clientY: number }): [number, number] {
    const r = this.canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * (this.canvas.width / r.width), (e.clientY - r.top) * (this.canvas.height / r.height)];
  }
  private ptr(phase: string, x: number, y: number, button = 0, shift = false): void {
    this.engine.send({ t: "pointer", phase, x, y, button, shift });
  }
  private mid(): [number, number] {
    const p = [...this.pointers.values()];
    return this.pos({ clientX: (p[0].x + p[1].x) / 2, clientY: (p[0].y + p[1].y) / 2 });
  }
  private dist(): number {
    const p = [...this.pointers.values()];
    return Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y);
  }

  private down(e: PointerEvent): void {
    this.canvas.focus({ preventScroll: true });
    this.canvas.setPointerCapture(e.pointerId);
    const was = this.pointers.size;
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, type: e.pointerType });
    if (this.pointers.size === 2 && e.pointerType === "touch") {
      // Second finger: cancel the one-finger drag and start a pan+zoom gesture at the midpoint.
      if (was === 1) this.ptr("cancel", 0, 0);
      this.pinching = true; this.pinchDist = this.dist();
      const [mx, my] = this.mid();
      this.ptr("down", mx, my, 1, false);
      return;
    }
    if (this.pointers.size === 1) {
      const [x, y] = this.pos(e);
      this.btn = Math.max(0, e.button);
      this.ptr("down", x, y, this.btn, e.shiftKey);
    }
  }

  private move(e: PointerEvent): void {
    const p = this.pointers.get(e.pointerId);
    if (!p) return;
    p.x = e.clientX; p.y = e.clientY;
    if (this.pinching && this.pointers.size >= 2) {
      const [mx, my] = this.mid();
      const d = this.dist();
      if (this.pinchDist > 0 && d > 0) {
        const factor = d / this.pinchDist;
        this.engine.send({ t: "wheel", x: mx, y: my, dy: -Math.log(factor) / 0.0015 });
      }
      this.pinchDist = d;
      this.ptr("move", mx, my, 1, false);
      return;
    }
    if (this.pointers.size === 1) {
      const [x, y] = this.pos(e);
      this.ptr("move", x, y, this.btn, e.shiftKey);
    }
  }

  private up(e: PointerEvent, phase: "up" | "cancel"): void {
    if (!this.pointers.has(e.pointerId)) return;
    const wasPinch = this.pinching;
    if (wasPinch && this.pointers.size >= 2) {
      const [mx, my] = this.mid();
      this.ptr("up", mx, my, 1, false);
      this.pointers.delete(e.pointerId);
      this.pinching = false;
      // Remaining finger continues as a normal drag.
      const rest = [...this.pointers.values()][0];
      if (rest) { const [x, y] = this.pos({ clientX: rest.x, clientY: rest.y }); this.btn = 0; this.ptr("down", x, y, 0, false); }
      return;
    }
    this.pointers.delete(e.pointerId);
    const [x, y] = this.pos(e);
    this.ptr(phase, x, y, this.btn, e.shiftKey);
  }

  private wheel(e: WheelEvent): void {
    e.preventDefault();
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1;
    const [x, y] = this.pos(e);
    this.engine.send({ t: "wheel", x, y, dy: e.deltaY * unit * (e.ctrlKey ? 8 : 1) });
  }

  /** Arrow keys pan, +/- zoom (so the graph is usable without a pointer). */
  private key(e: KeyboardEvent): void {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const cx = this.canvas.width / 2, cy = this.canvas.height / 2;
    const step = 48 * this.dpr;
    const pan = (dx: number, dy: number) => {
      this.ptr("down", cx, cy, 1); this.ptr("move", cx + dx, cy + dy, 1); this.ptr("up", cx + dx, cy + dy, 1);
    };
    switch (e.key) {
      case "ArrowLeft": pan(step, 0); break;
      case "ArrowRight": pan(-step, 0); break;
      case "ArrowUp": pan(0, step); break;
      case "ArrowDown": pan(0, -step); break;
      case "+": case "=": this.engine.send({ t: "wheel", x: cx, y: cy, dy: -250 }); break;
      case "-": case "_": this.engine.send({ t: "wheel", x: cx, y: cy, dy: 250 }); break;
      default: return;
    }
    e.preventDefault();
  }
}
