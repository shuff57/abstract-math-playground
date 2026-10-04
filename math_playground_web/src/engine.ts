// Thin typed wrapper over the wasm Calculator: JSON commands in, typed events out.
import init, { create_calculator, type Calculator } from "../../math_playground/pkg/math_playground_lib.js";
import wasmUrl from "../../math_playground/pkg/math_playground_lib_bg.wasm?url";

export type Kind =
  | "equation" | "expression" | "complex" | "points" | "vectorField"
  | "action" | "slider" | "slice" | "folder" | "note" | "table";

export interface DocItem { id: string; kind: Kind; latex: string; hidden?: boolean; color?: string | null; folder?: string | null; table?: { columns: { name: string; cells: string[] }[]; style?: string } }
export interface SliderCfg { min: number; max: number; step?: number; value: number }
export interface DocJson { v: number; view: { mode: string; window: { min: number[]; max: number[] }; angle?: string }; items: DocItem[]; sliders?: Record<string, SliderCfg> }
export interface TableState { id: string; columns: { name: string; cells: string[] }[]; rows: number; style: string }
export interface ScreenLabel { text: string; axis: number; x: number; y: number; visible: boolean; inset?: boolean }

export interface ItemInfo {
  id: string; kind: "derivative" | "value" | "regression";
  latex?: string; text?: string; value?: number;
  params?: { name: string; value: number; stdError?: number }[];
  r2?: number; rmse?: number; n?: number;
}

export type EngineEvent =
  | { t: "info"; items: ItemInfo[] }
  | { t: "colors"; items: { id: string; color: string }[] }
  | { t: "diagnostics"; items: { id: string; message: string }[] }
  | { t: "view"; mode: string; min: number[]; max: number[] }
  | { t: "labels"; labels: unknown[] }
  | { t: "doc"; json: string }
  | { t: "hash"; hash: string }
  | ({ t: "table" } & TableState)
  | { t: "itemEdited"; id: string; latex: string }
  | { t: "sliderValue"; name: string; value: number }
  | { t: "theme"; dark: boolean }
  | { t: "tickerState"; running: boolean; action: string | null }
  | { t: "slice"; active: boolean; dim: number; fixed: Record<string, number>; free: string[]; rect: [number, number, number, number] | null; curves: number; points: number; error: string | null }
  | { t: "error"; message: string };

export type Command = { t: string; [k: string]: unknown };
type Listener = (e: EngineEvent) => void;

export class Engine {
  private listeners: Listener[] = [];
  private constructor(private calc: Calculator) {
  }

  static async create(canvas: HTMLCanvasElement): Promise<Engine> {
    await init(wasmUrl);
    const calc = await create_calculator(canvas);
    return new Engine(calc);
  }

  get backend(): string { return this.calc.backend_name(); }
  on(l: Listener): void { this.listeners.push(l); }

  /** Sends a command, fans the returned events out to listeners, and returns them. */
  send(cmd: Command): EngineEvent[] {
    let events: EngineEvent[] = [];
    try {
      events = JSON.parse(this.calc.dispatch(JSON.stringify(cmd))) as EngineEvent[];
    } catch (err) {
      events = [{ t: "error", message: String(err) }];
    }
    for (const e of events) for (const l of this.listeners) l(e);
    return events;
  }

  /** Advances one frame, then delivers the events raised inside it (rebuild results, ticker steps). */
  frame(now: number): boolean {
    const drew = this.calc.frame(now);
    let events: EngineEvent[] = [];
    try {
      events = JSON.parse(this.calc.drain()) as EngineEvent[];
    } catch (err) {
      events = [{ t: "error", message: String(err) }];
    }
    for (const e of events) for (const l of this.listeners) l(e);
    return drew;
  }
  resize(w: number, h: number): void { this.calc.resize(w, h); }
  screenLabels(): ScreenLabel[] {
    try { return JSON.parse(this.calc.screen_labels_json()) as ScreenLabel[]; } catch { return []; }
  }

  /** Current document and share hash (export is synchronous). */
  exportDoc(): { doc?: DocJson; hash?: string } {
    const out: { doc?: DocJson; hash?: string } = {};
    for (const e of this.send({ t: "export" })) {
      if (e.t === "doc") out.doc = JSON.parse(e.json) as DocJson;
      if (e.t === "hash") out.hash = e.hash;
    }
    return out;
  }
}
