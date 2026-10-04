// Editable data-table grid for one table item. The engine is the source of truth: every edit is a
// command, and each `table` event re-syncs the grid (without disturbing the cell being typed in).
import type { Engine, TableState } from "./engine";
import { announce } from "./announce";

const STYLES: { id: string; label: string }[] = [
  { id: "points", label: "Points" }, { id: "line", label: "Line" }, { id: "hidden", label: "Hidden" },
];
const MAX_COLUMNS = 12, MAX_ROWS = 500;

export class TableGrid {
  readonly el: HTMLElement;
  private state: TableState | null = null;
  private grid: HTMLTableElement;
  private styleBtns = new Map<string, HTMLButtonElement>();
  private addRowBtn: HTMLButtonElement;
  private addColBtn: HTMLButtonElement;
  private pendingFocus: { r: number; c: number } | null = null;
  private timers = new Map<string, number>();

  constructor(private engine: Engine, private id: string, private label: () => string) {
    this.el = document.createElement("div");
    this.el.className = "tbl";
    this.grid = document.createElement("table");
    this.grid.className = "tbl-grid";
    const wrap = document.createElement("div");
    wrap.className = "tbl-scroll";
    wrap.append(this.grid);

    const bar = document.createElement("div");
    bar.className = "tbl-bar";
    this.addRowBtn = this.btn("+ Row", "Add row", () => {
      this.pendingFocus = { r: (this.state?.rows ?? 0), c: 0 };
      this.engine.send({ t: "addRow", id: this.id });
      announce(`Row added to ${this.label()}`, { immediate: true });
    });
    this.addColBtn = this.btn("+ Column", "Add column", () => {
      this.pendingFocus = { r: -1, c: this.state?.columns.length ?? 0 };
      this.engine.send({ t: "addColumn", id: this.id });
      announce(`Column added to ${this.label()}`, { immediate: true });
    });
    const seg = document.createElement("div");
    seg.className = "seg tbl-style";
    seg.setAttribute("role", "group");
    seg.setAttribute("aria-label", "Table drawing style");
    for (const s of STYLES) {
      const b = this.btn(s.label, `Draw table as ${s.label.toLowerCase()}`, () => {
        this.engine.send({ t: "setTableStyle", id: this.id, style: s.id });
        announce(`Table style ${s.label.toLowerCase()}`, { immediate: true });
      });
      b.classList.remove("tool"); b.removeAttribute("title");
      b.setAttribute("aria-pressed", "false");
      this.styleBtns.set(s.id, b);
      seg.append(b);
    }
    bar.append(this.addRowBtn, this.addColBtn, seg);
    this.el.append(wrap, bar);
  }

  private btn(text: string, label: string, fn: () => void): HTMLButtonElement {
    const b = document.createElement("button");
    b.type = "button"; b.className = "tool tbl-btn"; b.textContent = text; b.setAttribute("aria-label", label);
    b.addEventListener("click", fn);
    return b;
  }

  focus(): void { this.grid.querySelector<HTMLInputElement>("input")?.focus(); }
  destroy(): void { for (const t of this.timers.values()) clearTimeout(t); this.timers.clear(); }

  private cell(r: number, c: number): HTMLInputElement | null {
    return this.grid.querySelector<HTMLInputElement>(`input[data-r="${r}"][data-c="${c}"]`);
  }

  update(s: TableState): void {
    const prev = this.state;
    this.state = s;
    for (const [k, b] of this.styleBtns) b.setAttribute("aria-pressed", String(k === s.style));
    this.addRowBtn.disabled = s.rows >= MAX_ROWS;
    this.addColBtn.disabled = s.columns.length >= MAX_COLUMNS;
    const same = prev && prev.rows === s.rows && prev.columns.length === s.columns.length;
    if (!same || this.pendingFocus) this.rebuild(s);
    else this.sync(s);
  }

  /** Same shape: write engine values into inputs, except the one being edited. */
  private sync(s: TableState): void {
    this.relabel();
    const active = document.activeElement;
    s.columns.forEach((col, c) => {
      const h = this.cell(-1, c);
      if (h && h !== active && h.value !== col.name) h.value = col.name;
      col.cells.forEach((v, r) => {
        const i = this.cell(r, c);
        if (i && i !== active && i.value !== v) i.value = v;
      });
    });
  }

  /** Re-read the owning item row's label (it changes on reorder/add/remove) and refresh every aria-label. */
  relabel(): void {
    const s = this.state; if (!s) return;
    const name = this.label();
    this.grid.setAttribute("aria-label", `${name}: ${s.columns.length} columns, ${s.rows} rows`);
    for (const i of this.grid.querySelectorAll<HTMLInputElement>("input[data-r]")) {
      const r = Number(i.dataset.r), c = Number(i.dataset.c);
      i.setAttribute("aria-label", r < 0 ? `${name}, column ${c + 1} name` : `${name}, table row ${r + 1}, column ${s.columns[c]?.name ?? c + 1}`);
    }
  }

  private rebuild(s: TableState): void {
    const act = document.activeElement as HTMLElement | null;
    let focus = this.pendingFocus;
    if (!focus && act && this.grid.contains(act) && act.dataset.r !== undefined) {
      focus = { r: Number(act.dataset.r), c: Number(act.dataset.c) };
    }
    this.pendingFocus = null;
    const name = this.label();
    this.grid.setAttribute("aria-label", `${name}: ${s.columns.length} columns, ${s.rows} rows`);
    const head = document.createElement("thead");
    const hr = document.createElement("tr");
    const corner = document.createElement("th");
    corner.className = "tbl-rn"; corner.scope = "col";
    corner.innerHTML = '<span class="sr-only">Row</span>';
    hr.append(corner);
    s.columns.forEach((col, c) => {
      const th = document.createElement("th");
      th.scope = "col";
      const inp = this.input(-1, c, col.name, `${name}, column ${c + 1} name`);
      inp.classList.add("tbl-head");
      const rm = this.removeBtn(`Remove column ${col.name}`, () => {
        this.pendingFocus = { r: -1, c: Math.max(0, Math.min(c, s.columns.length - 2)) };
        this.engine.send({ t: "removeColumn", id: this.id, col: c });
        announce(`Column ${col.name} removed`, { immediate: true });
      });
      rm.disabled = s.columns.length <= 1;
      th.append(inp, rm);
      hr.append(th);
    });
    head.append(hr);
    const body = document.createElement("tbody");
    for (let r = 0; r < s.rows; r++) {
      const tr = document.createElement("tr");
      const rh = document.createElement("th");
      rh.scope = "row"; rh.className = "tbl-rn";
      const n = document.createElement("span"); n.textContent = String(r + 1); n.setAttribute("aria-hidden", "true");
      const sr = document.createElement("span"); sr.className = "sr-only"; sr.textContent = `Row ${r + 1}`;
      const rm = this.removeBtn(`Remove row ${r + 1}`, () => {
        this.pendingFocus = { r: Math.max(0, Math.min(r, s.rows - 2)), c: 0 };
        this.engine.send({ t: "removeRow", id: this.id, row: r });
        announce(`Row ${r + 1} removed`, { immediate: true });
      });
      rh.append(n, sr, rm);
      tr.append(rh);
      s.columns.forEach((col, c) => {
        const td = document.createElement("td");
        td.append(this.input(r, c, col.cells[r] ?? "", `${name}, table row ${r + 1}, column ${col.name}`));
        tr.append(td);
      });
      body.append(tr);
    }
    this.grid.replaceChildren(head, body);
    if (focus) {
      const r = Math.min(focus.r, s.rows - 1), c = Math.min(focus.c, s.columns.length - 1);
      (this.cell(r, c) ?? this.cell(-1, c))?.focus();
    }
  }

  private removeBtn(label: string, fn: () => void): HTMLButtonElement {
    const b = document.createElement("button");
    b.type = "button"; b.className = "tool tbl-rm"; b.textContent = "×";
    b.setAttribute("aria-label", label); b.title = label;
    b.addEventListener("click", fn);
    return b;
  }

  private input(r: number, c: number, value: string, label: string): HTMLInputElement {
    const i = document.createElement("input");
    i.type = "text"; i.className = "tbl-cell"; i.value = value;
    i.dataset.r = String(r); i.dataset.c = String(c);
    i.setAttribute("aria-label", label);
    i.autocomplete = "off"; i.spellcheck = false; i.autocapitalize = "off";
    i.addEventListener("input", () => {
      const key = `${i.dataset.r},${i.dataset.c}`;
      clearTimeout(this.timers.get(key));
      this.timers.set(key, window.setTimeout(() => this.commit(i), 60));
    });
    i.addEventListener("blur", () => this.commit(i));
    i.addEventListener("keydown", (e) => this.onKey(e, i));
    return i;
  }

  private commit(i: HTMLInputElement): void {
    const r = Number(i.dataset.r), c = Number(i.dataset.c);
    const key = `${r},${c}`;
    clearTimeout(this.timers.get(key)); this.timers.delete(key);
    const cur = this.state;
    if (!cur) return;
    if (r < 0) {
      if (cur.columns[c] && cur.columns[c].name !== i.value) this.engine.send({ t: "renameColumn", id: this.id, col: c, name: i.value });
    } else if ((cur.columns[c]?.cells[r] ?? "") !== i.value) {
      this.engine.send({ t: "setCell", id: this.id, row: r, col: c, value: i.value });
    }
  }

  /** Enter moves down a row (Shift+Enter up); Enter on the last row adds one. Tab is native. */
  private onKey(e: KeyboardEvent, i: HTMLInputElement): void {
    if (e.key !== "Enter" || e.isComposing) return;
    e.preventDefault();
    this.commit(i);
    const s = this.state; if (!s) return;
    const r = Number(i.dataset.r), c = Number(i.dataset.c);
    const to = e.shiftKey ? r - 1 : r + 1;
    if (to >= s.rows) {
      if (s.rows >= MAX_ROWS) return;
      this.pendingFocus = { r: s.rows, c };
      this.engine.send({ t: "addRow", id: this.id });
      announce("Row added", { immediate: true });
    } else if (to >= -1) this.cell(to, c)?.focus();
  }
}
