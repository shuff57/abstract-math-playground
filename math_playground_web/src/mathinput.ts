// Adapter around MathLive. Nothing else in the app imports mathlive.
// Starts as a plain <input> so first paint is fast, then upgrades to a <math-field> when the
// library has loaded (unless the row is in text mode or the load failed).

export interface MathInputOptions {
  latex: string;
  textMode: boolean;
  label: string;
  onEnter?: () => void;
}

type MathfieldLike = HTMLElement & {
  value: string;
  getValue(fmt?: string): string;
  setValue(v: string, o?: { silenceNotifications?: boolean }): void;
  menuItems: unknown[];
  mathVirtualKeyboardPolicy: string;
};

let libPromise: Promise<boolean> | null = null;
let libReady = false;
function loadMathLive(): Promise<boolean> {
  libPromise ??= Promise.all([import("mathlive"), import("mathlive/fonts.css")])
    .then(([m]) => {
      const M = m.MathfieldElement as unknown as { soundsDirectory: string | null; fontsDirectory: string | null; plonkSound: unknown; keypressSound: unknown };
      M.fontsDirectory = null; // fonts come from the bundled stylesheet
      M.soundsDirectory = null;
      M.plonkSound = null;
      M.keypressSound = null;
      libReady = true;
      return true;
    })
    .catch(() => false);
  return libPromise;
}

/** Renders `latex` as static (non-editable) math into `host`; falls back to plain `text` if MathLive
 * is unavailable. `host` carries role="img" and the plain text as its accessible name. */
export function renderStaticMath(host: HTMLElement, latex: string, text: string): void {
  host.setAttribute("role", "img");
  host.setAttribute("aria-label", text || latex);
  const plain = () => { host.replaceChildren(document.createTextNode(text || latex)); };
  plain();
  if (!latex) return;
  // A read-only <math-field> is MathLive's own static renderer and brings its styles in its shadow root.
  void loadMathLive().then((ok) => {
    if (!ok) return;
    const mf = document.createElement("math-field") as MathfieldLike;
    mf.className = "static-math";
    mf.setAttribute("read-only", "");
    mf.setAttribute("aria-hidden", "true");
    mf.setAttribute("tabindex", "-1");
    mf.setAttribute("math-virtual-keyboard-policy", "manual");
    host.replaceChildren(mf);
    mf.value = latex;
    try { mf.menuItems = []; } catch { /* default menu is harmless */ }
  });
}


/** MathLive abbreviates one-token arguments (`frac12`, `frac1{x}`, `sqrt2`); the parser wants braces, so add them. */
export function bracedArgs(src: string): string {
  if (!/\\(frac|sqrt)\s*[0-9A-Za-z\\]/.test(src)) return src;
  const arg = (i: number): [string, number] | null => {
    while (src[i] === " ") i++;
    if (i >= src.length) return null;
    if (src[i] === "{") { const e = matchBrace(i); return ["{" + bracedArgs(src.slice(i + 1, e)) + "}", e + 1]; }
    if (src[i] === "\\") { const m = /^\\(?!left|right)[A-Za-z]+/.exec(src.slice(i)); return m ? ["{" + m[0] + "}", i + m[0].length] : null; }
    if (/[0-9A-Za-z]/.test(src[i])) return ["{" + src[i] + "}", i + 1];
    return null;
  };
  const matchBrace = (i: number): number => {
    let d = 0;
    for (let j = i; j < src.length; j++) { if (src[j] === "{") d++; else if (src[j] === "}" && --d === 0) return j; }
    return src.length - 1;
  };
  let out = "", i = 0;
  const re = /\\(frac|sqrt)(?![A-Za-z])/g;
  for (let m: RegExpExecArray | null; (m = re.exec(src)); ) {
    if (m.index < i) continue;
    out += src.slice(i, m.index) + m[0];
    let j = m.index + m[0].length;
    if (m[1] === "sqrt" && src[j] === "[") { const k = src.indexOf("]", j); if (k < 0) { i = j; continue; } out += src.slice(j, k + 1); j = k + 1; }
    const need = m[1] === "frac" ? 2 : 1;
    for (let n = 0; n < need; n++) { const a = arg(j); if (!a) break; out += a[0]; j = a[1]; }
    i = j; re.lastIndex = j;
  }
  return out + src.slice(i);
}

export class MathInput {
  readonly el: HTMLElement;
  private plain: HTMLInputElement | null = null;
  private mf: MathfieldLike | null = null;
  private cb: Array<(latex: string) => void> = [];
  private latex: string;
  private text: boolean;
  private described = "";
  private destroyed = false;
  mathAvailable = true;

  constructor(private opts: MathInputOptions) {
    this.el = document.createElement("div");
    this.el.className = "field";
    this.latex = opts.latex;
    this.text = opts.textMode;
    this.mountPlain();
    if (!this.text) { if (libReady) this.swapToMath(); else void this.upgrade(); }
  }

  getLatex(): string {
    if (!this.mf) return this.plain?.value ?? this.latex;
    // MathLive turns a typed `d` (as in d/dx) into \differentialD, which the parser does not know: send a plain `d`.
    return bracedArgs(this.mf.getValue("latex").replace(/\\differentialD\s*/g, "d"));
  }
  setLatex(v: string): void {
    this.latex = v;
    if (this.mf) this.mf.setValue(v, { silenceNotifications: true });
    else if (this.plain) this.plain.value = v;
  }
  onChange(cb: (latex: string) => void): void { this.cb.push(cb); }
  focus(): void {
    const t = this.mf ?? this.plain;
    t?.focus();
    // A just-mounted <math-field> may drop its first focus() until MathLive has rendered it: retry.
    if (this.mf) {
      const mf = this.mf;
      const retry = () => { if (!this.destroyed && mf.isConnected && !mf.matches(":focus-within") && !mf.shadowRoot?.activeElement) mf.focus(); };
      requestAnimationFrame(retry); setTimeout(retry, 120);
    }
  }
  isTextMode(): boolean { return this.text; }
  describedBy(id: string): void {
    this.described = id;
    const t = this.mf ?? this.plain;
    if (!t) return;
    if (id) t.setAttribute("aria-describedby", id); else t.removeAttribute("aria-describedby");
  }
  destroy(): void { this.destroyed = true; this.cb = []; }

  async setTextMode(text: boolean): Promise<void> {
    if (text === this.text) return;
    this.latex = this.getLatex();
    this.text = text;
    this.el.replaceChildren();
    this.mf = null; this.plain = null;
    if (text) this.mountPlain();
    else { this.mountPlain(); if (libReady) this.swapToMath(); else await this.upgrade(); }
    this.focus();
    this.emit();
  }

  private emit(): void { const v = this.getLatex(); for (const c of this.cb) c(v); }

  private mountPlain(): void {
    const i = document.createElement("input");
    i.type = "text"; i.className = "plain"; i.value = this.latex;
    i.setAttribute("aria-label", this.opts.label);
    i.autocomplete = "off"; i.spellcheck = false; i.autocapitalize = "off";
    i.placeholder = "y = x^2";
    i.addEventListener("input", () => this.emit());
    i.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); this.opts.onEnter?.(); } });
    this.plain = i;
    this.el.replaceChildren(i);
    if (this.described) i.setAttribute("aria-describedby", this.described);
  }

  private async upgrade(): Promise<void> {
    const ok = await loadMathLive();
    if (!ok) { this.mathAvailable = false; return; }
    if (this.destroyed || this.text || !this.plain) return;
    this.swapToMath();
  }

  private swapToMath(): void {
    if (!this.plain) return;
    const hadFocus = document.activeElement === this.plain;
    const caretAtEnd = this.plain.selectionStart === this.plain.value.length;
    const mf = document.createElement("math-field") as MathfieldLike;
    mf.setAttribute("aria-label", this.opts.label);
    mf.setAttribute("smart-fence", "");
    mf.setAttribute("math-virtual-keyboard-policy", "auto");
    if (matchMedia("(pointer: coarse)").matches) mf.classList.add("touch");
    const initial = this.getLatex();
    mf.addEventListener("input", () => this.emit());
    mf.addEventListener("keydown", (e) => { if ((e as KeyboardEvent).key === "Enter") { e.preventDefault(); this.opts.onEnter?.(); } }, true);
    this.mf = mf; this.plain = null;
    this.el.replaceChildren(mf);
    mf.value = initial;
    try { mf.menuItems = []; } catch { /* menu stays default */ }
    if (this.described) mf.setAttribute("aria-describedby", this.described);
    if (hadFocus) {
      // A freshly mounted math-field finishes initialising asynchronously, so focus again once it has.
      const focusIt = () => { mf.focus(); if (caretAtEnd) mf.executeCommand?.("moveToMathfieldEnd"); };
      focusIt(); requestAnimationFrame(focusIt);
    }
  }
}

declare global {
  interface HTMLElement { executeCommand?(cmd: string): boolean }
}
