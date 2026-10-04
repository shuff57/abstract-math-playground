// Single aria-live region. Messages are throttled so rapid changes (panning) don't flood a screen reader.
const live = () => document.getElementById("live") as HTMLElement;
let timer = 0;
let pending = "";
let last = 0;

export function announce(msg: string, opts: { immediate?: boolean } = {}): void {
  pending = msg;
  const wait = opts.immediate ? 0 : Math.max(900 - (performance.now() - last), 400);
  clearTimeout(timer);
  timer = window.setTimeout(() => {
    last = performance.now();
    const el = live();
    el.textContent = "";
    // Re-set on the next tick so identical consecutive messages are still announced.
    requestAnimationFrame(() => { el.textContent = pending; });
  }, wait);
}

let toastTimer = 0;
export function toast(msg: string, ms = 2600, el?: HTMLElement): void {
  const t = document.getElementById("toast") as HTMLElement;
  t.textContent = msg;
  if (el) t.append(el);
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => { t.hidden = true; }, ms);
}
