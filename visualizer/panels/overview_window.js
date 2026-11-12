// Overview strip with a movable viewing window (st.components.v2).
// Drag inside the frame to move it, drag either side to resize it, press outside to
// centre it there. On release the new [start, end] goes to Python as the "window" trigger.

const MARGIN = { l: 48, r: 16, t: 28, b: 30 };
const EDGE_PX = 8;  // how close to a side counts as grabbing it

export default function (component) {
  const { parentElement, data, setTriggerValue } = component;
  let s = parentElement.__overview;
  if (!s) {
    s = parentElement.__overview = setUp(parentElement);
  }
  s.data = data;
  s.send = setTriggerValue;
  if (!s.drag) s.win = data.window.slice();
  draw(s);
  return () => {
    s.observer.disconnect();
    s.canvas.remove();
    delete parentElement.__overview;
  };
}

function setUp(root) {
  const canvas = document.createElement("canvas");
  canvas.style.display = "block";
  canvas.style.width = "100%";
  canvas.style.touchAction = "none";
  root.appendChild(canvas);
  const s = { canvas, drag: null };

  canvas.addEventListener("pointerdown", (e) => {
    const x = localX(s, e);
    const mode = hitTest(s, x);
    if (!mode) return;
    if (mode === "outside") {  // jump: centre the window on the pointer, then move it
      const w = s.win[1] - s.win[0];
      s.win = shiftInside(toTime(s, x) - w / 2, toTime(s, x) + w / 2, s.data.total);
    }
    s.drag = { mode: mode === "outside" ? "move" : mode, x0: x, win0: s.win.slice() };
    canvas.setPointerCapture(e.pointerId);
    draw(s);
  });
  canvas.addEventListener("pointermove", (e) => {
    const x = localX(s, e);
    if (!s.drag) {
      canvas.style.cursor = cursorFor(hitTest(s, x));
      return;
    }
    const dt = toTime(s, x) - toTime(s, s.drag.x0);
    const [a0, b0] = s.drag.win0;
    const { total, min_width: minW, max_width: maxW } = s.data;
    if (s.drag.mode === "move") s.win = shiftInside(a0 + dt, b0 + dt, total);
    else if (s.drag.mode === "start") s.win = [clamp(a0 + dt, Math.max(0, b0 - maxW), b0 - minW), b0];
    else s.win = [a0, clamp(b0 + dt, a0 + minW, Math.min(total, a0 + maxW))];
    draw(s);
  });
  const finish = () => {
    if (!s.drag) return;
    s.drag = null;
    draw(s);
    if (s.win[0] !== s.data.window[0] || s.win[1] !== s.data.window[1]) {
      s.send("window", s.win.map((v) => Math.round(v * 1e4) / 1e4));
    }
  };
  canvas.addEventListener("pointerup", finish);
  canvas.addEventListener("pointercancel", finish);
  canvas.addEventListener("pointerleave", () => { if (!s.drag) canvas.style.cursor = ""; });

  s.observer = new ResizeObserver(() => draw(s));
  s.observer.observe(canvas);
  return s;
}

// ---- geometry --------------------------------------------------------------

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function shiftInside(a, b, total) {
  const w = b - a;
  if (a < 0) { a = 0; b = w; }
  if (b > total) { b = total; a = total - w; }
  return [Math.max(0, a), Math.min(total, b)];
}

function localX(s, e) { return e.clientX - s.canvas.getBoundingClientRect().left; }

function plotWidth(s) { return Math.max(1, s.canvas.clientWidth - MARGIN.l - MARGIN.r); }

function toX(s, t) { return MARGIN.l + (t / s.data.total) * plotWidth(s); }

function toTime(s, x) { return ((x - MARGIN.l) / plotWidth(s)) * s.data.total; }

function hitTest(s, x) {
  if (x < MARGIN.l - EDGE_PX || x > MARGIN.l + plotWidth(s) + EDGE_PX) return null;
  const xa = toX(s, s.win[0]), xb = toX(s, s.win[1]);
  const da = Math.abs(x - xa), db = Math.abs(x - xb);
  if (Math.min(da, db) <= EDGE_PX) return da <= db ? "start" : "end";
  return x > xa && x < xb ? "move" : "outside";
}

function cursorFor(mode) {
  return { start: "ew-resize", end: "ew-resize", move: "grab", outside: "pointer" }[mode] || "";
}

// ---- drawing ---------------------------------------------------------------

function draw(s) {
  const { canvas, data } = s;
  const width = canvas.clientWidth;
  const height = data.height;
  const dpr = window.devicePixelRatio || 1;
  if (!width) return;
  canvas.style.height = height + "px";
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const ink = getComputedStyle(canvas).color || "#000";
  const top = MARGIN.t, bottom = height - MARGIN.b;
  const left = MARGIN.l, right = MARGIN.l + plotWidth(s);
  const [lo, hi] = yRange(data);
  const toY = (v) => bottom - ((v - lo) / (hi - lo)) * (bottom - top);

  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, right - left, bottom - top);
  ctx.clip();
  for (const ser of data.series) drawSeries(ctx, s, ser, toY, data.approximate);
  ctx.fillStyle = "#e07020";
  for (const t of data.events) {
    const x = toX(s, t), y = toY(hi) + 4;
    ctx.beginPath();
    ctx.moveTo(x - 4, y);
    ctx.lineTo(x + 4, y);
    ctx.lineTo(x, y + 6);
    ctx.fill();
  }
  ctx.restore();

  // Shade outside the window, then frame it
  const xa = toX(s, s.win[0]), xb = toX(s, s.win[1]);
  ctx.fillStyle = "rgba(128, 128, 128, 0.35)";
  ctx.fillRect(left, top, xa - left, bottom - top);
  ctx.fillRect(xb, top, right - xb, bottom - top);
  ctx.strokeStyle = ink;
  ctx.lineWidth = s.drag ? 3 : 4;
  ctx.strokeRect(xa, top, xb - xa, bottom - top);
  for (const x of [xa, xb]) {  // grips on each side
    ctx.fillStyle = ink;
    roundRect(ctx, x - 4, (top + bottom) / 2 - 12, 8, 24, 3);
  }

  // Side labels (live times while dragging)
  ctx.fillStyle = ink;
  ctx.font = "11px sans-serif";
  ctx.textBaseline = "bottom";
  const label = (i, name) => (s.drag ? s.win[i].toFixed(2) + " s" : name);
  const narrow = xb - xa < 100;  // labels sit outside the frame when it is narrow
  const startText = label(0, "start"), endText = label(1, "end");
  ctx.textAlign = "left";
  const startW = ctx.measureText(startText).width, endW = ctx.measureText(endText).width;
  ctx.fillText(startText, clamp(narrow ? xa - startW - 2 : xa, 0, width - startW), top - 4);
  ctx.fillText(endText, clamp(narrow ? xb + 2 : xb - endW, 0, width - endW), top - 4);

  // Time axis
  ctx.globalAlpha = 0.7;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.font = "12px sans-serif";
  for (const t of niceTicks(data.total, (right - left) / 70)) {
    ctx.fillText(String(+t.toFixed(3)), toX(s, t), bottom + 8);
  }
  ctx.globalAlpha = 1;
}

function drawSeries(ctx, s, ser, toY, approximate) {
  const t = s.data.t;
  if (ser.lo && ser.hi && ser.lo.length) {
    ctx.beginPath();
    for (let i = 0; i < t.length; i++) ctx.lineTo(toX(s, t[i]), toY(ser.hi[i]));
    for (let i = t.length - 1; i >= 0; i--) ctx.lineTo(toX(s, t[i]), toY(ser.lo[i]));
    ctx.closePath();
    ctx.globalAlpha = approximate ? 0.35 : 0.2;
    ctx.fillStyle = ser.color;
    ctx.fill();
    ctx.globalAlpha = 1;
  }
  if (ser.y && ser.y.length) {
    ctx.beginPath();
    for (let i = 0; i < t.length; i++) ctx.lineTo(toX(s, t[i]), toY(ser.y[i]));
    ctx.strokeStyle = ser.color;
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

function yRange(data) {
  let lo = Infinity, hi = -Infinity;
  for (const ser of data.series) {
    for (const arr of [ser.y, ser.lo, ser.hi]) {
      if (!arr) continue;
      for (const v of arr) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
  }
  if (!isFinite(lo)) return [0, 1];
  const pad = (hi - lo || 1) * 0.08;
  return [lo - pad, hi + pad * 2];  // extra room on top for the event markers
}

function niceTicks(total, maxTicks) {
  const raw = total / Math.max(1, maxTicks);
  const mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((v) => v >= raw) || 10 * mag;
  const ticks = [];
  for (let t = 0; t <= total + 1e-9; t += step) ticks.push(t);
  return ticks;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.fill();
}
