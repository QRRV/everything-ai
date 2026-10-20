# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""The course's building blocks to play with an idea (decision 1.10): gradient descent that you start and steer, a
training run that you replay, a tokenizer that splits what you type, the table of equations that a small model learns
from, and a picture that follows a slider at once. Python computes what the lesson teaches; the widget shows it and
moves it (docs/ontwerp/wat-maakt-uitleg-mooi.md). Drawn only with the course's tokens, so light and dark both work, and all
code ships with the course. A lesson gives each its own task and its labels in its own language: a building block
replaces a still picture that shows the same, it is not added on top (lessjabloon, block 9)."""
from __future__ import annotations

import math
import re
from decimal import Decimal

import anywidget
import traitlets

from everything_ai.widgets import _BASE_CSS

_PLAY_CSS = _BASE_CSS + r"""
.ea-play { max-width: 46rem; border: 1px solid var(--ea-line); border-radius: 10px; padding: 0.9rem 1rem 0.8rem; background: var(--ea-page); }
.ea-play .ea-task { margin: 0 0 0.7rem; font-size: 0.97rem; }
.ea-play .ea-task i { font-family: "Source Serif var", "Source Serif 4", Georgia, serif; font-size: 1.08em; }
.ea-play .ea-row { display: flex; flex-wrap: wrap; gap: 0.6rem 1rem; align-items: center; }
.ea-play .ea-charts { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 0.8rem; align-items: start; margin-bottom: 0.4rem; }
.ea-play .ea-charts.ea-one { grid-template-columns: minmax(0, 1fr); }
.ea-play .ea-charts.ea-one svg { max-width: 34rem; }
@container (max-width: 40rem) { .ea-play .ea-charts { grid-template-columns: 1fr; } .ea-play .ea-charts svg + svg { max-width: 22rem; } }
.ea-play .ea-ctitle { font-size: 0.85rem; font-weight: 600; color: var(--ea-ink-2); margin: 0.2rem 0 0.1rem; }
.ea-play svg { width: 100%; height: auto; display: block; overflow: visible; }
.ea-play svg text { font-family: "Inter var", Inter, system-ui, sans-serif; font-size: 12px; fill: var(--ea-ink-3); }
.ea-play svg .ea-title { font-size: 13px; font-weight: 600; fill: var(--ea-ink-2); }
.ea-play svg .ea-grid { stroke: var(--ea-line); stroke-width: 1; }
.ea-play svg .ea-axis { stroke: var(--ea-axis); stroke-width: 1; }
.ea-play button { margin-top: 0; display: inline-flex; gap: 0.35rem; align-items: center; min-height: 2rem; }
.ea-play button.ea-quiet { border-color: var(--ea-line); color: var(--ea-ink-2); font-weight: 550; }
.ea-play .ea-readout { font-variant-numeric: tabular-nums; font-size: 0.95rem; color: var(--ea-ink-2); min-height: 1.5rem; }
.ea-play .ea-readout b { font-weight: 650; }
.ea-play .ea-c-param { color: var(--ea-param-text); }
.ea-play .ea-c-loss { color: var(--ea-loss-text); }
.ea-play .ea-c-hyper { color: var(--ea-hyper-text); }
.ea-play .ea-verdict { margin-top: 0.55rem; background: var(--ea-inset); border-radius: 8px; padding: 0.5rem 0.8rem; }
.ea-play input[type=range] { accent-color: var(--ea-ink); }
.ea-play .ea-lr { display: flex; align-items: center; gap: 0.6rem; flex: 1 1 16rem; min-width: 14rem; }
.ea-play .ea-lr .ea-name { font-family: "Source Serif var", "Source Serif 4", Georgia, serif; font-style: italic; font-size: 1.15rem; font-weight: 600; color: var(--ea-hyper-text); }
.ea-play .ea-lr input { -webkit-appearance: none; appearance: none; flex: 1; height: 1.5rem; margin: 0; background: transparent; cursor: pointer; }
.ea-play .ea-lr input::-webkit-slider-runnable-track { height: 6px; border-radius: 3px; background: linear-gradient(to right, var(--ea-hyper) 0 var(--p), var(--ea-line) var(--p) 100%); }
.ea-play .ea-lr input::-moz-range-track { height: 6px; border-radius: 3px; background: var(--ea-line); }
.ea-play .ea-lr input::-moz-range-progress { height: 6px; border-radius: 3px; background: var(--ea-hyper); }
.ea-play .ea-lr input::-webkit-slider-thumb { -webkit-appearance: none; width: 18px; height: 18px; margin-top: -6px; border-radius: 50%; background: var(--ea-page); border: 2px solid var(--ea-hyper); }
.ea-play .ea-lr input::-moz-range-thumb { width: 14px; height: 14px; border-radius: 50%; background: var(--ea-page); border: 2px solid var(--ea-hyper); }
.ea-play .ea-lr output { font-variant-numeric: tabular-nums; font-weight: 600; color: var(--ea-hyper-text); min-width: 5.5rem; }
.ea-play input.ea-scrub { -webkit-appearance: none; appearance: none; height: 1.5rem; margin: 0; background: transparent; cursor: pointer; }
.ea-play input.ea-scrub::-webkit-slider-runnable-track { height: 4px; border-radius: 2px; background: linear-gradient(to right, var(--ea-ink-2) 0 var(--p), var(--ea-line) var(--p) 100%); }
.ea-play input.ea-scrub::-moz-range-track { height: 4px; border-radius: 2px; background: var(--ea-line); }
.ea-play input.ea-scrub::-moz-range-progress { height: 4px; border-radius: 2px; background: var(--ea-ink-2); }
.ea-play input.ea-scrub::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px; margin-top: -6px; border-radius: 50%; background: var(--ea-page); border: 2px solid var(--ea-ink); }
.ea-play input.ea-scrub::-moz-range-thumb { width: 12px; height: 12px; border-radius: 50%; background: var(--ea-page); border: 2px solid var(--ea-ink); }
.ea-play .ea-drag { cursor: grab; outline: none; }
.ea-play .ea-drag:active { cursor: grabbing; }
@media (prefers-reduced-motion: reduce) { .ea-play * { transition: none !important; } }
"""

_SVG = r"""
const NS = "http://www.w3.org/2000/svg";
const mk = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const el_ = (tag, cls, parent, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.innerHTML = text; if (parent) parent.appendChild(e); return e; };
const still = () => window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
"""


def _need(widget, keys) -> None:
    """Every text a widget shows comes from the lesson, in its own language: none may be missing."""
    missing = [k for k in keys if not widget.labels.get(k)]
    if missing:
        raise ValueError(f"{type(widget).__name__} needs the labels {', '.join(missing)}")


class Descent(anywidget.AnyWidget):
    """Gradient descent on the loss for every value of one parameter: drag the start, set the learning rate, and play
    the steps. Beside it, the loss after each step (a loss curve, from the lesson that introduces it) or how far the
    parameter moved in each step. Python computes the curve, the path and the steps; the widget draws them and moves the
    point (only while you drag does it read the loss off the curve itself)."""

    _esm = _SVG + r"""
    function render({ model, el }) {
      const L = model.get("labels"), dec = model.get("decimal"), beside = model.get("beside");
      const fmt = (v, d) => { if (!isFinite(v) || Math.abs(v) >= 1e7) return "∞"; let s = Number(v).toFixed(d); if (dec === ",") { s = s.replace(".", ","); } return s.replace("-", "−"); };
      const big = v => (!isFinite(v) || v > 1e12) ? "> 10¹²" : String(Math.round(v)).replace("-", "−");   // as the course's charts and texts: 2790, not 2,790
      const root = el_("div", "ea-w ea-play", el); root.style.containerType = "inline-size";
      const task = el_("p", "ea-task", root); el_("span", "ea-kick", task, L.task_kick); task.insertAdjacentHTML("beforeend", L.task);
      const charts = el_("div", "ea-charts" + (beside ? "" : " ea-one"), root);
      const W = 420, H = 250, PL = 46, PR = 12, PT = 26, PB = 34;
      const svg = mk("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": L.alt }, charts);
      const W2 = 280, H2 = 250;   // 420 : 280 is the 3 : 2 of the grid, so both charts are drawn at the same scale and their axes line up
      const svg2 = mk("svg", { viewBox: `0 0 ${W2} ${H2}`, role: "img", "aria-label": L.alt2 }, charts); if (!beside) svg2.style.display = "none";
      const controls = el_("div", "ea-row", root);
      const play = el_("button", "", controls), stepb = el_("button", "ea-quiet", controls, L.step), reset = el_("button", "ea-quiet", controls, "↺ " + L.reset);
      const lrbox = el_("label", "ea-lr", controls); el_("span", "ea-name", lrbox, "η"); if (!model.get("show_lr")) lrbox.style.display = "none";
      const lr = el_("input", "", lrbox); lr.type = "range"; lr.min = model.get("lr_min"); lr.max = model.get("lr_max"); lr.step = model.get("lr_step"); lr.setAttribute("aria-label", L.lr_name);
      const lrout = el_("output", "", lrbox);
      const readout = el_("div", "ea-readout", root); readout.setAttribute("aria-live", "polite");
      let k = 0, playing = false, timer = null, anim = null, drag = null, focused = false;
      const curve = () => model.get("curve"), path = () => model.get("path");
      let x0, x1, y1;
      const sx = v => PL + (v - x0) / (x1 - x0) * (W - PL - PR);
      const sy = v => H - PB - Math.min(Math.max(v, 0), y1) / y1 * (H - PT - PB);
      const syc = v => H - PB - Math.min(Math.max(v, -y1), 2 * y1) / y1 * (H - PT - PB);   // the curve: what is above ymax runs off the top, the clip cuts it
      const ix = px => x0 + (px - PL) / (W - PL - PR) * (x1 - x0);
      const lossAt = w => { const c = curve(); for (let i = 1; i < c.length; i++) if (c[i][0] >= w) { const f = (w - c[i-1][0]) / (c[i][0] - c[i-1][0]); return c[i-1][1] + f * (c[i][1] - c[i-1][1]); } return c[c.length-1][1]; };
      const uid = "c" + Math.random().toString(36).slice(2);
      const clip = mk("clipPath", { id: uid }, mk("defs", {}, svg)); mk("rect", { x: PL, y: PT - 10, width: W - PL - PR, height: H - PT - PB + 10 }, clip);
      const gGrid = mk("g", {}, svg), plot = mk("g", { "clip-path": `url(#${uid})` }, svg), gCurve = mk("path", { fill: "none", stroke: "var(--ea-loss)", "stroke-width": 2.2 }, plot);
      const gTrail = mk("g", {}, plot), next = mk("line", { stroke: "var(--ea-hyper)", "stroke-width": 1.6, "stroke-dasharray": "4 4" }, plot);
      const ring = mk("circle", { r: 12.5, fill: "none", stroke: "var(--ea-ink)", "stroke-width": 2, "pointer-events": "none" }, svg);
      const pt = mk("circle", { r: 8, fill: "var(--ea-param)", stroke: "var(--ea-page)", "stroke-width": 2.5, "pointer-events": "none" }, svg);
      const hit = mk("circle", { r: 16, fill: "transparent", class: "ea-drag", tabindex: 0, role: "slider", "aria-label": L.w_name }, svg);   // at least 24 px to grab, also in a narrow window
      const offbg = mk("rect", { rx: 4, fill: "var(--ea-inset)" }, svg), off = mk("text", { "text-anchor": "end", class: "ea-title", style: "fill: var(--ea-param-text)" }, svg);
      mk("text", { x: 4, y: 14, class: "ea-title" }, svg).textContent = L.title;
      const g2 = mk("g", {}, svg2), line2 = mk("path", { fill: "none", stroke: "var(--ea-loss)", "stroke-width": 2.2 }, svg2), dot2 = mk("circle", { r: 4.5, fill: "var(--ea-loss)", stroke: "var(--ea-page)", "stroke-width": 2 }, svg2);
      const bars = mk("g", {}, svg2);
      mk("text", { x: 4, y: 14, class: "ea-title" }, svg2).textContent = L.title2;
      if (beside !== "loss") line2.style.display = dot2.style.display = "none";
      function axes() {
        const c = curve(); x0 = c[0][0]; x1 = c[c.length - 1][0]; y1 = model.get("ymax");
        gGrid.innerHTML = "";
        for (const t of model.get("yticks")) { mk("line", { x1: PL, x2: W - PR, y1: sy(t), y2: sy(t), class: "ea-grid" }, gGrid); mk("text", { x: PL - 6, y: sy(t) + 4, "text-anchor": "end" }, gGrid).textContent = big(t); }
        for (const t of model.get("xticks")) mk("text", { x: sx(t), y: H - PB + 18, "text-anchor": "middle" }, gGrid).textContent = fmt(t, Number.isInteger(t) ? 0 : 1);
        mk("line", { x1: PL, x2: W - PR, y1: H - PB, y2: H - PB, class: "ea-axis" }, gGrid);
        mk("text", { x: W - PR, y: H - 4, "text-anchor": "end" }, gGrid).textContent = L.x_name;
        gCurve.setAttribute("d", "M" + c.map(d => `${sx(d[0]).toFixed(1)},${syc(d[1]).toFixed(1)}`).join("L"));
        hit.setAttribute("aria-valuemin", x0); hit.setAttribute("aria-valuemax", x1);
      }
      const PL2 = 48, PR2 = 10, PT2 = 26, PB2 = 34;
      function mini(upto) {
        const p = path(), log = model.get("log_steps"), N = model.get("steps_n");
        const X = i => PL2 + i / N * (W2 - PL2 - PR2);
        let Y, ticks, tick;
        if (log) {
          const lo = Math.floor(Math.log10(Math.max(1, Math.min(...p.map(d => d[1]))))), hi = Math.max(lo + 1, Math.ceil(Math.log10(Math.min(1e12, Math.max(10, ...p.map(d => d[1]))))));
          Y = v => H2 - PB2 - (Math.log10(Math.min(Math.max(v, 1), 1e12)) - lo) / Math.max(hi - lo, 1) * (H2 - PT2 - PB2);
          ticks = []; for (let e = lo; e <= hi; e += Math.max(1, Math.ceil((hi - lo) / 4))) ticks.push(10 ** e);
          tick = v => "10" + String(Math.round(Math.log10(v))).split("").map(c => "⁰¹²³⁴⁵⁶⁷⁸⁹"[Number(c)]).join("");
        } else {   // the same loss axis as the curve beside it: values above it stay at its top
          Y = v => H2 - PB2 - Math.min(Math.max(v, 0), y1) / y1 * (H2 - PT2 - PB2);
          ticks = model.get("yticks"); tick = big;
        }
        g2.innerHTML = "";
        for (const v of ticks) { mk("line", { x1: PL2, x2: W2 - PR2, y1: Y(v), y2: Y(v), class: "ea-grid" }, g2); mk("text", { x: PL2 - 6, y: Y(v) + 4, "text-anchor": "end" }, g2).textContent = tick(v); }
        mk("line", { x1: PL2, x2: W2 - PR2, y1: H2 - PB2, y2: H2 - PB2, class: "ea-axis" }, g2);
        for (const t of [0, Math.round(N / 2), N]) mk("text", { x: X(t), y: H2 - PB2 + 18, "text-anchor": "middle" }, g2).textContent = t;
        mk("text", { x: W2 - PR2, y: H2 - 4, "text-anchor": "end" }, g2).textContent = L.steps_name;
        const pts = p.slice(0, Math.floor(upto) + 1);
        line2.setAttribute("d", pts.length ? "M" + pts.map((d, i) => `${X(i).toFixed(1)},${Y(d[1]).toFixed(1)}`).join("L") : "");
        const last = pts[pts.length - 1]; dot2.setAttribute("cx", X(pts.length - 1)); dot2.setAttribute("cy", Y(last[1]));
      }
      const nice = v => { const e = 10 ** Math.floor(Math.log10(v)); for (const f of [1, 2, 2.5, 5, 10]) if (f * e >= v * 0.9999) return f * e; return 10 * e; };
      const places = v => { let d = 0; while (d < 6 && Math.abs(Math.round(v * 10 ** d) - v * 10 ** d) > 1e-6) d++; return d; };
      function sizes(upto) {   // how far w moved in each step: every step there is, so the shape shows before you touch anything; the steps taken in orange
        const m = model.get("moves").map(Math.abs), N = model.get("steps_n"), i = Math.floor(upto);
        const top = nice(Math.max(1e-9, ...m.filter(isFinite))), d = places(top / 2);
        const X = j => PL2 + (j - 0.5) / N * (W2 - PL2 - PR2), bw = 0.62 * (W2 - PL2 - PR2) / N;
        const Y = v => H2 - PB2 - Math.min(v, top) / top * (H2 - PT2 - PB2);
        g2.innerHTML = ""; bars.innerHTML = "";
        for (const v of [0, top / 2, top]) { mk("line", { x1: PL2, x2: W2 - PR2, y1: Y(v), y2: Y(v), class: "ea-grid" }, g2); mk("text", { x: PL2 - 6, y: Y(v) + 4, "text-anchor": "end" }, g2).textContent = fmt(v, v ? d : 0); }
        mk("line", { x1: PL2, x2: W2 - PR2, y1: H2 - PB2, y2: H2 - PB2, class: "ea-axis" }, g2);
        for (const t of [...new Set([1, Math.round(N / 2), N])]) mk("text", { x: X(t), y: H2 - PB2 + 18, "text-anchor": "middle" }, g2).textContent = t;
        mk("text", { x: W2 - PR2, y: H2 - 4, "text-anchor": "end" }, g2).textContent = L.steps_name;
        m.forEach((v, j) => { const s = j + 1, h = Math.max(1, H2 - PB2 - Y(v)), done = s <= i, nextOne = s === i + 1 && !playing;
          mk("rect", { x: X(s) - bw / 2, y: H2 - PB2 - h, width: bw, height: h, rx: 2, fill: done ? "var(--ea-param)" : "var(--ea-line)",
            stroke: nextOne ? "var(--ea-hyper)" : "none", "stroke-width": 1.6, "stroke-dasharray": "4 3" }, bars); });
        if (i > 0) { const v = m[i - 1]; mk("text", { x: X(i), y: Y(v) - 6, "text-anchor": "middle", class: "ea-title", style: "fill: var(--ea-param-text)" }, bars).textContent = v < 0.0005 ? "< " + fmt(0.001, 3) : fmt(v, 3); }
      }
      function place(w, loss) {
        const inside = w >= x0 && w <= x1 && loss <= y1;
        pt.style.display = hit.style.display = inside ? "" : "none"; ring.style.display = inside && focused ? "" : "none";
        const cx = sx(Math.min(Math.max(w, x0), x1)), cy = sy(loss);
        for (const c of [pt, hit, ring]) { c.setAttribute("cx", cx); c.setAttribute("cy", cy); }
        off.style.display = offbg.style.display = inside ? "none" : ""; off.setAttribute("x", W - PR - 8); off.setAttribute("y", PT + 16); off.textContent = inside ? "" : L.gone;
        if (!inside) { const b = off.getBBox(); offbg.setAttribute("x", b.x - 6); offbg.setAttribute("y", b.y - 3); offbg.setAttribute("width", b.width + 12); offbg.setAttribute("height", b.height + 6); }
      }
      const say = (i, w, loss) => `${L.step_name} <b>${i}</b> · <span class="ea-c-param"><i>${L.x_name}</i> = <b>${fmt(w, 3)}</b></span> · <span class="ea-c-loss">loss = <b>${big(loss)}</b></span>`;
      function draw() {
        const p = path(), i = Math.min(Math.floor(k), p.length - 1);
        gTrail.innerHTML = "";
        for (let j = 0; j < i; j++) { const a = p[j], b = p[j + 1]; if (a[1] > y1 || b[1] > y1) break;
          mk("line", { x1: sx(a[0]), y1: sy(a[1]), x2: sx(b[0]), y2: sy(b[1]), stroke: "var(--ea-ink-3)", "stroke-width": 1 }, gTrail);
          mk("circle", { cx: sx(a[0]), cy: sy(a[1]), r: 3.5, fill: "var(--ea-param)", opacity: 0.45 }, gTrail); }
        const cur = p[i]; place(cur[0], cur[1]);
        if (i < p.length - 1 && !playing && cur[1] <= y1) { const nx = p[i + 1]; next.style.display = ""; next.setAttribute("x1", sx(cur[0])); next.setAttribute("y1", sy(cur[1])); next.setAttribute("x2", sx(Math.min(Math.max(nx[0], x0), x1))); next.setAttribute("y2", sy(nx[1])); }
        else next.style.display = "none";
        if (beside === "loss") mini(i); else if (beside === "step") sizes(i);
        readout.innerHTML = say(i, cur[0], cur[1]);
        const w0 = model.get("w0"); hit.setAttribute("aria-valuenow", +w0.toFixed(4)); hit.setAttribute("aria-valuetext", fmt(w0, 3));
        play.textContent = playing ? "❚❚ " + L.pause : "▶ " + L.play;
        stepb.disabled = i >= p.length - 1;
      }
      function tween(from, to, done) {
        const p = path(); if (still()) { k = to; draw(); done && done(); return; }
        const a = p[from], b = p[Math.min(to, p.length - 1)], t0 = performance.now(), ms = 380;
        cancelAnimationFrame(anim);
        const f = now => { const u = Math.min(1, (now - t0) / ms), e = u * u * (3 - 2 * u);
          if (b[1] <= y1 * 4) { const w = a[0] + (b[0] - a[0]) * e; place(w, Math.min(lossAt(Math.min(Math.max(w, x0), x1)), a[1] + (b[1] - a[1]) * e)); }
          if (u < 1) anim = requestAnimationFrame(f); else { k = to; draw(); done && done(); } };
        anim = requestAnimationFrame(f);
      }
      const halt = () => { playing = false; clearTimeout(timer); cancelAnimationFrame(anim); };
      function advance() { const n = path().length - 1, i = Math.floor(k); if (i >= n) { playing = false; draw(); return; } tween(i, i + 1, () => { if (playing) timer = setTimeout(advance, 120); }); }
      play.onclick = () => { if (playing) { halt(); draw(); return; } if (Math.floor(k) >= path().length - 1) k = 0; playing = true; draw(); advance(); };
      stepb.onclick = () => { const i = Math.floor(k); halt(); if (i < path().length - 1) tween(i, i + 1); };
      reset.onclick = () => { halt(); model.set("w0", model.get("w0_start")); model.set("lr", model.get("lr_start")); model.save_changes(); k = 0; sync(); draw(); };
      function sync() { const v = model.get("lr"); lr.value = v; lrout.textContent = fmt(v, model.get("lr_digits"));
        lr.parentElement.style.setProperty("--p", ((v - lr.min) / (lr.max - lr.min) * 100) + "%"); }
      lr.addEventListener("input", () => { halt(); model.set("lr", Number(lr.value)); model.save_changes(); k = 0; sync(); });
      const toW = ev => { const r = svg.getBoundingClientRect(); return Math.min(Math.max(ix((ev.clientX - r.left) / r.width * W), x0), x1); };
      const ringed = on => { focused = on; ring.style.display = on && pt.style.display !== "none" ? "" : "none"; };
      hit.addEventListener("pointerdown", ev => { drag = true; halt(); ringed(false); hit.setPointerCapture(ev.pointerId); next.style.display = "none"; gTrail.innerHTML = ""; });
      hit.addEventListener("pointermove", ev => { if (!drag) return; const w = toW(ev); place(w, lossAt(w)); readout.innerHTML = say(0, w, lossAt(w)); });
      hit.addEventListener("pointerup", ev => { if (!drag) return; drag = false; model.set("w0", toW(ev)); model.save_changes(); k = 0; draw(); });
      hit.addEventListener("keydown", ev => { const d = { ArrowLeft: -0.05, ArrowRight: 0.05 }[ev.key]; if (!d) return; ev.preventDefault(); ringed(true); model.set("w0", Math.min(Math.max(model.get("w0") + d, x0), x1)); model.save_changes(); k = 0; draw(); });
      hit.addEventListener("focus", () => ringed(hit.matches(":focus-visible")));
      hit.addEventListener("blur", () => ringed(false));
      model.on("change:path", () => { k = 0; draw(); });
      model.on("change:lr", sync);
      axes(); sync(); draw();
      return () => { halt(); };
    }
    export default { render };
    """
    _css = _PLAY_CSS
    labels = traitlets.Dict({}).tag(sync=True)
    decimal = traitlets.Unicode(".").tag(sync=True)
    curve = traitlets.List([]).tag(sync=True)
    path = traitlets.List([]).tag(sync=True)
    ymax = traitlets.Float(1.0).tag(sync=True)
    xticks = traitlets.List([]).tag(sync=True)
    yticks = traitlets.List([]).tag(sync=True)
    w0 = traitlets.Float(0.0).tag(sync=True)
    w0_start = traitlets.Float(0.0).tag(sync=True)
    lr = traitlets.Float(0.1).tag(sync=True)
    lr_start = traitlets.Float(0.1).tag(sync=True)
    lr_min = traitlets.Float(0.0).tag(sync=True)
    lr_max = traitlets.Float(1.0).tag(sync=True)
    lr_step = traitlets.Float(0.01).tag(sync=True)
    lr_digits = traitlets.Int(2).tag(sync=True)
    steps_n = traitlets.Int(25).tag(sync=True)
    show_lr = traitlets.Bool(True).tag(sync=True)
    log_steps = traitlets.Bool(False).tag(sync=True)   # the loss per step on a log axis, for lessons that know one
    beside = traitlets.Unicode("loss").tag(sync=True)   # beside the curve: "loss" (the loss after each step), "step" (how far w moved in each step) or ""
    moves = traitlets.List([]).tag(sync=True)

    LABELS = ("task_kick", "task", "title", "title2", "x_name", "steps_name", "step", "reset", "play", "pause", "lr_name",
              "w_name", "step_name", "gone", "alt", "alt2")

    def __init__(self, loss, grad, *, w_range, ymax, steps=25, **kw):
        super().__init__(**kw)
        _need(self, self.LABELS)
        if self.beside not in ("loss", "step", ""):
            raise ValueError('Descent shows beside the curve "loss", "step" or nothing ("")')
        self._loss, self._grad, self._steps = loss, grad, steps
        self.steps_n = steps
        lo, hi = w_range
        self.curve = [[lo + (hi - lo) * i / 200, loss(lo + (hi - lo) * i / 200)] for i in range(201)]
        self.ymax = ymax
        self.w0_start, self.lr_start = self.w0, self.lr
        self._compute()
        self.observe(lambda _: self._compute(), names=["w0", "lr"])

    def _compute(self):
        w, out = self.w0, []
        for _ in range(self._steps + 1):
            value = self._loss(w)
            out.append([w, value if math.isfinite(value) else 1e300])
            if not math.isfinite(value) or abs(w) > 1e150:
                break
            w = w - self.lr * self._grad(w)
        self.moves = [b[0] - a[0] for a, b in zip(out, out[1:])]
        self.path = out


class Replay(anywidget.AnyWidget):
    """A training run that you play back: accuracy on the training and the validation data over the steps, on a log
    axis, with the moments the plan looks for (t and v) marked when the run gets there, and its verdict at the end.
    It opens at the end, so the whole run shows without a click; Play starts again from the first step, and dragging
    across the chart or the slider picks a step. Python gives the logs and the verdict; the widget plays them."""

    _esm = _SVG + r"""
    function render({ model, el }) {
      const L = model.get("labels"), dec = model.get("decimal");
      const n = model.get("steps").length, steps = model.get("steps"), tr = model.get("train"), va = model.get("val");
      const big = v => String(Math.round(v)).replace(/\B(?=(\d{3})+(?!\d))/g, dec === "," ? "." : ",");
      const pct = v => (Math.round(v * 1000) / 10).toFixed(1).replace(".", dec) + "%";
      const root = el_("div", "ea-w ea-play", el);
      const task = el_("p", "ea-task", root); el_("span", "ea-kick", task, L.task_kick); task.insertAdjacentHTML("beforeend", L.task);
      el_("div", "ea-ctitle", root, L.title);
      const W = 460, H = 250, PL = 40, PR = 70, PT = 30, PB = 36;   // drawn at about its own size, so text stays 11 px or more
      const svg = mk("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": L.alt }, root); svg.style.maxWidth = "34rem";
      const lx0 = Math.log10(steps[0]), lx1 = Math.log10(steps[n - 1]);
      const sx = s => PL + (Math.log10(s) - lx0) / (lx1 - lx0) * (W - PL - PR), sy = a => H - PB - a * (H - PT - PB);
      for (const a of [0, 0.5, 1]) { mk("line", { x1: PL, x2: W - PR, y1: sy(a), y2: sy(a), class: "ea-grid" }, svg); mk("text", { x: PL - 6, y: sy(a) + 4, "text-anchor": "end" }, svg).textContent = Math.round(a * 100) + "%"; }
      mk("line", { x1: PL, x2: W - PR, y1: sy(model.get("threshold")), y2: sy(model.get("threshold")), stroke: "var(--ea-ink-3)", "stroke-width": 1, "stroke-dasharray": "2 3" }, svg);
      mk("line", { x1: PL, x2: W - PR, y1: H - PB, y2: H - PB, class: "ea-axis" }, svg);
      for (let e = Math.ceil(lx0); e <= Math.floor(lx1); e++) mk("text", { x: sx(10 ** e), y: H - PB + 18, "text-anchor": "middle" }, svg).textContent = big(10 ** e);
      mk("text", { x: W - PR, y: H - 4, "text-anchor": "end" }, svg).textContent = L.x_name;
      const lt = mk("path", { fill: "none", stroke: "var(--ea-ink)", "stroke-width": 2.2, "stroke-linejoin": "round" }, svg);
      const lv = mk("path", { fill: "none", stroke: "var(--ea-ink)", "stroke-width": 2.2, "stroke-dasharray": "6 4" }, svg);
      const tt = mk("text", { fill: "var(--ea-ink)", "font-weight": 600 }, svg), tv = mk("text", { fill: "var(--ea-ink)", "font-weight": 600 }, svg);
      const marks = mk("g", {}, svg), head = mk("line", { stroke: "var(--ea-axis)", "stroke-width": 1 }, svg);
      const controls = el_("div", "ea-row", root);
      const play = el_("button", "", controls), back = el_("button", "ea-quiet", controls, "← " + L.back), fwd = el_("button", "ea-quiet", controls, L.fwd + " →");
      const scrub = el_("input", "ea-scrub", controls); scrub.type = "range"; scrub.min = 0; scrub.max = n - 1; scrub.step = 1; scrub.style.flex = "1 1 12rem"; scrub.setAttribute("aria-label", L.x_name);
      const readout = el_("div", "ea-readout", root); readout.setAttribute("aria-live", "polite");
      const verdict = el_("div", "ea-verdict", root); verdict.style.display = "none";
      let i = n - 1, playing = false, raf = null, last = 0;   // the whole run at first: the insight without a click
      const tI = model.get("t_index"), vI = model.get("v_index");
      function draw() {
        const d = (arr, upto) => "M" + arr.slice(0, upto + 1).map((a, j) => `${sx(steps[j]).toFixed(1)},${sy(a).toFixed(1)}`).join("L");
        lt.setAttribute("d", d(tr, i)); lv.setAttribute("d", d(va, i));
        const yt = sy(tr[i]), yv = sy(va[i]), apart = Math.abs(yt - yv) < 16;
        const up = apart ? Math.max(0, Math.max(yt, yv) + 16 - (H - PB - 3)) : 0;   // both labels above the axis, clear of its numbers
        tt.setAttribute("x", sx(steps[i]) + 8); tt.setAttribute("y", (apart ? Math.min(yt, yv) - 4 - up : yt + 4)); tt.textContent = L.train;
        tv.setAttribute("x", sx(steps[i]) + 8); tv.setAttribute("y", (apart ? Math.max(yt, yv) + 16 - up : yv + 4)); tv.textContent = L.val;
        head.setAttribute("x1", sx(steps[i])); head.setAttribute("x2", sx(steps[i])); head.setAttribute("y1", PT); head.setAttribute("y2", H - PB);
        marks.innerHTML = "";
        for (const [j, name] of [[tI, "t"], [vI, "v"]]) if (j >= 0 && i >= j) {
          const y = name === "t" ? tr[j] : va[j];
          mk("circle", { cx: sx(steps[j]), cy: sy(y), r: 5, fill: "var(--ea-ink)", stroke: "var(--ea-page)", "stroke-width": 2 }, marks);
          mk("line", { x1: sx(steps[j]), x2: sx(steps[j]), y1: PT - 6, y2: H - PB, stroke: "var(--ea-ink-3)", "stroke-width": 1, "stroke-dasharray": "1 3" }, marks);
          mk("text", { x: sx(steps[j]), y: PT - 12, "text-anchor": "middle", fill: "var(--ea-ink)", "font-weight": 650, "font-style": "italic", "font-size": 15, "font-family": "Source Serif 4, Georgia, serif" }, marks).textContent = name; }
        scrub.value = i; scrub.style.setProperty("--p", (i / (n - 1) * 100) + "%");
        readout.innerHTML = `${L.step_name} <b>${big(steps[i])}</b> · ${L.train} <b>${pct(tr[i])}</b> · ${L.val} <b>${pct(va[i])}</b>`;
        const end = i >= n - 1; verdict.style.display = end ? "" : "none"; verdict.innerHTML = end ? model.get("verdict") : "";
        play.textContent = playing ? "❚❚ " + L.pause : "▶ " + L.play;
      }
      function frame(now) { if (!playing) return; if (now - last > 45) { last = now; i = Math.min(i + 1, n - 1); draw(); if (i >= n - 1) { playing = false; draw(); return; } } raf = requestAnimationFrame(frame); }
      play.onclick = () => { if (playing) { playing = false; draw(); return; } if (i >= n - 1) i = 0; playing = true; draw();
        if (still()) { const tick = () => { if (!playing) return; i = Math.min(i + 5, n - 1); draw(); if (i < n - 1) setTimeout(tick, 400); else { playing = false; draw(); } }; tick(); } else raf = requestAnimationFrame(frame); };
      back.onclick = () => { playing = false; i = Math.max(0, i - 1); draw(); };
      fwd.onclick = () => { playing = false; i = Math.min(n - 1, i + 1); draw(); };
      scrub.addEventListener("input", () => { playing = false; i = Number(scrub.value); draw(); });
      // dragging across the chart picks the logged step nearest to the pointer, as the slider does
      svg.style.cursor = "ew-resize"; svg.style.touchAction = "pan-y";
      const pick = ev => { const r = svg.getBoundingClientRect(), lx = lx0 + ((ev.clientX - r.left) / r.width * W - PL) / (W - PL - PR) * (lx1 - lx0);
        let j = 0; for (let k = 1; k < n; k++) if (Math.abs(Math.log10(steps[k]) - lx) < Math.abs(Math.log10(steps[j]) - lx)) j = k;
        playing = false; i = j; draw(); };
      let dragging = false;
      svg.addEventListener("pointerdown", ev => { if (ev.button !== 0) return; dragging = true; svg.setPointerCapture(ev.pointerId); pick(ev); });
      svg.addEventListener("pointermove", ev => { if (dragging) pick(ev); });
      svg.addEventListener("pointerup", () => { dragging = false; }); svg.addEventListener("pointercancel", () => { dragging = false; });
      draw();
      return () => { playing = false; cancelAnimationFrame(raf); };
    }
    export default { render };
    """
    _css = _PLAY_CSS
    labels = traitlets.Dict({}).tag(sync=True)
    decimal = traitlets.Unicode(".").tag(sync=True)
    steps = traitlets.List([]).tag(sync=True)
    train = traitlets.List([]).tag(sync=True)
    val = traitlets.List([]).tag(sync=True)
    threshold = traitlets.Float(0.99).tag(sync=True)
    t_index = traitlets.Int(-1).tag(sync=True)
    v_index = traitlets.Int(-1).tag(sync=True)
    verdict = traitlets.Unicode("").tag(sync=True)
    LABELS = ("task_kick", "task", "title", "x_name", "train", "val", "play", "pause", "back", "fwd", "step_name", "alt")

    def __init__(self, **kw):
        super().__init__(**kw)
        _need(self, self.LABELS)
        if not (len(self.steps) == len(self.train) == len(self.val) >= 2):
            raise ValueError("Replay needs the same number of steps, training and validation values, at least two")
        if not (self.steps[0] > 0 and all(b > a for a, b in zip(self.steps, self.steps[1:]))):
            raise ValueError("Replay needs steps above 0 that go up: they lie on a log axis")
        if not all(-1 <= i < len(self.steps) for i in (self.t_index, self.v_index)):
            raise ValueError("Replay needs t_index and v_index within the steps, or -1 when the run never gets there")


class Tokens(anywidget.AnyWidget):
    """A tokenizer you type into: the text in a box, its tokens as tiles below it, with each token's id on hover or
    focus. Python encodes; the widget shows."""

    _esm = _SVG + r"""
    function render({ model, el }) {
      const L = model.get("labels"), dec = model.get("decimal");
      const root = el_("div", "ea-w ea-play", el);
      const task = el_("p", "ea-task", root); el_("span", "ea-kick", task, L.task_kick); task.insertAdjacentHTML("beforeend", L.task);
      const box = el_("textarea", "", root); box.rows = 2; box.value = model.get("text"); box.setAttribute("aria-label", L.box);
      box.maxLength = 2000;   // Python tokenizes at every pause: a long paste would hold up the lesson
      const row = el_("div", "ea-row", root); row.style.margin = "0.5rem 0";
      const again = el_("button", "ea-quiet", row, "↻ " + L.again); const count = el_("div", "ea-readout", row); count.setAttribute("aria-live", "polite");
      if (!model.get("examples").length) again.style.display = "none";   // nothing to show again: no button that would put "undefined" in the box
      const tiles = el_("div", "ea-tiles", root);   // not read out at every key: the count says how many, a tile itself says what it is
      const info = el_("div", "ea-readout", root); info.setAttribute("aria-hidden", "true");
      let wait = null, ex = 0;
      const show = s => s.replace(/ /g, "␣").replace(/\n/g, "↵");
      function draw() {
        const toks = model.get("tokens"); tiles.innerHTML = "";
        toks.forEach(([id, s], j) => { const t = el_("span", "ea-tile" + (j % 2 ? " ea-odd" : ""), tiles); t.tabIndex = 0;
          const lead = s.startsWith(" "); t.innerHTML = (lead ? `<span class="ea-space">␣</span>` : "") + show(lead ? s.slice(1) : s).replace(/[<&]/g, c => c === "<" ? "&lt;" : "&amp;");
          const k = [...new TextEncoder().encode(s)].length;
          t.setAttribute("aria-label", `${L.token} ${id}: ${s.replace(/ /g, "␣").replace(/\n/g, "↵")}, ${k} ${k === 1 && L.byte ? L.byte : L.bytes}`);
          const tell = () => { info.innerHTML = `${L.token} <b>${id}</b>: <code>${s.replace(/ /g, "␣").replace(/[<&]/g, c => c === "<" ? "&lt;" : "&amp;")}</code> · ${k} ${k === 1 && L.byte ? L.byte : L.bytes}`; };
          t.addEventListener("mouseenter", tell); t.addEventListener("focus", tell); });
        const chars = [...box.value].length, n = toks.length;
        count.innerHTML = `<b>${chars}</b> ${chars === 1 && L.char ? L.char : L.chars} → <b>${n}</b> ${n === 1 && L.one ? L.one : L.tokens}` + (n ? ` · ${(chars / n).toFixed(1).replace(".", dec)} ${L.per}` : "");
      }
      box.addEventListener("input", () => { clearTimeout(wait); wait = setTimeout(() => { model.set("text", box.value); model.save_changes(); }, 90); });
      again.onclick = () => { const e = model.get("examples"); ex = (ex + 1) % e.length; box.value = e[ex]; model.set("text", box.value); model.save_changes(); };
      model.on("change:tokens", draw);
      draw();
    }
    export default { render };
    """
    _css = _PLAY_CSS + r"""
.ea-play textarea { min-height: 3.2rem; font-size: 1.1rem; }
.ea-play .ea-tiles { display: flex; flex-wrap: wrap; gap: 4px; margin: 0.2rem 0 0.5rem; }
.ea-play .ea-tile { font-family: "JetBrains Mono var", "JetBrains Mono", ui-monospace, monospace; font-size: 0.95rem; font-variant-ligatures: none;
  padding: 0.18rem 0.4rem; border-radius: 5px; background: var(--ea-data-wash); color: var(--ea-data-text); border: 1px solid transparent; white-space: pre; cursor: default; transition: border-color 120ms; }
.ea-play .ea-tile.ea-odd { background: var(--ea-inset); }
.ea-play .ea-tile:hover { border-color: var(--ea-data); }
.ea-play .ea-tile:focus-visible { border-color: var(--ea-data); outline: 2px solid var(--ea-ink); outline-offset: 2px; }
.ea-play .ea-space { color: var(--ea-ink-3); }
"""
    labels = traitlets.Dict({}).tag(sync=True)
    decimal = traitlets.Unicode(".").tag(sync=True)
    text = traitlets.Unicode("").tag(sync=True)
    tokens = traitlets.List([]).tag(sync=True)
    examples = traitlets.List([]).tag(sync=True)

    LABELS = ("task_kick", "task", "box", "again", "chars", "tokens", "per", "token", "bytes")   # and, if given, the
    # singular of three of them: char, one (for "1 token") and byte

    def __init__(self, encode, decode, **kw):
        super().__init__(**kw)
        _need(self, self.LABELS)
        self._encode, self._decode = encode, decode
        self._compute()
        self.observe(lambda _: self._compute(), names=["text"])

    def _compute(self):
        ids = self._encode(self.text)
        self.tokens = [[i, self._decode([i])] for i in ids]


class Table(anywidget.AnyWidget):
    """The whole table of an operation a ∘ b modulo p, a cell per equation: filled where the equation is training data,
    open where the model has to find the answer itself. Point at a cell, and the cells with the same answer keep their
    strength while the rest fades, the open ones with a ring: the pattern the model can learn from the filled ones. Set
    the fraction, or take another seed. Python makes the split (with the lesson's own code) and the answers; the widget
    draws them."""

    _esm = _SVG + r"""
    function render({ model, el }) {
      const L = model.get("labels"), dec = model.get("decimal"), p = model.get("p");
      const big = v => String(v).replace(/\B(?=(\d{3})+(?!\d))/g, dec === "," ? "." : ",");
      const root = el_("div", "ea-w ea-play", el);
      const task = el_("p", "ea-task", root); el_("span", "ea-kick", task, L.task_kick); task.insertAdjacentHTML("beforeend", L.task);
      const wrap = el_("div", "ea-table", root);
      const cv = el_("canvas", "", wrap); cv.setAttribute("role", "img"); cv.setAttribute("aria-label", L.alt); cv.tabIndex = 0;
      const side = el_("div", "ea-side", wrap);
      const readout = el_("div", "ea-readout ea-cell", side); readout.setAttribute("aria-live", "polite");
      const counts = el_("div", "ea-readout", side);
      el_("div", "ea-key", side, [["train", L.train], ["val", L.val], ["same", L.ring]].map(([k, t]) => `<div><span class="ea-sw ea-sw-${k}"></span><span>${t}</span></div>`).join(""));
      const row = el_("div", "ea-row", root);
      const fbox = el_("label", "ea-lr", row); el_("span", "ea-name ea-word", fbox, L.fraction_name);   // what you set: cyan
      const fr = el_("input", "", fbox); fr.type = "range"; fr.min = 5; fr.max = 95; fr.step = 5; fr.setAttribute("aria-label", L.fraction_name);
      const fout = el_("output", "", fbox);
      const again = el_("button", "ea-quiet", row, "↻ " + L.seed);
      const start = model.get("point");
      let at = start.length === 2 ? start : null, cell = 4;
      const css = n => getComputedStyle(root).getPropertyValue(n).trim();
      function draw() {
        const mask = model.get("mask"), ans = model.get("answers"), room = wrap.clientWidth, dpr = window.devicePixelRatio || 1;
        cell = Math.max(2, Math.min(5, Math.floor(Math.min(room > 560 ? room - 200 : room, 5 * p) / p)));
        const size = cell * p, gap = cell > 3 ? 1 : 0;
        cv.width = size * dpr; cv.height = size * dpr; cv.style.width = size + "px"; cv.style.height = size + "px";
        const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.fillStyle = css("--ea-inset"); g.fillRect(0, 0, size, size);
        // the cells with the answer of the cell pointed at stay at full strength, the rest fades: emphasis, not a role
        // (style guide 1.7); the open ones among them get a ring of ink
        const c = at ? ans[at[0] * p + at[1]] : null;
        g.fillStyle = css("--ea-data");
        for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) if (mask[a * p + b] === "1") {
          g.globalAlpha = c === null || ans[a * p + b] === c ? 1 : 0.38; g.fillRect(b * cell, a * cell, cell - gap, cell - gap); }
        g.globalAlpha = 1;
        if (at) {
          g.strokeStyle = css("--ea-ink"); g.lineWidth = 1;
          for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) if (ans[a * p + b] === c && mask[a * p + b] !== "1") g.strokeRect(b * cell + 0.5, a * cell + 0.5, cell - 2, cell - 2);
          g.lineWidth = 2; g.strokeRect(at[1] * cell - 1, at[0] * cell - 1, cell + 1, cell + 1);
        }
        const n = [...mask].filter(c => c === "1").length;
        counts.innerHTML = `<b>${big(n)}</b> ${L.of} <b>${big(p * p)}</b> ${L.counts}`;
        const f = Math.round(model.get("fraction") * 100); fr.value = f; fout.textContent = f + "%";
        fbox.style.setProperty("--p", ((f - 5) / 90 * 100) + "%");
        tell();
      }
      const say = h => { if (readout.innerHTML !== h) readout.innerHTML = h; };   // a screen reader hears a change only
      function tell() {
        if (!at) { say(L.point); return; }
        const [a, b] = at, i = a * p + b, mask = model.get("mask"), ans = model.get("answers"), known = mask[i] === "1", c = ans[i];
        let same = 0, seen = 0;
        for (let j = 0; j < p * p; j++) if (ans[j] === c) { same++; if (mask[j] === "1") seen++; }
        say(`<span class="ea-eq">${a} ${L.op} ${b} = ${known ? c : "?"}</span><br>${known ? L.is_train : L.is_val}<br>`
          + L.same.replace("{n}", big(same)).replace("{k}", big(seen)));
      }
      function pick(ev) { const r = cv.getBoundingClientRect(), b = Math.floor((ev.clientX - r.left) / cell), a = Math.floor((ev.clientY - r.top) / cell);
        return a >= 0 && a < p && b >= 0 && b < p ? [a, b] : null; }
      cv.addEventListener("pointermove", ev => { const c = pick(ev); if (c) { at = c; draw(); } });
      cv.addEventListener("keydown", ev => {
        const d = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[ev.key]; if (!d) return;
        ev.preventDefault(); at = at || [0, 0];
        at = [Math.min(p - 1, Math.max(0, at[0] + d[0])), Math.min(p - 1, Math.max(0, at[1] + d[1]))]; draw(); });
      fr.addEventListener("input", () => { model.set("fraction", Number(fr.value) / 100); model.save_changes(); });
      again.onclick = () => { model.set("seed", model.get("seed") + 1); model.save_changes(); };
      model.on("change:mask", draw);
      const ro = new ResizeObserver(() => draw()); ro.observe(wrap);
      draw();
      return () => { model.off("change:mask", draw); ro.disconnect(); };
    }
    export default { render };
    """
    _css = _PLAY_CSS + r"""
.ea-play .ea-table { display: flex; flex-wrap: wrap; gap: 0.8rem 1.2rem; align-items: flex-start; margin-bottom: 0.5rem; }
.ea-play .ea-table canvas { display: block; border-radius: 4px; cursor: crosshair; }
.ea-play .ea-table canvas:focus-visible { outline: 2px solid var(--ea-ink); outline-offset: 3px; }
.ea-play .ea-side { flex: 1 1 10rem; min-width: 10rem; display: flex; flex-direction: column; gap: 0.6rem; }
.ea-play .ea-cell { min-height: 4.4rem; }
.ea-play .ea-eq { font-family: "JetBrains Mono var", "JetBrains Mono", ui-monospace, monospace; font-size: 1.25rem; font-weight: 600; color: var(--ea-ink); }
.ea-play .ea-key { font-size: 0.9rem; color: var(--ea-ink-2); line-height: 1.35; display: flex; flex-direction: column; gap: 0.4rem; }
.ea-play .ea-key > div { display: flex; gap: 0.45rem; align-items: flex-start; }
.ea-play .ea-sw { flex: none; width: 0.85rem; height: 0.85rem; border-radius: 2px; margin-top: 0.15rem; box-sizing: border-box; }
.ea-play .ea-sw-train { background: var(--ea-data); }
.ea-play .ea-sw-val { background: var(--ea-inset); border: 1px solid var(--ea-line); }
.ea-play .ea-sw-same { background: var(--ea-inset); border: 1.5px solid var(--ea-ink); }
.ea-play .ea-lr .ea-name.ea-word { font-family: inherit; font-style: normal; font-size: 0.95rem; }
.ea-play .ea-lr output { min-width: 3rem; }
"""
    labels = traitlets.Dict({}).tag(sync=True)
    decimal = traitlets.Unicode(".").tag(sync=True)
    p = traitlets.Int(97).tag(sync=True)
    fraction = traitlets.Float(0.3).tag(sync=True)
    seed = traitlets.Int(0).tag(sync=True)
    point = traitlets.List([]).tag(sync=True)
    mask = traitlets.Unicode("").tag(sync=True)
    answers = traitlets.List([]).tag(sync=True)

    LABELS = ("task_kick", "task", "alt", "train", "val", "ring", "fraction_name", "seed", "of", "counts", "point", "op",
              "is_train", "is_val", "same")

    def __init__(self, split, answer, **kw):
        """`split(p, fraction, seed)` gives the training equations as (a, b, c); `answer(a, b)` the c of every cell.
        `point` is the cell [a, b] pointed at from the start, so the pattern shows without touching anything."""
        super().__init__(**kw)
        _need(self, self.LABELS)
        if self.point and not (len(self.point) == 2 and all(isinstance(v, int) and 0 <= v < self.p for v in self.point)):
            raise ValueError("Table needs a point [a, b] within the table")
        if not (0.05 <= self.fraction <= 0.95 and math.isclose(self.fraction * 20, round(self.fraction * 20))):
            raise ValueError("Table needs a fraction its slider can show: 0.05 to 0.95, in steps of 0.05")
        self._split = split
        self.answers = [answer(a, b) for a in range(self.p) for b in range(self.p)]
        self._compute()
        self.observe(lambda _: self._compute(), names=["fraction", "seed"])

    def _compute(self):
        cells = ["0"] * (self.p * self.p)
        for a, b, _ in self._split(self.p, self.fraction, self.seed):
            cells[a * self.p + b] = "1"
        self.mask = "".join(cells)


class Explore(anywidget.AnyWidget):
    """A picture that follows a slider at once. Python draws the picture and the sentence under it for every stop of
    the slider, once; the widget flips between them while you slide, or drag along the chart that shows the value
    (chart.figure(drag=...)), without asking Python, and gives the value back to Python for the check and the
    progress. For the lessons where a slider moves a picture: L.explore makes one."""

    _esm = _SVG + r"""
    function render({ model, el }) {
      const L = model.get("labels"), dec = model.get("decimal"), role = model.get("role"), values = model.get("values");
      const root = el_("div", "ea-w ea-play ea-explore", el);
      root.style.setProperty("--c", role ? `var(--ea-${role})` : "var(--ea-ink-2)");
      root.style.setProperty("--ct", role ? `var(--ea-${role}-text)` : "var(--ea-ink)");
      root.style.setProperty("--wash", role ? `var(--ea-${role}-wash)` : "var(--ea-line)");
      if (L.task) { const task = el_("p", "ea-task", root); el_("span", "ea-kick", task, L.task_kick); task.insertAdjacentHTML("beforeend", L.task); }
      const row = el_("div", "ea-slide", root);
      const name = el_("span", "ea-slide-name", row, L.name);
      if (/^\p{L}{2,}$/u.test(name.textContent)) name.style.fontStyle = "normal";   // a word, not a symbol such as w₁
      const inp = el_("input", "", row); inp.type = "range"; inp.min = 0; inp.max = values.length - 1; inp.step = 1;
      inp.setAttribute("aria-label", name.textContent);
      const out = el_("output", "ea-slide-value", row);
      const pic = el_("div", "ea-pic", root);
      const said = el_("div", "ea-said", root);
      const heard = el_("div", "ea-heard", root); heard.setAttribute("aria-live", "polite");   // what a screen reader says, once per pause
      const spoken = node => {   // a formula said once, as its MathML says it: not its TeX, nor the letters drawn for the eye
        if (node.nodeType === 3) return node.textContent;
        if (node.nodeType !== 1 || ["annotation", "style", "script"].includes(node.localName) || node.getAttribute("aria-hidden") === "true") return "";
        if (node.localName === "marimo-tex") {   // KaTeX draws it in a shadow root; until then, its TeX without the marks
          const m = node.shadowRoot && node.shadowRoot.querySelector(".katex-mathml");
          return m ? spoken(m) : node.textContent.replace(/\|\|[(\[]|\|\|[)\]]/g, "");
        }
        return Array.from(node.childNodes, spoken).join("");
      };
      const digits = model.get("digits");
      // a whole number of five digits or more with a separator, as the text writes it: 12,800, or 12.800 in Dutch;
      // four digits without one, 3200
      const show = v => {
        const s = Number(v).toFixed(digits).replace(".", dec);
        return (digits === 0 && Math.abs(v) >= 10000 ? s.replace(/\B(?=(\d{3})+(?!\d))/g, dec === "," ? "." : ",") : s).replace("-", "−");
      };
      let shown = -1;
      function draw() {
        const i = model.get("index"), f = model.get("frames")[i];
        inp.value = i; out.textContent = show(values[i]); inp.setAttribute("aria-valuetext", out.textContent);
        row.style.setProperty("--p", (i / Math.max(1, values.length - 1) * 100) + "%");
        if (i !== shown) { pic.innerHTML = f[0]; said.innerHTML = f[1]; shown = i; zone(); }
      }
      // the picture follows at once; Python hears the value when you pause, so a check behind it runs once, not at
      // every step of a drag
      let wait = null, saying = null;
      const say = (tries = 0) => {   // once KaTeX has drawn the formulas in the sentence, or after half a second
        clearTimeout(saying); saying = null;
        const drawing = [...said.querySelectorAll("marimo-tex")].some(t => !(t.shadowRoot && t.shadowRoot.querySelector(".katex-mathml")));
        if (drawing && tries < 10) { saying = setTimeout(() => say(tries + 1), 50); return; }
        heard.textContent = spoken(said);
      };
      const send = () => { clearTimeout(wait); wait = null; model.save_changes(); say(); };
      function go(i) {
        i = Math.max(0, Math.min(values.length - 1, i)); if (i === model.get("index")) return;
        model.set("index", i); draw(); clearTimeout(wait); clearTimeout(saying); saying = null; wait = setTimeout(send, 150);   // moved on: no word of a stop you passed
      }
      inp.addEventListener("input", () => go(Number(inp.value)));
      // dragging along the x-axis of the chart that shows the value: a see-through strip over it to grab, and the
      // listeners on the box around the picture, as the picture itself is swapped at every step
      const axis = () => { const svg = pic.querySelector("svg[data-drag]"); return svg && [svg, svg.dataset.drag.split(" ").map(Number)]; };
      function zone() {
        const a = axis(); if (!a) return;
        const [svg, [x0, x1]] = a;
        mk("rect", { class: "ea-zone", x: x0 - 8, y: 0, width: x1 - x0 + 16, height: svg.viewBox.baseVal.height }, svg);
      }
      function pick(ev) {
        const [svg, [x0, x1, lo, hi, log]] = axis(), r = svg.getBoundingClientRect();
        const t = ((ev.clientX - r.left) / r.width * svg.viewBox.baseVal.width - x0) / (x1 - x0);
        const v = log ? 10 ** (Math.log10(lo) + t * (Math.log10(hi) - Math.log10(lo))) : lo + t * (hi - lo);
        const far = w => log ? Math.abs(Math.log10(w) - Math.log10(v)) : Math.abs(w - v);   // on a log axis, as far as it looks
        let best = 0; values.forEach((w, j) => { if (far(w) < far(values[best])) best = j; });
        go(best);
      }
      const down = ev => { if (ev.button !== 0 || !ev.target.classList.contains("ea-zone")) return; pic.setPointerCapture(ev.pointerId); pic.classList.add("ea-dragging"); pick(ev); };
      const move = ev => { if (pic.hasPointerCapture(ev.pointerId)) pick(ev); };
      const up = ev => { if (pic.hasPointerCapture(ev.pointerId)) pic.releasePointerCapture(ev.pointerId); pic.classList.remove("ea-dragging"); if (wait) send(); };
      pic.addEventListener("pointerdown", down); pic.addEventListener("pointermove", move);
      pic.addEventListener("pointerup", up); pic.addEventListener("pointercancel", up);
      const redraw = () => { shown = -1; draw(); };
      model.on("change:index", draw); model.on("change:frames", redraw);
      draw();
      inp.addEventListener("change", () => { if (wait) send(); });   // the slider let go, or a key pressed
      return () => { if (wait) send(); clearTimeout(saying); model.off("change:index", draw); model.off("change:frames", redraw); };
    }
    export default { render };
    """
    _css = _PLAY_CSS + r"""
.ea-explore .ea-slide { display: flex; align-items: center; gap: 0.75rem; max-width: 26rem; margin: 0.1rem 0 0.2rem; }
.ea-explore .ea-slide-name { font-family: "Source Serif var", "Source Serif 4", Georgia, serif; font-style: italic; font-size: 1.15rem;
  font-weight: 600; color: var(--ct); min-width: 1.2rem; }
.ea-explore .ea-slide input { -webkit-appearance: none; appearance: none; flex: 1; height: 1.5rem; margin: 0; background: transparent; cursor: pointer; }
.ea-explore .ea-slide input::-webkit-slider-runnable-track { height: 6px; border-radius: 3px; background: linear-gradient(to right, var(--c) 0 var(--p), var(--ea-line) var(--p) 100%); }
.ea-explore .ea-slide input::-moz-range-track { height: 6px; border-radius: 3px; background: var(--ea-line); }
.ea-explore .ea-slide input::-moz-range-progress { height: 6px; border-radius: 3px; background: var(--c); }
.ea-explore .ea-slide input::-webkit-slider-thumb { -webkit-appearance: none; width: 18px; height: 18px; margin-top: -6px; border-radius: 50%;
  background: var(--ea-page); border: 2px solid var(--c); }
.ea-explore .ea-slide input::-moz-range-thumb { width: 14px; height: 14px; border-radius: 50%; background: var(--ea-page); border: 2px solid var(--c); }
.ea-explore .ea-slide input:hover::-webkit-slider-thumb, .ea-explore .ea-slide input:focus-visible::-webkit-slider-thumb { background: var(--wash); }
.ea-explore .ea-slide input:hover::-moz-range-thumb, .ea-explore .ea-slide input:focus-visible::-moz-range-thumb { background: var(--wash); }
.ea-explore .ea-slide input:focus-visible { outline: 2px solid var(--ea-ink); outline-offset: 3px; border-radius: 4px; }
.ea-explore .ea-slide-value { font-variant-numeric: tabular-nums; font-weight: 600; color: var(--ct); min-width: 3.5rem; text-align: right; }
.ea-explore .ea-pic figure { margin: 0.2rem 0 0.4rem !important; }
.ea-explore .ea-zone { fill: transparent; cursor: ew-resize; touch-action: none; }
.ea-explore .ea-dragging { user-select: none; }
.ea-explore .ea-said { color: var(--ea-ink); min-height: 1.6rem; }
.ea-explore .ea-heard { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
.ea-explore .ea-said i, .ea-explore .ea-task i { font-family: "Source Serif var", "Source Serif 4", Georgia, serif; font-size: 1.08em; }
.ea-explore .ea-num { font-variant-numeric: tabular-nums; }
.ea-explore .c-data { color: var(--ea-data-text); font-weight: 600; }
.ea-explore .c-param { color: var(--ea-param-text); font-weight: 600; }
.ea-explore .c-pred { color: var(--ea-pred-text); font-weight: 600; }
.ea-explore .c-residual { color: var(--ea-residual-text); font-weight: 600; }
.ea-explore .c-loss { color: var(--ea-loss-text); font-weight: 600; }
.ea-explore .c-grad { color: var(--ea-grad-text); font-weight: 600; }
.ea-explore .c-hyper { color: var(--ea-hyper-text); font-weight: 600; }
"""
    labels = traitlets.Dict({}).tag(sync=True)
    decimal = traitlets.Unicode(".").tag(sync=True)
    role = traitlets.Unicode("param").tag(sync=True)
    values = traitlets.List([]).tag(sync=True)
    digits = traitlets.Int(1).tag(sync=True)
    index = traitlets.Int(0).tag(sync=True)
    frames = traitlets.List([]).tag(sync=True)

    LABELS = ("name",)
    MOST = 241   # every frame is drawn up front and sent at once: more stops would make the lesson slow to open
    WIDEST = {1: 460, 2: 330}   # px of a chart.figure panel, alone or two side by side: its text stays 11 px at 700 px

    def __init__(self, frame, *, low: float | None = None, high: float | None = None, step: float | None = None,
                 values: list | None = None, value: float | None = None, **kw):
        """`frame(v)` gives the picture at the value v, as SVG text, and the sentence under it, as HTML: a pair, or
        the picture alone. The stops are low to high in steps of step, or the `values` given, from small to large,
        such as budgets that double. With a task, the labels also give task_kick and task."""
        super().__init__(**kw)
        _need(self, self.LABELS + (("task_kick",) if self.labels.get("task") else ()))
        if values is not None:
            if not 2 <= len(values) <= self.MOST or list(values) != sorted(set(values)):
                raise ValueError(f"Explore needs 2 to {self.MOST} values, each larger than the one before: {values}")
            stops = list(values)
        else:
            n = round((high - low) / step)
            if not math.isclose(low + n * step, high, abs_tol=step * 1e-6):
                raise ValueError(f"Explore needs a step that fits the range a whole number of times: {step} does not fit {low} to {high}")
            if not 1 <= n < self.MOST:
                raise ValueError(f"Explore draws every stop up front: give it 2 to {self.MOST} stops, not {n + 1}")
            stops = [low, step]
        decimals = [max(0, -Decimal(str(x)).normalize().as_tuple().exponent) for x in stops]   # 0.25 has two, 5 none
        if max(decimals) > 6:
            raise ValueError(f"Explore shows its values with at most 6 decimals: {', '.join(map(str, stops))} need more")
        self.values = list(values) if values is not None else [round(low + i * step, 10) for i in range(n + 1)]
        self.digits = max(decimals)
        self.frames = [[f, ""] if isinstance(f, str) else list(f) for f in map(frame, self.values)]
        widths = [int(w) for w in re.findall(r"<svg [^>]*max-width:(\d+)px", self.frames[0][0])]
        if widths and max(widths) > self.WIDEST.get(len(widths), 330):
            raise ValueError(f"Explore shows {len(widths)} chart(s) of {max(widths)} px: in a play block one chart is at most "
                             f"{self.WIDEST[1]} px wide and two side by side at most {self.WIDEST[2]} px each, or their text "
                             "gets smaller than 11 px in a window of 700 px")
        self.value = self.values[0] if value is None else value

    @property
    def value(self) -> float:
        """The value the slider is at."""
        return self.values[self.index]

    @value.setter
    def value(self, v: float) -> None:
        self.index = min(range(len(self.values)), key=lambda i: abs(self.values[i] - v))
