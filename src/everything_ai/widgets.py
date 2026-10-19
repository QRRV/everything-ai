# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""The course's own small widgets: a question with options (a choice question or a guess),
an open question with a model answer, a row of hints, a sort, and a slider.

They are drawn only with the course's tokens (var(--ea-*)), so they follow the light and
dark theme, and all their code ships with the course: nothing comes from the internet.
"""
from __future__ import annotations

import anywidget
import traitlets

_BASE_CSS = r"""
:host, .ea-w { font-family: "Inter var", Inter, system-ui, sans-serif; color: var(--ea-ink); }
.ea-w { max-width: 40rem; font-size: 0.97rem; line-height: 1.55; }
.ea-w * { box-sizing: border-box; }
.ea-w code { font-family: "JetBrains Mono var", "JetBrains Mono", ui-monospace, monospace; font-size: 0.86em;
  font-variant-ligatures: none; background: var(--ea-inset); border-radius: 4px; padding: 0.06em 0.3em; }
.ea-w .ea-q { font-family: "Source Serif var", "Source Serif 4", Georgia, serif; font-size: 1.125rem; line-height: 1.6; margin: 0 0 0.6rem; }
.ea-w .ea-kick { font-size: 0.72rem; font-weight: 650; letter-spacing: 0.07em; text-transform: uppercase;
  color: var(--ea-page); background: var(--ea-ink); border-radius: 4px; padding: 0.12rem 0.42rem; margin-right: 0.5rem;
  font-family: "Inter var", Inter, system-ui, sans-serif; vertical-align: 0.15em; }
.ea-w .ea-opts { display: flex; flex-direction: column; gap: 0.4rem; margin: 0; padding: 0; border: 0; }
.ea-w label.ea-opt { display: flex; gap: 0.65rem; align-items: baseline; padding: 0.5rem 0.8rem; border: 1px solid var(--ea-line);
  border-radius: 8px; cursor: pointer; background: var(--ea-page); transition: border-color 120ms, background-color 120ms; }
.ea-w label.ea-opt:hover { border-color: var(--ea-axis); }
.ea-w label.ea-opt.ea-picked { border-color: var(--ea-ink); background: var(--ea-inset); }
.ea-w label.ea-opt input { accent-color: var(--ea-ink); margin: 0; transform: translateY(0.1rem); }
.ea-w input:focus-visible, .ea-w button:focus-visible, .ea-w textarea:focus-visible, .ea-w summary:focus-visible {
  outline: 2px solid var(--ea-ink); outline-offset: 2px; }
.ea-w .ea-fb { margin-top: 0.7rem; }
.ea-w .ea-check { border: 1px solid var(--ea-line); border-radius: 8px; padding: 0.7rem 1rem 0.75rem 2.7rem; position: relative; }
.ea-w .ea-check svg { position: absolute; left: 0.95rem; top: 0.85rem; width: 1.1rem; height: 1.1rem; color: var(--ea-ink-2); }
.ea-w .ea-check--ok { border-color: var(--ea-ink); box-shadow: inset 3px 0 0 var(--ea-ink); }
.ea-w .ea-check--ok svg { color: var(--ea-ink); }
.ea-w .ea-state { font-weight: 680; }
.ea-w .ea-reveal { background: var(--ea-inset); border-radius: 8px; padding: 0.75rem 1rem; }
.ea-w .ea-reveal p { margin: 0.25rem 0; }
.ea-w .ea-muted { color: var(--ea-ink-2); font-size: 0.9rem; }
.ea-w textarea { width: 100%; min-height: 5.5rem; font: inherit; font-family: "Source Serif var", "Source Serif 4", Georgia, serif;
  font-size: 1.05rem; color: var(--ea-ink); background: var(--ea-page); border: 1px solid var(--ea-axis); border-radius: 8px; padding: 0.6rem 0.75rem; resize: vertical; }
.ea-w button { font: inherit; font-weight: 600; color: var(--ea-ink); background: var(--ea-page); border: 1px solid var(--ea-ink);
  border-radius: 6px; padding: 0.35rem 0.85rem; cursor: pointer; margin-top: 0.55rem; }
.ea-w button:disabled { color: var(--ea-ink-3); border-color: var(--ea-line); cursor: default; }
.ea-w ul.ea-points { list-style: none; padding: 0; margin: 0.5rem 0 0; }
.ea-w ul.ea-points li { margin: 0.3rem 0; }
.ea-w ul.ea-points label { display: flex; gap: 0.55rem; align-items: baseline; cursor: pointer; }
.ea-w ul.ea-points input { accent-color: var(--ea-ink); }
.ea-w details { border-top: 1px solid var(--ea-line); padding: 0.5rem 0; }
.ea-w details:last-child { border-bottom: 1px solid var(--ea-line); }
.ea-w summary { cursor: pointer; font-weight: 600; color: var(--ea-ink-2); list-style: none; }
.ea-w summary::-webkit-details-marker { display: none; }
.ea-w summary::before { content: "+"; display: inline-block; width: 1.2rem; color: var(--ea-ink-3); font-weight: 500; }
.ea-w details[open] summary::before { content: "\2212"; }
.ea-w details[aria-disabled="true"] summary { color: var(--ea-ink-3); cursor: default; }
.ea-w .ea-hint-body { margin: 0.35rem 0 0.1rem 1.2rem; }
.ea-w .ea-hint-body p { margin: 0.2rem 0; }
.ea-w pre { font-family: "JetBrains Mono var", "JetBrains Mono", ui-monospace, monospace; font-size: 0.86rem; background: var(--ea-inset);
  border-radius: 6px; padding: 0.55rem 0.7rem; margin: 0.4rem 0; white-space: pre-wrap; font-variant-ligatures: none; }
@media (prefers-reduced-motion: reduce) { .ea-w * { transition: none !important; } }
"""

_ICON_OK = ('<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">'
            '<circle cx="8" cy="8" r="7" fill="currentColor" stroke="none"/>'
            '<path d="M4.8 8.3 7 10.4 11.3 5.8" stroke="var(--ea-page)" stroke-width="1.8"/></svg>')
_ICON_NOTYET = ('<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">'
                '<circle cx="8" cy="8" r="6.2"/><path d="M8 1.8 A6.2 6.2 0 0 1 14.2 8 L8 8 Z" fill="currentColor" stroke="none"/></svg>')


class Question(anywidget.AnyWidget):
    """A question with options. kind="choice" is graded and explains every option;
    kind="guess" is never graded and shows what really happens next to your guess."""

    _esm = r"""
    function render({ model, el }) {
      const root = document.createElement("div"); root.className = "ea-w";
      const words = model.get("words");
      const q = document.createElement("div"); q.className = "ea-q";
      q.innerHTML = (model.get("label") ? `<span class="ea-kick">${model.get("label")}</span>` : "") + model.get("question");
      const opts = document.createElement("fieldset"); opts.className = "ea-opts";
      const legend = document.createElement("legend"); legend.style.cssText = "position:absolute;left:-9999px";
      legend.textContent = q.textContent; opts.appendChild(legend);
      const fb = document.createElement("div"); fb.className = "ea-fb"; fb.setAttribute("aria-live", "polite");
      const name = "q" + Math.random().toString(36).slice(2);
      const labels = [];
      for (const o of model.get("options")) {
        const lab = document.createElement("label"); lab.className = "ea-opt";
        const inp = document.createElement("input"); inp.type = "radio"; inp.name = name; inp.value = o.key;
        const span = document.createElement("span"); span.innerHTML = o.html;
        lab.append(inp, span); opts.appendChild(lab); labels.push([lab, inp]);
        inp.addEventListener("change", () => { model.set("value", o.key); model.save_changes(); });
      }
      function show() {
        const v = model.get("value");
        for (const [lab, inp] of labels) { inp.checked = inp.value === v; lab.classList.toggle("ea-picked", inp.checked); }
        if (!v) { fb.innerHTML = ""; return; }
        if (model.get("kind") === "guess") {
          const picked = model.get("options").find(o => o.key === v);
          fb.innerHTML = `<div class="ea-reveal"><p class="ea-muted">${words.your_guess}: ${picked ? picked.html : v}</p>${model.get("reveal")}</div>`;
        } else {
          const ok = v === model.get("correct");
          const text = (model.get("feedback") || {})[v] || "";
          fb.innerHTML = `<div class="ea-check ${ok ? "ea-check--ok" : ""}" role="status">${ok ? model.get("icon_ok") : model.get("icon_notyet")}` +
            `<span class="ea-state">${ok ? words.correct : words.not_yet}.</span> ${text}</div>`;
        }
      }
      model.on("change:value", show);
      root.append(q, opts, fb); el.appendChild(root); show();
    }
    export default { render };
    """
    _css = _BASE_CSS
    kind = traitlets.Unicode("choice").tag(sync=True)
    label = traitlets.Unicode("").tag(sync=True)
    question = traitlets.Unicode("").tag(sync=True)
    options = traitlets.List([]).tag(sync=True)
    feedback = traitlets.Dict({}).tag(sync=True)
    correct = traitlets.Unicode("").tag(sync=True)
    reveal = traitlets.Unicode("").tag(sync=True)
    words = traitlets.Dict({}).tag(sync=True)
    icon_ok = traitlets.Unicode(_ICON_OK).tag(sync=True)
    icon_notyet = traitlets.Unicode(_ICON_NOTYET).tag(sync=True)
    value = traitlets.Unicode("").tag(sync=True)


class Explain(anywidget.AnyWidget):
    """An open question. Nobody grades it: you put your answer next to a model answer
    and tick the points that are in yours."""

    _esm = r"""
    function render({ model, el }) {
      const root = document.createElement("div"); root.className = "ea-w";
      const words = model.get("words");
      const q = document.createElement("div"); q.className = "ea-q"; q.innerHTML = model.get("question");
      const ta = document.createElement("textarea"); ta.placeholder = model.get("placeholder");
      ta.setAttribute("aria-label", q.textContent);
      const btn = document.createElement("button"); btn.type = "button"; btn.textContent = words.compare;
      const out = document.createElement("div"); out.className = "ea-fb"; out.setAttribute("aria-live", "polite");
      function value() { return Object.assign({ text: "", ticked: [], compared: false }, model.get("value") || {}); }
      function save(v) { model.set("value", v); model.save_changes(); }
      ta.value = value().text;
      let timer = null;
      ta.addEventListener("input", () => {
        btn.disabled = !ta.value.trim();
        clearTimeout(timer); timer = setTimeout(() => save(Object.assign(value(), { text: ta.value })), 400);
      });
      btn.addEventListener("click", () => save(Object.assign(value(), { text: ta.value, compared: true })));
      function show() {
        const v = value();
        btn.disabled = !ta.value.trim();
        btn.hidden = v.compared;
        if (!v.compared) { out.innerHTML = ""; return; }
        const pts = model.get("points");
        out.innerHTML = `<div class="ea-reveal">${model.get("model_answer")}<p class="ea-muted">${words.tick}</p><ul class="ea-points"></ul></div>`;
        const ul = out.querySelector("ul");
        pts.forEach((p, i) => {
          const li = document.createElement("li"); const lab = document.createElement("label");
          const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = !!(v.ticked || [])[i];
          cb.addEventListener("change", () => { const nv = value(); const t = pts.map((_, j) => !!(nv.ticked || [])[j]); t[i] = cb.checked; nv.ticked = t; save(nv); });
          const span = document.createElement("span"); span.innerHTML = p;
          lab.append(cb, span); li.appendChild(lab); ul.appendChild(li);
        });
      }
      model.on("change:value", show);
      root.append(q, ta, btn, out); el.appendChild(root); show();
    }
    export default { render };
    """
    _css = _BASE_CSS
    question = traitlets.Unicode("").tag(sync=True)
    placeholder = traitlets.Unicode("").tag(sync=True)
    model_answer = traitlets.Unicode("").tag(sync=True)
    points = traitlets.List([]).tag(sync=True)
    words = traitlets.Dict({}).tag(sync=True)
    value = traitlets.Dict({}).tag(sync=True)


class Hints(anywidget.AnyWidget):
    """A row of hints that each say more, with the solution last. A hint opens only after the one before it."""

    _esm = r"""
    function render({ model, el }) {
      const root = document.createElement("div"); root.className = "ea-w";
      const hints = model.get("hints"); const words = model.get("words");
      const items = hints.map((h, i) => ({ title: words.hint.replace("{n}", i + 1), body: h }));
      items.push({ title: words.solution, body: `<p class="ea-muted">${words.first_yourself}</p>` + model.get("solution") });
      const ds = items.map((it, i) => {
        const d = document.createElement("details");
        const s = document.createElement("summary"); s.textContent = it.title;
        const b = document.createElement("div"); b.className = "ea-hint-body"; b.innerHTML = it.body;
        d.append(s, b);
        s.addEventListener("click", (e) => {
          if (i > model.get("opened")) { e.preventDefault(); return; }   // one hint at a time
        });
        d.addEventListener("toggle", () => {
          if (d.open && i + 1 > model.get("opened")) { model.set("opened", i + 1); model.save_changes(); }
        });
        return d;
      });
      function show() {
        const n = model.get("opened");
        ds.forEach((d, i) => d.setAttribute("aria-disabled", i > n ? "true" : "false"));
      }
      model.on("change:opened", show);
      root.append(...ds); el.appendChild(root); show();
    }
    export default { render };
    """
    _css = _BASE_CSS
    hints = traitlets.List([]).tag(sync=True)
    solution = traitlets.Unicode("").tag(sync=True)
    words = traitlets.Dict({}).tag(sync=True)
    opened = traitlets.Int(0).tag(sync=True)   # how many items have been opened; len(hints) + 1 means the solution


class Sort(anywidget.AnyWidget):
    """Put every item in a group: one row per item, with the groups as buttons to pick from.
    The check cell below says what is right; the widget itself never grades."""

    _esm = r"""
    function render({ model, el }) {
      const root = document.createElement("div"); root.className = "ea-w ea-sort";
      if (model.get("question")) {
        const q = document.createElement("div"); q.className = "ea-q"; q.innerHTML = model.get("question"); root.appendChild(q);
      }
      const groups = model.get("groups");
      const rows = [];
      model.get("items").forEach((it, i) => {
        const fs = document.createElement("fieldset"); fs.className = "ea-sort-row";
        const lg = document.createElement("legend"); lg.className = "ea-sort-item"; lg.innerHTML = it.html;
        const seg = document.createElement("div"); seg.className = "ea-seg";
        const name = "s" + i + Math.random().toString(36).slice(2);
        const labs = groups.map(g => {
          const lab = document.createElement("label");
          const inp = document.createElement("input"); inp.type = "radio"; inp.name = name; inp.value = g.key;
          const span = document.createElement("span"); span.innerHTML = g.html;
          lab.append(inp, span); seg.appendChild(lab);
          inp.addEventListener("change", () => {
            const v = Object.assign({}, model.get("value")); v[it.key] = g.key;
            model.set("value", v); model.save_changes();
          });
          return [lab, inp];
        });
        fs.append(lg, seg); root.appendChild(fs); rows.push([it.key, labs]);
      });
      function show() {
        const v = model.get("value") || {};
        for (const [key, labs] of rows) for (const [lab, inp] of labs) {
          inp.checked = v[key] === inp.value; lab.classList.toggle("ea-picked", inp.checked);
        }
      }
      model.on("change:value", show);
      el.appendChild(root); show();
    }
    export default { render };
    """
    _css = _BASE_CSS + r"""
.ea-sort { max-width: 44rem; }
.ea-sort fieldset.ea-sort-row { border: 0; border-top: 1px solid var(--ea-line); margin: 0; padding: 0.65rem 0 0.7rem; min-width: 0; }
.ea-sort fieldset.ea-sort-row:last-child { border-bottom: 1px solid var(--ea-line); }
.ea-sort legend.ea-sort-item { float: left; width: 100%; padding: 0; margin: 0 0 0.45rem; }
.ea-sort .ea-seg { clear: both; display: flex; flex-wrap: wrap; gap: 0.4rem; }
.ea-sort .ea-seg label { position: relative; border: 1px solid var(--ea-line); border-radius: 999px; padding: 0.22rem 0.8rem;
  font-size: 0.9rem; cursor: pointer; background: var(--ea-page); color: var(--ea-ink-2); transition: border-color 120ms, background-color 120ms; }
.ea-sort .ea-seg label:hover { border-color: var(--ea-axis); color: var(--ea-ink); }
.ea-sort .ea-seg label.ea-picked { background: var(--ea-ink); border-color: var(--ea-ink); color: var(--ea-page); font-weight: 600; }
.ea-sort .ea-seg input { position: absolute; opacity: 0; width: 1px; height: 1px; }
.ea-sort .ea-seg label:has(input:focus-visible) { outline: 2px solid var(--ea-ink); outline-offset: 2px; }
"""
    question = traitlets.Unicode("").tag(sync=True)
    items = traitlets.List([]).tag(sync=True)
    groups = traitlets.List([]).tag(sync=True)
    value = traitlets.Dict({}).tag(sync=True)


class Slider(anywidget.AnyWidget):
    """A slider in the colour of the role of what it sets, such as orange for a parameter, with
    its name in front and its value behind it. A slider that sets no role is ink."""

    _esm = r"""
    function render({ model, el }) {
      const root = document.createElement("div"); root.className = "ea-w ea-slider";
      const role = model.get("role");
      root.style.setProperty("--c", role ? `var(--ea-${role})` : "var(--ea-ink-2)");
      root.style.setProperty("--ct", role ? `var(--ea-${role}-text)` : "var(--ea-ink)");
      root.style.setProperty("--wash", role ? `var(--ea-${role}-wash)` : "var(--ea-line)");
      const name = document.createElement("span"); name.className = "ea-slider-name"; name.innerHTML = model.get("label");
      if (/^\p{L}{2,}$/u.test(name.textContent)) name.style.fontStyle = "normal";   // a word, not a symbol such as w₁
      const inp = document.createElement("input"); inp.type = "range";
      for (const k of ["min", "max", "step"]) inp[k] = model.get(k);
      inp.setAttribute("aria-label", name.textContent);
      const out = document.createElement("output"); out.className = "ea-slider-value";
      const digits = (String(model.get("step")).split(".")[1] || "").length;
      const show = v => Number(v).toFixed(digits).replace("-", "−").replace(".", model.get("decimal"));
      function draw() {
        const v = model.get("value"); inp.value = v; out.textContent = show(v);
        inp.setAttribute("aria-valuetext", out.textContent);   // what a screen reader says: as shown, decimal comma too
        root.style.setProperty("--p", ((v - inp.min) / (inp.max - inp.min) * 100) + "%");
      }
      inp.addEventListener("input", () => { model.set("value", Number(inp.value)); model.save_changes(); draw(); });
      model.on("change:value", draw);
      root.append(name, inp, out); el.appendChild(root); draw();
    }
    export default { render };
    """
    _css = _BASE_CSS + r"""
.ea-slider { display: flex; align-items: center; gap: 0.75rem; max-width: 26rem; margin: 0.3rem 0; }
.ea-slider .ea-slider-name { font-family: "Source Serif var", "Source Serif 4", Georgia, serif; font-style: italic; font-size: 1.15rem;
  font-weight: 600; color: var(--ct); min-width: 1.2rem; }
.ea-slider input { -webkit-appearance: none; appearance: none; flex: 1; height: 1.5rem; margin: 0; background: transparent; cursor: pointer; }
.ea-slider input::-webkit-slider-runnable-track { height: 6px; border-radius: 3px; background: linear-gradient(to right, var(--c) 0 var(--p), var(--ea-line) var(--p) 100%); }
.ea-slider input::-moz-range-track { height: 6px; border-radius: 3px; background: var(--ea-line); }
.ea-slider input::-moz-range-progress { height: 6px; border-radius: 3px; background: var(--c); }
.ea-slider input::-webkit-slider-thumb { -webkit-appearance: none; width: 18px; height: 18px; margin-top: -6px; border-radius: 50%;
  background: var(--ea-page); border: 2px solid var(--c); transition: background-color 120ms; }
.ea-slider input::-moz-range-thumb { width: 14px; height: 14px; border-radius: 50%; background: var(--ea-page); border: 2px solid var(--c); }
.ea-slider input:hover::-webkit-slider-thumb, .ea-slider input:focus-visible::-webkit-slider-thumb { background: var(--wash); }
.ea-slider input:hover::-moz-range-thumb, .ea-slider input:focus-visible::-moz-range-thumb { background: var(--wash); }
.ea-slider input:focus-visible { outline: 2px solid var(--ea-ink); outline-offset: 3px; border-radius: 4px; }
@media (prefers-reduced-motion: reduce) { .ea-slider input::-webkit-slider-thumb { transition: none; } }
.ea-slider .ea-slider-value { font-variant-numeric: tabular-nums; font-weight: 600; color: var(--ct); min-width: 3.5rem; text-align: right; }
"""
    role = traitlets.Unicode("param").tag(sync=True)
    label = traitlets.Unicode("").tag(sync=True)
    min = traitlets.Float(0.0).tag(sync=True)
    max = traitlets.Float(1.0).tag(sync=True)
    step = traitlets.Float(0.1).tag(sync=True)
    value = traitlets.Float(0.0).tag(sync=True)
    decimal = traitlets.Unicode(".").tag(sync=True)
