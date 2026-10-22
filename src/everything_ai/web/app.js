// SPDX-FileCopyrightText: 2026 Everything AI contributors
// SPDX-License-Identifier: MIT
// The start page: where you are, the whole route, goals, and the map of all of AI.
// Plain JavaScript without a build step or anything from the internet; the course works offline.
"use strict";

const T = {
  en: {
    atEnd: "At the end:",
    hero: "Learn AI by building it yourself",
    sub: "From your first line of Python to your own language model and a paper you reproduce. Every lesson runs on your own computer, and every exercise is checked straight away.",
    start: "Start here", cont: "Continue", place: "Lesson {n} of {total}", minutes: "about {n} minutes",
    computer: "Your computer", ram: "{n} GB memory", noGpu: "no NVIDIA graphics card found",
    basicsAny: "Every lesson in the basics runs on any computer.",
    routeTitle: "The main route",
    routeSub: "One route from zero to your own LLM and a reproduced paper, in three levels. The route is the order we recommend, but you can open any lesson: if you miss something, the lesson tells you where to find it.",
    extras: "Show extra lessons", goals: "Or follow a goal", goalAll: "The whole route",
    goalCount: "{n} lessons instead of {m}: only what this goal builds on, in route order.",
    coming: "coming soon", extra: "extra", done: "done", started: "started", startOver: "Start over",
    confirmReset: "Start this lesson over? Your copy goes back to the version of the course. Your progress stays.",
    workingTitles: "", doneCount: "{d} of {t} done", ofBuilt: "{b} of {t} lessons ready",
    mapTitle: "The whole map",
    mapSub: "All of AI in 18 branches. The main route comes first; every other branch builds on it and opens here as soon as it is ready.",
    inRoute: "in the main route", later: "coming soon", lessonsIn: "{n} lessons in the main route, the rest coming soon",
    lessonIn: "1 lesson in the main route, the rest coming soon", newVersion: "a new version of this lesson is ready: start over to get it",
    footer: "Text and figures CC BY-SA 4.0 · lesson code MIT-0 · app MIT", agents: "Learning with your own AI assistant? Point it at AGENTS.md in the course folder.",
    themes: { system: "System", light: "Light", dark: "Dark" }, themeLabel: "Theme", langLabel: "Language",
    allDone: "You finished every lesson that is ready. New lessons appear here as soon as they are built.",
  },
  nl: {
    atEnd: "Aan het eind:",
    hero: "Leer AI door het zelf te bouwen",
    sub: "Van je eerste regel Python tot je eigen language model en een paper die je nabouwt. Elke les draait op je eigen computer, en elke oefening wordt meteen nagekeken.",
    start: "Begin hier", cont: "Ga verder", place: "Les {n} van {total}", minutes: "ongeveer {n} minuten",
    computer: "Jouw computer", ram: "{n} GB geheugen", noGpu: "geen NVIDIA-videokaart gevonden",
    basicsAny: "Elke les in basis werkt op elke computer.",
    routeTitle: "De hoofdroute",
    routeSub: "Eén route van nul tot je eigen LLM en een nagebouwde paper, in drie niveaus. De route is de volgorde die we aanraden, maar je kunt elke les openen: mis je iets, dan zegt de les waar je het vindt.",
    extras: "Toon verdieping", goals: "Of volg een doel", goalAll: "De hele route",
    goalCount: "{n} lessen in plaats van {m}: alleen waar dit doel op voortbouwt, in de volgorde van de route.",
    coming: "komt nog", extra: "verdieping", done: "af", started: "begonnen", startOver: "Opnieuw beginnen",
    confirmReset: "Deze les opnieuw beginnen? Je kopie gaat terug naar de versie van de course. Je voortgang blijft.",
    workingTitles: "Lessen die nog komen, staan er met hun Engelse werktitel.", doneCount: "{d} van {t} af", ofBuilt: "{b} van {t} lessen klaar",
    mapTitle: "De hele kaart",
    mapSub: "Alles van AI in 18 takken. De hoofdroute komt eerst; elke andere tak bouwt erop voort en gaat hier open zodra hij klaar is.",
    inRoute: "in de hoofdroute", later: "komt nog", lessonsIn: "{n} lessen in de hoofdroute, de rest komt nog",
    lessonIn: "1 les in de hoofdroute, de rest komt nog", newVersion: "er is een nieuwe versie van deze les: begin opnieuw om die te krijgen",
    footer: "Tekst en figuren CC BY-SA 4.0 · lescode MIT-0 · app MIT", agents: "Leer je met je eigen AI-assistent? Wijs hem op AGENTS.md in de map van de course.",
    themes: { system: "Systeem", light: "Licht", dark: "Donker" }, themeLabel: "Thema", langLabel: "Taal",
    allDone: "Je hebt elke les gedaan die klaar is. Nieuwe lessen verschijnen hier zodra ze gebouwd zijn.",
  },
};

let S = null;            // the state from /course/api/state
let goal = null;         // the id of the chosen goal route, or null for the whole route

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fill = (s, v) => s.replace(/\{(\w+)\}/g, (_, k) => v[k]);
const lang = () => S.settings.lang;
const t = (k, v) => (v ? fill(T[lang()][k], v) : T[lang()][k]);

const ICON = {
  done: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7" fill="currentColor"/><path d="M4.8 8.3 7 10.4 11.3 5.8" fill="none" stroke="var(--ea-page)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  started: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.2" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M8 1.8 A6.2 6.2 0 0 1 8 14.2 Z" fill="currentColor"/></svg>',
  open: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.2" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>',
  coming: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.2" fill="none" stroke="currentColor" stroke-width="1.3" stroke-dasharray="2 2.4"/></svg>',
};

function applyTheme(theme) {
  try { localStorage.setItem("ea-theme", theme); } catch (e) {}
  const dark = theme === "dark" || (theme === "system" && matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.classList.toggle("dark", dark);
}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => S && applyTheme(S.settings.theme));

async function load() {
  const r = await fetch("/course/api/state", { cache: "no-store" });
  S = await r.json();
  applyTheme(S.settings.theme);
  render();
}

// marimo accepts a POST only with its server token, which the page carries in a meta tag
function post(url, body) {
  const token = document.querySelector('meta[name="marimo-server-token"]')?.content || "";
  const headers = { "Marimo-Server-Token": token };
  if (body !== undefined) headers["content-type"] = "application/json";
  return fetch(url, { method: "POST", headers, body: body === undefined ? undefined : JSON.stringify(body) });
}

async function saveSettings(changes) {
  const r = await post("/course/api/settings", changes);
  S.settings = await r.json();
  applyTheme(S.settings.theme);
  render();
}

function steps() {
  return S.route.filter((l) => S.settings.extras || !l.extra);
}

function numbers() {
  const n = {}; steps().forEach((l, i) => { n[l.id] = i + 1; }); return n;
}

function title(l) {
  return l.title[lang()] || l.title.en;
}

function status(l) {
  if (!l.built) return "coming";
  if (l.done) return "done";
  if (l.opened) return "started";
  return "open";
}

function nextLesson() {
  const route = steps();
  return route.find((l) => l.built && !l.done) || null;
}

function renderSwitches() {
  document.documentElement.lang = lang();
  const ls = document.getElementById("lang-switch");
  ls.setAttribute("aria-label", t("langLabel"));
  for (const b of ls.querySelectorAll("button")) {
    b.setAttribute("aria-pressed", String(b.dataset.lang === lang()));
    b.onclick = () => saveSettings({ lang: b.dataset.lang });
  }
  const ts = document.getElementById("theme-switch");
  ts.setAttribute("aria-label", t("themeLabel"));
  for (const b of ts.querySelectorAll("button")) {
    b.textContent = t("themes")[b.dataset.theme];
    b.setAttribute("aria-pressed", String(b.dataset.theme === S.settings.theme));
    b.onclick = () => saveSettings({ theme: b.dataset.theme });
  }
}

function renderHero() {
  const nx = nextLesson(); const n = numbers(); const total = steps().length;
  const anyDone = S.route.some((l) => l.done || l.opened);
  let card = `<p class="muted">${esc(t("allDone"))}</p>`;
  if (nx) {
    const mod = S.modules[nx.module] ? S.modules[nx.module][lang()] : nx.module;
    card = `<a class="next" href="/course/open/${esc(nx.id)}?lang=${lang()}">
      <span class="kicker">${esc(anyDone ? t("cont") : t("start"))}</span>
      <span class="next-title">${esc(title(nx))}</span>
      <span class="meta">${esc(t("place", { n: n[nx.id], total }))} · ${esc(mod)} · ${esc(t("minutes", { n: nx.minutes || 15 }))}</span>
      <span class="arrow" aria-hidden="true">→</span></a>`;
  }
  const sys = S.system || {};
  const parts = [sys.os, sys.ram_gb ? t("ram", { n: sys.ram_gb }) : null,
    sys.gpu ? (sys.gpu.name + (sys.gpu.vram_gb ? ` (${sys.gpu.vram_gb} GB)` : "")) : t("noGpu")].filter(Boolean);
  return `<section class="hero">
    <h1>${esc(t("hero"))}</h1>
    <p class="lede">${esc(t("sub"))}</p>
    ${card}
    <p class="system"><b>${esc(t("computer"))}:</b> ${esc(parts.join(" · "))}. ${esc(t("basicsAny"))}</p>
  </section>`;
}

function renderLevels() {
  const core = S.route.filter((l) => !l.extra);
  const cells = ["basis", "gevorderd", "expert"].map((lvl) => {
    const ls = core.filter((l) => l.level === lvl);
    const d = ls.filter((l) => l.done).length;
    const b = ls.filter((l) => l.built).length;
    const pct = ls.length ? (100 * d) / ls.length : 0;
    return `<div class="level">
      <div class="level-name">${esc(S.levels[lvl][lang()])}</div>
      <div class="bar" role="img" aria-label="${esc(t("doneCount", { d, t: ls.length }))}"><span style="width:${pct}%"></span></div>
      <div class="level-meta">${esc(t("doneCount", { d, t: ls.length }))} · ${esc(t("ofBuilt", { b, t: ls.length }))}</div>
    </div>`;
  });
  return `<section class="levels">${cells.join("")}</section>`;
}

function renderGoals() {
  const total = steps().length;
  const chips = [`<button type="button" class="chip" data-goal="" aria-pressed="${goal === null}">${esc(t("goalAll"))}</button>`]
    .concat(S.goals.map((g) => `<button type="button" class="chip" data-goal="${esc(g.id)}" aria-pressed="${goal === g.id}">${esc(g.title[lang()])}</button>`));
  let note = "";
  if (goal) {
    const g = S.goals.find((x) => x.id === goal);
    note = `<p class="goal-note">${esc(t("goalCount", { n: g.lessons.length, m: (numbers()[g.lesson] || total) }))}</p>`;
  }
  return `<div class="goals"><span class="goals-label">${esc(t("goals"))}</span><div class="chips">${chips.join("")}</div>${note}</div>`;
}

function renderRoute() {
  const n = numbers();
  const inGoal = goal ? new Set(S.goals.find((g) => g.id === goal).lessons) : null;
  const shown = steps().filter((l) => !inGoal || inGoal.has(l.id));
  const nx = nextLesson();
  let html = "";
  for (const lvl of ["basis", "gevorderd", "expert"]) {
    const ofLevel = shown.filter((l) => l.level === lvl);
    if (!ofLevel.length) continue;
    const ms = S.milestones.find((m) => S.route.find((l) => l.id === m.after && l.level === lvl));
    html += `<section class="lvl"><h2>${esc(S.levels[lvl][lang()])}</h2>${ms ? `<p class="milestone">${esc(T[lang()].atEnd)} ${esc(ms[lang()].replace(/^[^:]*:\s*/, ""))}</p>` : ""}`;
    const modules = [];
    for (const l of ofLevel) {
      const last = modules[modules.length - 1];
      if (last && last.id === l.module) last.lessons.push(l); else modules.push({ id: l.module, lessons: [l] });
    }
    for (const m of modules) {
      const d = m.lessons.filter((l) => l.done).length;
      const hasBuilt = m.lessons.some((l) => l.built);
      const isOpen = hasBuilt || (nx && m.lessons.includes(nx)) || !!goal;
      const name = S.modules[m.id] ? S.modules[m.id][lang()] : m.id;
      html += `<details class="module" ${isOpen ? "open" : ""}><summary><span class="mod-name">${esc(name)}</span>
        <span class="mod-meta">${esc(t("doneCount", { d, t: m.lessons.length }))}</span></summary><ol class="lessons">`;
      for (const l of m.lessons) {
        const st = status(l);
        const label = { done: t("done"), started: t("started"), open: "", coming: t("coming") }[st];
        const extra = l.extra ? `<span class="tag">${esc(t("extra"))}</span>` : "";
        const name = l.built ? `<a href="/course/open/${esc(l.id)}?lang=${lang()}">${esc(title(l))}</a>` : `<span class="wip">${esc(title(l))}</span>`;
        const update = l.copy && Object.values(l.copy).includes("update") ? `<span class="tag">${esc(t("newVersion"))}</span>` : "";
        const reset = l.built && l.opened ? `<button type="button" class="link reset" data-id="${esc(l.id)}">${esc(t("startOver"))}</button>` : "";
        html += `<li class="lesson st-${st}"><span class="num">${n[l.id] || ""}</span><span class="ico" title="${esc(label)}">${ICON[st]}</span>
          <span class="name">${name}${extra}${update}</span><span class="state">${esc(label)}</span>${reset}</li>`;
      }
      html += `</ol></details>`;
    }
    html += `</section>`;
  }
  const wt = t("workingTitles") ? `<p class="muted small">${esc(t("workingTitles"))}</p>` : "";
  return `<section class="route" id="route">
    <div class="route-head"><h2 class="section">${esc(t("routeTitle"))}</h2>
      <label class="toggle"><input type="checkbox" id="extras" ${S.settings.extras ? "checked" : ""}> ${esc(t("extras"))}</label></div>
    <p class="lede small">${esc(t("routeSub"))}</p>
    ${renderGoals()}${wt}${html}</section>`;
}

function renderMap() {
  const built = {};   // lessons of a later branch that the main route already has, by their branch
  for (const l of S.route) if (l.built && l.tak) built[l.tak] = (built[l.tak] || 0) + 1;
  const cards = S.branches.slice().sort((a, b) => a.order - b.order || 0).map((b) => {
    const n = built[b.id] || 0;
    const st = b.status === "main-route" ? t("inRoute") : n > 1 ? t("lessonsIn", { n }) : n === 1 ? t("lessonIn") : t("later");
    return `<li class="branch ${b.status === "main-route" ? "now" : "later"}"><span class="b-title">${esc(b.title[lang()])}</span>
      <span class="b-state">${esc(st)}</span></li>`;
  });
  return `<section class="map"><h2 class="section">${esc(t("mapTitle"))}</h2><p class="lede small">${esc(t("mapSub"))}</p>
    <ul class="branches">${cards.join("")}</ul></section>`;
}

function render() {
  renderSwitches();
  const app = document.getElementById("app");
  app.innerHTML = renderHero() + renderLevels() + renderRoute() + renderMap() +
    `<footer class="foot"><p>${esc(t("agents"))}</p><p class="muted">${esc(t("footer"))}</p></footer>`;
  app.setAttribute("aria-busy", "false");
  document.getElementById("extras").onchange = (e) => saveSettings({ extras: e.target.checked });
  for (const c of app.querySelectorAll(".chip")) c.onclick = () => { goal = c.dataset.goal || null; render(); };
  for (const b of app.querySelectorAll(".reset")) {
    b.onclick = async () => {
      if (!confirm(t("confirmReset"))) return;
      await post(`/course/api/reset/${encodeURIComponent(b.dataset.id)}`);
      load();
    };
  }
}

window.addEventListener("pageshow", load);   // also when you come back with the browser's back button
