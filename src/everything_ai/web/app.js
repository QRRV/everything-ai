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
    routeSub: "From your first line of Python to your own LLM and a paper you reproduce, in three levels. The route is the order we recommend, but you can open any lesson: if you miss something, the lesson tells you where to find it.",
    extras: "Show extra lessons", goals: "Or follow a goal", goalAll: "The whole route",
    goalCount: "{n} lessons instead of {m}: only what this goal builds on, in route order.",
    coming: "coming soon", extra: "extra", done: "done", started: "started", startOver: "Start over",
    confirmReset: "Start this lesson over? Your copy goes back to the version of the course. Your progress stays.",
    workingTitles: "", doneCount: "{d} of {t} done", ofBuilt: "{b} of {t} lessons ready",
    mapTitle: "The whole map",
    mapSub: "All of AI in 18 branches. Each branch opens here as soon as its first lessons are ready.",
    later: "coming soon", lessonsIn: "{n} lessons ready, the rest coming soon",
    lessonIn: "1 lesson ready, the rest coming soon", newVersion: "a new version of this lesson is ready: start over to get it",
    footer: "Text and figures CC BY-SA 4.0 · lesson code MIT-0 · app MIT", agents: "Learning with your own AI assistant? Point it at AGENTS.md in the course folder.",
    themes: { system: "System", light: "Light", dark: "Dark" }, themeLabel: "Theme", langLabel: "Language",
    allDone: "You finished every lesson that is ready. New lessons appear here as soon as they are built.",
    routes: "Your route", branchSub: "This route starts after the lesson “{a}” and builds on the lessons before it. You can open any lesson: if you miss something, the lesson tells you where to find it.",
    openRoute: "Its own route: {b} of {t} lessons ready", building: "being built",
    readyIn: "{n} lessons ready in “{r}”", oneReadyIn: "1 lesson ready in “{r}”",
    before: "First what this route builds on", placeIn: "Lesson {n} of {total} in “{r}”",
    branchSoon: "You did every lesson this route builds on. Its own first lessons are being built now; they appear here as soon as they are ready.",
  },
  nl: {
    atEnd: "Aan het eind:",
    hero: "Leer AI door het zelf te bouwen",
    sub: "Van je eerste regel Python tot je eigen language model en een paper die je nabouwt. Elke les draait op je eigen computer, en elke oefening wordt meteen nagekeken.",
    start: "Begin hier", cont: "Ga verder", place: "Les {n} van {total}", minutes: "ongeveer {n} minuten",
    computer: "Jouw computer", ram: "{n} GB geheugen", noGpu: "geen NVIDIA-videokaart gevonden",
    basicsAny: "Elke les in basis werkt op elke computer.",
    routeSub: "Van je eerste regel Python tot je eigen LLM en een paper die je nabouwt, in drie niveaus. De route is de volgorde die we aanraden, maar je kunt elke les openen: mis je iets, dan zegt de les waar je het vindt.",
    extras: "Toon verdieping", goals: "Of volg een doel", goalAll: "De hele route",
    goalCount: "{n} lessen in plaats van {m}: alleen waar dit doel op voortbouwt, in de volgorde van de route.",
    coming: "komt nog", extra: "verdieping", done: "af", started: "begonnen", startOver: "Opnieuw beginnen",
    confirmReset: "Deze les opnieuw beginnen? Je kopie gaat terug naar de versie van de course. Je voortgang blijft.",
    workingTitles: "Lessen die nog komen, staan er met hun Engelse werktitel.", doneCount: "{d} van {t} af", ofBuilt: "{b} van {t} lessen klaar",
    mapTitle: "De hele kaart",
    mapSub: "Alles van AI in 18 takken. Elke tak gaat hier open zodra zijn eerste lessen klaar zijn.",
    later: "komt nog", lessonsIn: "{n} lessen klaar, de rest komt nog",
    lessonIn: "1 les klaar, de rest komt nog", newVersion: "er is een nieuwe versie van deze les: begin opnieuw om die te krijgen",
    footer: "Tekst en figuren CC BY-SA 4.0 · lescode MIT-0 · app MIT", agents: "Leer je met je eigen AI-assistent? Wijs hem op AGENTS.md in de map van de course.",
    themes: { system: "Systeem", light: "Licht", dark: "Donker" }, themeLabel: "Thema", langLabel: "Taal",
    allDone: "Je hebt elke les gedaan die klaar is. Nieuwe lessen verschijnen hier zodra ze gebouwd zijn.",
    routes: "Jouw route", branchSub: "Deze route begint na de les “{a}” en bouwt voort op de lessen daarvoor. Je kunt elke les openen: mis je iets, dan zegt de les waar je het vindt.",
    openRoute: "Eigen route: {b} van {t} lessen klaar", building: "in aanbouw",
    readyIn: "{n} lessen klaar in “{r}”", oneReadyIn: "1 les klaar in “{r}”",
    before: "Eerst waar deze route op voortbouwt", placeIn: "Les {n} van {total} in “{r}”",
    branchSoon: "Je hebt elke les gedaan waar deze route op voortbouwt. De eerste eigen lessen worden nu gebouwd; ze verschijnen hier zodra ze klaar zijn.",
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

function routeName(full) {
  const r = full ? S.mainRoute.title : S.mainRoute.short;
  return r[lang()] || r.en;
}

function branchRoute() {
  return S.branchRoutes.find((r) => r.id === S.settings.route) || null;
}

function currentRoute() {
  const b = branchRoute();
  return b ? b.steps : S.route;
}

function shownSteps(route) {
  return route.filter((l) => S.settings.extras || !l.extra);
}

function steps() {
  return shownSteps(currentRoute());
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

// The next lesson to do, with the route it is numbered in. A branch route starts after a step of the first route:
// first come the core steps of the first route up to there that are not done yet, then the branch's own lessons.
function nextLesson() {
  const b = branchRoute();
  if (b) {
    const upTo = S.route.findIndex((l) => l.id === b.after);
    const first = S.route.slice(0, upTo + 1).find((l) => !l.extra && l.built && !l.done);
    if (first) return { lesson: first, route: shownSteps(S.route), before: true };
  }
  const own = steps().find((l) => l.built && !l.done);
  return own ? { lesson: own, route: steps(), before: false } : null;
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
  const next = nextLesson();
  const anyDone = currentRoute().some((l) => l.done || l.opened);
  const soon = branchRoute() && !currentRoute().some((l) => l.built);
  let card = `<p class="muted">${esc(t(soon ? "branchSoon" : "allDone"))}</p>`;
  if (next) {
    const nx = next.lesson;
    const v = { n: next.route.indexOf(nx) + 1, total: next.route.length, r: routeName(false) };
    const mod = S.modules[nx.module] ? S.modules[nx.module][lang()] : nx.module;
    card = `<a class="next" href="/course/open/${esc(nx.id)}?lang=${lang()}">
      <span class="kicker">${esc(next.before ? t("before") : anyDone ? t("cont") : t("start"))}</span>
      <span class="next-title">${esc(title(nx))}</span>
      <span class="meta">${esc(t(next.before ? "placeIn" : "place", v))} · ${esc(mod)} · ${esc(t("minutes", { n: nx.minutes || 15 }))}</span>
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
  const core = currentRoute().filter((l) => !l.extra);
  const cells = ["basis", "gevorderd", "expert"].filter((lvl) => core.some((l) => l.level === lvl)).map((lvl) => {
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

function afterTitle(branch) {
  const l = S.route.find((x) => x.id === branch.after);
  return l ? title(l) : branch.after;
}

function renderRouteChoice() {
  if (!S.branchRoutes.length) return "";
  const b = branchRoute();
  const chips = [`<button type="button" class="chip route-chip" data-route="${esc(S.mainRoute.id)}" aria-pressed="${!b}">${esc(routeName(false))}</button>`]
    .concat(S.branchRoutes.map((r) => `<button type="button" class="chip route-chip" data-route="${esc(r.id)}" aria-pressed="${b === r}">${esc(r.short[lang()])}</button>`));
  return `<div class="goals"><span class="goals-label">${esc(t("routes"))}</span><div class="chips">${chips.join("")}</div></div>`;
}

function renderRoute() {
  const n = numbers();
  const branch = branchRoute();
  if (branch) goal = null;   // the goal routes are shortcuts through the first route
  const inGoal = goal ? new Set(S.goals.find((g) => g.id === goal).lessons) : null;
  const shown = steps().filter((l) => !inGoal || inGoal.has(l.id));
  const nx = nextLesson()?.lesson;
  let html = "";
  for (const lvl of ["basis", "gevorderd", "expert"]) {
    const ofLevel = shown.filter((l) => l.level === lvl);
    if (!ofLevel.length) continue;
    const ms = branch ? null : S.milestones.find((m) => S.route.find((l) => l.id === m.after && l.level === lvl));
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
    <div class="route-head"><h2 class="section">${esc(branch ? branch.title[lang()] : routeName(true))}</h2>
      <label class="toggle"><input type="checkbox" id="extras" ${S.settings.extras ? "checked" : ""}> ${esc(t("extras"))}</label></div>
    <p class="lede small">${esc(branch ? t("branchSub", { a: afterTitle(branch) }) : t("routeSub"))}</p>
    ${renderRouteChoice()}${branch ? "" : renderGoals()}${wt}${html}</section>`;
}

function renderMap() {
  const built = {};   // built lessons of the first route, by their branch
  for (const l of S.route) if (l.built && l.tak) built[l.tak] = (built[l.tak] || 0) + 1;
  const cards = S.branches.slice().sort((a, b) => a.order - b.order || 0).map((b) => {
    const n = built[b.id] || 0;
    const own = S.branchRoutes.find((r) => r.tak === b.id);
    if (own) {
      const ready = own.steps.filter((l) => l.built).length;
      const st = n > 1 ? t("readyIn", { n, r: routeName(false) }) : n === 1 ? t("oneReadyIn", { r: routeName(false) }) : t("building");
      return `<li class="branch building"><span class="b-title">${esc(b.title[lang()])}</span>
        <span class="b-state">${esc(st)}</span>
        <button type="button" class="link route-chip" data-route="${esc(own.id)}">${esc(t("openRoute", { b: ready, t: own.steps.length }))}</button></li>`;
    }
    const st = n > 1 ? t("lessonsIn", { n }) : n === 1 ? t("lessonIn") : t("later");
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
  for (const c of app.querySelectorAll(".chip[data-goal]")) c.onclick = () => { goal = c.dataset.goal || null; render(); };
  for (const c of app.querySelectorAll(".route-chip")) {
    c.onclick = async () => {
      goal = null;
      await saveSettings({ route: c.dataset.route });
      document.getElementById("route").scrollIntoView({ behavior: "smooth", block: "start" });
    };
  }
  for (const b of app.querySelectorAll(".reset")) {
    b.onclick = async () => {
      if (!confirm(t("confirmReset"))) return;
      await post(`/course/api/reset/${encodeURIComponent(b.dataset.id)}`);
      load();
    };
  }
}

window.addEventListener("pageshow", load);   // also when you come back with the browser's back button
