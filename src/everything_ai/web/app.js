// SPDX-FileCopyrightText: 2026 Everything AI contributors
// SPDX-License-Identifier: MIT
// The start page: your way through the course, with the moments where you choose. First the kern, as rows of dots,
// one dot per lesson; then the crossroads where you choose a domain, the levels of that domain, and what a second
// domain shares with it. A goal is a shorter way: it pales the lessons it does not need.
// Plain JavaScript without a build step or anything from the internet; the course works offline.
"use strict";

const T = {
  en: {
    hero: "Learn AI by building it yourself",
    sub: "Start with the core: Python, data, your first models and neural networks. Then choose a domain, such as large language models or computer vision. Every lesson runs on your own computer, and every exercise is checked straight away.",
    start: "Start here", cont: "Continue", place: "Lesson {n} of {total}", minutes: "about {n} minutes", tasteOf: "a taste of {d}",
    computer: "Your computer", ram: "{n} GB memory", noGpu: "no NVIDIA graphics card found",
    basicsAny: "Every lesson in the basics runs on any computer.",
    before: "First what this route builds on", placeIn: "Lesson {n} of {total} in “{r}”",
    allDone: "You finished every lesson that is ready. New lessons appear here as soon as they are built.",
    branchSoon: "You did every lesson this route builds on. Its own first lessons are being built now; they appear here as soon as they are ready.",
    way: "Your way", waySub: "Where you are, what comes next, and where you choose. You can open any lesson: if you miss something, the lesson tells you where to find it.",
    wayCount: "{d} of {t} lessons of the core done", wayBranch: "{d} of {t} of “{r}”",
    extras: "Show extra lessons", extra: "extra",
    lgDone: "lesson done", lgReady: "ready to do", lgSoon: "coming soon", lgTaste: "a taste of a domain", lgChoice: "a choice",
    here: "you are here", tastes: "a taste of {d}", doneCount: "{d} of {t} done",
    shortcut: "A shorter way?", shortcutSub: "With a goal you do only the lessons it builds on, even in the middle of the core. The lessons it does not need fade.",
    goalChip: "{n} lessons instead of {m}", wholeWay: "The whole way",
    goalCount: "Only what this goal builds on, in the order of the course. The faded lessons still open.",
    cross: "Crossroads: choose your domain",
    crossSub: "{n} domains, all equally important. {k} have a route of their own; the others are coming. You can choose now: you then do the rest of the core first. Later you can choose more.",
    crossTasted: "You already had a taste of {k}.", mine: "your choice", tasted: "tasted",
    stReady: "ready · {b} lessons", stPart: "{b} of {t} lessons ready", stElsewhere: "first lessons in “{r}”", stSoon: "coming soon",
    sharedDone: "{d} of {t} done already", onlyKern: "No domain for now, only the core",
    levels: "After each level: deeper or wider",
    levelsSub: "Every domain has three levels. After each level you choose: the next level, or a second domain first. For example {d}:",
    lvLessons: "{n} lessons", lvSoon: "coming soon",
    afterLevel: "End of the {l} level: deeper or wider?",
    afterLevelSub: "Go on with {next}, or choose a second domain first at the crossroads; what you did here counts there too.",
    second: "A second domain: what you did counts",
    secondSub: "Domains share lessons. Choose another domain after one, and you skip what is already done:",
    pair: "After {a}, {s} of the {t} lessons of {b} are done",
    coming: "coming soon", done: "done", started: "started", startOver: "Start over",
    confirmReset: "Start this lesson over? Your copy goes back to the version of the course. Your progress stays.",
    workingTitles: "", newVersion: "a new version of this lesson is ready: start over to get it",
    footer: "Text and figures CC BY-SA 4.0 · lesson code MIT-0 · app MIT", agents: "Learning with your own AI assistant? Point it at AGENTS.md in the course folder.",
    themes: { system: "System", light: "Light", dark: "Dark" }, themeLabel: "Theme", langLabel: "Language",
  },
  nl: {
    hero: "Leer AI door het zelf te bouwen",
    sub: "Begin met de kern: Python, data, je eerste modellen en neural networks. Kies daarna een domein, zoals large language models of computer vision. Elke les draait op je eigen computer, en elke oefening wordt meteen nagekeken.",
    start: "Begin hier", cont: "Ga verder", place: "Les {n} van {total}", minutes: "ongeveer {n} minuten", tasteOf: "voorproefje van {d}",
    computer: "Jouw computer", ram: "{n} GB geheugen", noGpu: "geen NVIDIA-videokaart gevonden",
    basicsAny: "Elke basisles werkt op elke computer.",
    before: "Eerst waar deze route op voortbouwt", placeIn: "Les {n} van {total} in “{r}”",
    allDone: "Je hebt elke les gedaan die klaar is. Nieuwe lessen verschijnen hier zodra ze gebouwd zijn.",
    branchSoon: "Je hebt elke les gedaan waar deze route op voortbouwt. De eerste eigen lessen worden nu gebouwd; ze verschijnen hier zodra ze klaar zijn.",
    way: "Jouw weg", waySub: "Waar je bent, wat er komt, en waar je kiest. Je kunt altijd elke les openen: mis je iets, dan zegt de les waar je het vindt.",
    wayCount: "{d} van {t} lessen van de kern af", wayBranch: "{d} van {t} van “{r}”",
    extras: "Toon verdieping", extra: "verdieping",
    lgDone: "les af", lgReady: "klaar om te doen", lgSoon: "komt nog", lgTaste: "voorproefje van een domein", lgChoice: "keuzemoment",
    here: "je bent hier", tastes: "voorproefje van {d}", doneCount: "{d} van {t} af",
    shortcut: "Een kortere weg?", shortcutSub: "Met een doel doe je alleen de lessen waar het op voortbouwt, ook midden in de kern. De lessen die het niet nodig heeft, vervagen.",
    goalChip: "{n} lessen in plaats van {m}", wholeWay: "De hele weg",
    goalCount: "Alleen waar dit doel op voortbouwt, in de volgorde van de course. De vervaagde lessen gaan ook gewoon open.",
    cross: "Kruispunt: kies je domein",
    crossSub: "{n} domeinen, allemaal even belangrijk. {k} hebben al een eigen route; de andere komen nog. Je mag nu al kiezen: dan doe je eerst de rest van de kern. Later kun je er meer kiezen.",
    crossTasted: "Je hebt er al {k} geproefd.", mine: "jouw keuze", tasted: "geproefd",
    stReady: "klaar · {b} lessen", stPart: "{b} van {t} lessen klaar", stElsewhere: "eerste lessen in “{r}”", stSoon: "komt nog",
    sharedDone: "{d} van {t} al gedaan", onlyKern: "Nog geen domein, alleen de kern",
    levels: "Na elk niveau: dieper of breder",
    levelsSub: "Elk domein heeft drie niveaus. Na elk niveau kies je: het volgende niveau, of eerst een tweede domein. Bijvoorbeeld {d}:",
    lvLessons: "{n} lessen", lvSoon: "komt nog",
    afterLevel: "Einde van niveau {l}: dieper of breder?",
    afterLevelSub: "Ga door met {next}, of kies eerst een tweede domein op het kruispunt; wat je hier deed, telt daar ook mee.",
    second: "Een tweede domein: wat je deed, telt mee",
    secondSub: "Domeinen delen lessen. Wie na het ene domein het andere kiest, slaat over wat al af is:",
    pair: "Na {a} heb je al {s} van de {t} lessen van {b}",
    coming: "komt nog", done: "af", started: "begonnen", startOver: "Opnieuw beginnen",
    confirmReset: "Deze les opnieuw beginnen? Je kopie gaat terug naar de versie van de course. Je voortgang blijft.",
    workingTitles: "Lessen die nog komen, staan er met hun Engelse werktitel.", newVersion: "er is een nieuwe versie van deze les: begin opnieuw om die te krijgen",
    footer: "Tekst en figuren CC BY-SA 4.0 · lescode MIT-0 · app MIT", agents: "Leer je met je eigen AI-assistent? Wijs hem op AGENTS.md in de map van de course.",
    themes: { system: "Systeem", light: "Licht", dark: "Donker" }, themeLabel: "Thema", langLabel: "Taal",
  },
};

let S = null;            // the state from /course/api/state
let goal = null;         // the id of the chosen goal, or null for the whole way

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fill = (s, v) => s.replace(/\{(\w+)\}/g, (_, k) => v[k]);
const lang = () => S.settings.lang;
const t = (k, v) => (v ? fill(T[lang()][k], v) : T[lang()][k]);
const LEVELS = ["basis", "gevorderd", "expert"];

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

// --- the data the page draws from -----------------------------------------------------------------------------------

const name = (o) => o[lang()] || o.en;
const title = (l) => l.title[lang()] || l.title.en;
const partName = (d) => name(S.parts[d].name);
const moduleName = (key) => (S.modules[key] ? name(S.modules[key]) : key.split("/").pop());

// the route of the domain the learner chose, or null for the kern alone
function branchRoute() {
  return S.domainRoutes.find((r) => r.id === S.settings.route) || null;
}

function routeName(r) {
  return r ? name(r.short) : name(S.kern.short);
}

function shownSteps(route) {
  return route.filter((l) => S.settings.extras || !l.extra);
}

function chosenGoal() {
  return goal ? S.goals.find((g) => g.id === goal) || null : null;
}

// the domains a lesson of the kern gives a taste of
function tasteIndex() {
  const ix = {};
  for (const [d, ids] of Object.entries(S.tasters)) for (const id of ids) (ix[id] = ix[id] || []).push(d);
  return ix;
}

function status(l) {
  if (!l.built) return "coming";
  if (l.done) return "done";
  if (l.opened) return "started";
  return "open";
}

// The next lesson to do, with the route it is numbered in. The route of a domain starts after the kern: first come
// the core steps of the kern that are not done yet, then the domain's own route. With a goal, only what it needs.
function nextLesson() {
  const g = chosenGoal();
  const needed = (l) => !g || g.lessons.includes(l.id);
  const b = branchRoute();
  if (b) {
    const first = S.route.find((l) => !l.extra && l.built && !l.done && needed(l));
    if (first) return { lesson: first, route: shownSteps(S.route), before: true };
  }
  const steps = shownSteps(b ? b.steps : S.route);
  const own = steps.find((l) => l.built && !l.done && needed(l));
  return own ? { lesson: own, route: steps, before: false } : null;
}

// consecutive lessons of one module, in the order of the route
function modulesOf(steps) {
  const out = [];
  for (const l of steps) {
    const last = out[out.length - 1];
    if (last && last.key === l.module) last.lessons.push(l); else out.push({ key: l.module, lessons: [l] });
  }
  return out;
}

// how many lessons of route `b` are core steps of route `a` too: what a second domain shares with the first
function shared(a, b) {
  const ids = new Set(a.steps.filter((l) => !l.extra).map((l) => l.id));
  const theirs = b.steps.filter((l) => !l.extra);
  return { s: theirs.filter((l) => ids.has(l.id)).length, t: theirs.length };
}

// --- drawing ---------------------------------------------------------------------------------------------------------

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
    b.textContent = t("themes")[b.dataset.choice];
    b.setAttribute("aria-pressed", String(b.dataset.choice === S.settings.theme));
    b.onclick = () => saveSettings({ theme: b.dataset.choice });
  }
}

function picture(pid) {
  const p = S.pictures[pid];
  if (!p) return "";
  return p.svg.replace("<svg ", `<svg role="img" aria-label="${esc(name(p.about))}" `);
}

function renderHero() {
  const next = nextLesson();
  const b = branchRoute();
  const anyDone = (b ? b.steps : S.route).some((l) => l.done || l.opened);
  const soon = b && !b.steps.some((l) => l.built);
  let card = `<p class="muted">${esc(t(soon ? "branchSoon" : "allDone"))}</p>`;
  if (next) {
    const nx = next.lesson;
    const v = { n: next.route.indexOf(nx) + 1, total: next.route.length, r: routeName(null) };
    const tastes = (tasteIndex()[nx.id] || []).map(partName);
    const meta = [t(next.before ? "placeIn" : "place", v), moduleName(nx.module), t("minutes", { n: nx.minutes || 15 })]
      .concat(tastes.length ? [t("tasteOf", { d: tastes.join(", ") })] : []);
    card = `<a class="next" href="/course/open/${esc(nx.id)}?lang=${lang()}">
      <span class="kicker">${esc(next.before ? t("before") : anyDone ? t("cont") : t("start"))}</span>
      <span class="next-title">${esc(title(nx))}</span>
      <span class="meta">${meta.map(esc).join(" · ")}</span>
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

// one dot per lesson: filled is done, a ring is ready, a dashed ring is coming; a square gives a taste of a domain
function dot(l, num, ctx) {
  const cls = ["d", l.done ? "done" : !l.built ? "soon" : l.opened ? "started" : "ready"];
  if (ctx.tastes[l.id] && ctx.kern) cls.push("sq");
  if (ctx.next && ctx.next.id === l.id) cls.push("here");
  if (ctx.keep && !ctx.keep.has(l.id)) cls.push("skip");
  const tip = esc(`${num}. ${title(l)}`);
  return l.built
    ? `<a class="${cls.join(" ")}" href="/course/open/${esc(l.id)}?lang=${lang()}" title="${tip}" tabindex="-1"></a>`
    : `<span class="${cls.join(" ")}" title="${tip}"></span>`;
}

function lessonRow(l, num, ctx) {
  const st = status(l);
  const label = { done: t("done"), started: t("started"), open: "", coming: t("coming") }[st];
  const extra = l.extra ? `<span class="tag">${esc(t("extra"))}</span>` : "";
  const tastes = ctx.kern && ctx.tastes[l.id] ? `<span class="tag taste">${esc(t("tastes", { d: ctx.tastes[l.id].map(partName).join(", ") }))}</span>` : "";
  const nm = l.built ? `<a href="/course/open/${esc(l.id)}?lang=${lang()}">${esc(title(l))}</a>` : `<span class="wip">${esc(title(l))}</span>`;
  const update = l.copy && Object.values(l.copy).includes("update") ? `<span class="tag">${esc(t("newVersion"))}</span>` : "";
  const reset = l.built && l.opened ? `<button type="button" class="link reset" data-id="${esc(l.id)}">${esc(t("startOver"))}</button>` : "";
  const skip = ctx.keep && !ctx.keep.has(l.id) ? " skip" : "";
  return `<li class="lesson st-${st}${skip}"><span class="num">${num}</span><span class="ico" title="${esc(label)}">${ICON[st]}</span>
    <span class="name">${nm}${extra}${tastes}${update}</span><span class="state">${esc(label)}</span>${reset}</li>`;
}

// a module on the path: its name, a row of dots and the count; open it for the lessons by title
function moduleStep(m, ctx) {
  const core = m.lessons.filter((l) => !l.extra);
  const d = core.filter((l) => l.done).length;
  const here = ctx.next && m.lessons.some((l) => l.id === ctx.next.id);
  const st = core.length && d === core.length ? "done" : here ? "now" : m.lessons.some((l) => l.built) ? "todo" : "soon";
  const dots = m.lessons.map((l) => dot(l, ctx.num[l.id], ctx)).join("");
  const rows = m.lessons.map((l) => lessonRow(l, ctx.num[l.id], ctx)).join("");
  const tastes = ctx.kern ? [...new Set(m.lessons.flatMap((l) => ctx.tastes[l.id] || []))].map(partName) : [];
  const badge = here ? ` <span class="here-badge">${esc(t("here"))}</span>` : "";
  const count = t("doneCount", { d, t: core.length });
  return `<li class="step st-${st}"><span class="mark" aria-hidden="true"></span>
    <details class="mod"><summary>
      <span class="mod-name">${esc(moduleName(m.key))}${badge}</span>
      <span class="dots" aria-hidden="true">${dots}</span>
      <span class="mod-count" aria-label="${esc(count)}">${d}/${core.length}</span></summary>
      <ol class="lessons">${rows}</ol></details>
    ${tastes.length ? `<p class="taste-line"><span class="d sq" aria-hidden="true"></span>${esc(t("tastes", { d: tastes.join(", ") }))}</p>` : ""}</li>`;
}

function choiceStep(cls, inner) {
  return `<li class="step choice ${cls}"><span class="mark" aria-hidden="true"></span><div class="choice-box">${inner}</div></li>`;
}

// the whole way to a goal: the core of the kern first when the goal sits in the route of a domain
function wholeWay(g) {
  const r = S.domainRoutes.find((x) => x.id === g.route);
  const steps = shownSteps(r ? r.steps : S.route);
  const at = steps.findIndex((l) => l.id === g.lesson) + 1 || steps.length;
  return (r ? shownSteps(S.route).length : 0) + at;
}

function goalsChoice() {
  if (!S.goals.length) return "";
  const g = chosenGoal();
  const chips = S.goals.map((x) => `<button type="button" class="chip" data-goal="${esc(x.id)}" aria-pressed="${goal === x.id}">
      ${esc(name(x.title))} · <b>${esc(t("goalChip", { n: x.lessons.length, m: wholeWay(x) }))}</b></button>`);
  if (g) chips.push(`<button type="button" class="chip" data-goal="">${esc(t("wholeWay"))}</button>`);
  const note = g ? `<p class="goal-note">${esc(t("goalCount", { n: g.lessons.length, m: wholeWay(g) }))}</p>` : "";
  return choiceStep("shortcut", `<h3>${esc(t("shortcut"))}</h3><p>${esc(t("shortcutSub"))}</p><div class="chips">${chips.join("")}</div>${note}`);
}

function tile(d, ctx) {
  const own = S.domainRoutes.find((r) => r.domein === d);
  const p = S.parts[d];
  let state = t("stSoon");
  let doneTag = "";
  if (own) {
    const core = own.steps.filter((l) => !l.extra);
    const b = core.filter((l) => l.built).length;
    state = b === core.length ? t("stReady", { b }) : t("stPart", { b, t: core.length });
    const dn = core.filter((l) => l.done).length;
    if (dn) doneTag = `<span class="tag">${esc(t("sharedDone", { d: dn, t: core.length }))}</span>`;
  } else if (p.built) {
    const where = S.domainRoutes.find((r) => r.steps.some((l) => l.part === d && l.built));
    state = t("stElsewhere", { r: where ? routeName(where) : partName("kern") });
  }
  const mine = own && own === ctx.branch;
  const tags = (mine ? `<span class="tag mine">${esc(t("mine"))}</span>` : "") +
    ((S.tasters[d] || []).some((id) => ctx.doneIds.has(id)) ? `<span class="tag">${esc(t("tasted"))}</span>` : "") + doneTag;
  const inner = `<span class="picbox">${picture(d)}</span><span class="t-name">${esc(partName(d))}</span>
    <span class="t-about">${esc(name(S.pictures[d].about))}</span><span class="t-state">${esc(state)}</span>
    ${tags ? `<span class="t-tags">${tags}</span>` : ""}`;
  const kind = own ? (mine ? "mine" : "open") : p.built ? "elsewhere" : "soon";
  return own
    ? `<button type="button" class="tile ${kind}" data-route="${esc(own.id)}" aria-pressed="${mine}">${inner}</button>`
    : `<div class="tile ${kind}">${inner}</div>`;
}

function crossroads(ctx) {
  const n = S.groups.reduce((a, g) => a + g.domains.length, 0);
  const tasted = Object.entries(S.tasters).filter(([, ids]) => ids.some((id) => ctx.doneIds.has(id))).length;
  const groups = S.groups.map((g) => `<h4 class="group">${esc(name(g.name))}</h4><div class="tiles">${g.domains.map((d) => tile(d, ctx)).join("")}</div>`);
  const back = ctx.branch ? `<button type="button" class="link route-chip" data-route="${esc(S.kern.id)}">${esc(t("onlyKern"))}</button>` : "";
  return choiceStep("cross", `<h3>${esc(t("cross"))}</h3>
    <p>${esc(t("crossSub", { n, k: S.domainRoutes.length }))}${tasted ? " " + esc(t("crossTasted", { k: tasted })) : ""}</p>
    ${groups.join("")}${back}`);
}

// the levels of one domain as chips: basis 21 lessons ◇ gevorderd 63 lessons ◇ expert coming soon
function levelChips(r) {
  return LEVELS.map((lvl) => {
    const n = r.steps.filter((l) => !l.extra && l.level === lvl && l.built).length;
    return `<span class="lv${n ? "" : " soon"}"><b>${esc(S.levels[lvl][lang()])}</b> ${esc(n ? t("lvLessons", { n }) : t("lvSoon"))}</span>`;
  }).join(`<span class="choice-mini" aria-hidden="true"></span>`);
}

function levelsExample(ctx) {
  // a domain you tasted first; then the one whose levels show best what a level is: most levels with ten lessons ready
  const tasted = (r) => ((S.tasters[r.domein] || []).some((id) => ctx.doneIds.has(id)) ? 1 : 0);
  const ready = (r, lvl) => r.steps.filter((l) => !l.extra && l.level === lvl && l.built).length;
  const full = (r) => LEVELS.filter((lvl) => ready(r, lvl) >= 10).length;
  const all = (r) => LEVELS.reduce((a, lvl) => a + ready(r, lvl), 0);
  const r = [...S.domainRoutes].sort((a, b) => tasted(b) - tasted(a) || full(b) - full(a) || all(b) - all(a))[0];
  if (!r) return "";
  return choiceStep("levels", `<h3>${esc(t("levels"))}</h3><p>${esc(t("levelsSub", { d: routeName(r) }))}</p><div class="lvline">${levelChips(r)}</div>`);
}

function secondDomain(ctx) {
  const others = (a) => S.domainRoutes.filter((b) => b !== a).map((b) => ({ a, b, ...shared(a, b) })).filter((x) => x.s > 0);
  let pairs;
  if (ctx.branch) pairs = others(ctx.branch).sort((x, y) => y.s - x.s).slice(0, 3);
  else {
    // the pairs that share most, each pair once (what A shares with B, B shares with A)
    const all = S.domainRoutes.flatMap(others).sort((x, y) => y.s - x.s);
    pairs = all.filter((x, i) => !all.slice(0, i).some((y) => y.a === x.b && y.b === x.a)).slice(0, 3);
  }
  if (!pairs.length) return "";
  const chips = pairs.map((x) => `<span class="chip static">${esc(t("pair", { a: routeName(x.a), b: routeName(x.b), s: x.s, t: x.t }))}</span>`);
  return choiceStep("second", `<h3>${esc(t("second"))}</h3><p>${esc(t("secondSub"))}</p><div class="chips">${chips.join("")}</div>`);
}

function renderWay() {
  const branch = branchRoute();
  const g = chosenGoal();
  const nx = nextLesson()?.lesson || null;
  const all = S.route.concat(...S.domainRoutes.map((r) => r.steps));
  const ctx = { branch, next: nx, goal: !!g, keep: g ? new Set(g.lessons) : null, tastes: {}, doneIds: new Set(all.filter((l) => l.done).map((l) => l.id)) };
  for (const [d, ids] of Object.entries(S.tasters)) for (const id of ids) (ctx.tastes[id] = ctx.tastes[id] || []).push(d);
  const kernSteps = shownSteps(S.route);
  const kernCore = S.route.filter((l) => !l.extra);
  const kctx = { ...ctx, kern: true, num: Object.fromEntries(kernSteps.map((l, i) => [l.id, i + 1])) };
  const items = [];
  const mods = modulesOf(kernSteps);
  const hereAt = mods.findIndex((m) => nx && m.lessons.some((l) => l.id === nx.id));
  items.push(`<li class="step part"><span class="mark" aria-hidden="true"></span><div class="part-head">
    <span class="picbox small">${picture("kern")}</span><div><h3>${esc(partName("kern"))}</h3>
    <p class="muted">${esc(t("wayCount", { d: kernCore.filter((l) => l.done).length, t: kernCore.length }))}</p></div></div></li>`);
  mods.forEach((m, i) => {
    items.push(moduleStep(m, kctx));
    if (i === hereAt) items.push(goalsChoice());
  });
  if (hereAt < 0) items.push(goalsChoice());
  const ms = (S.kern.milestones || [])[0];
  if (ms) items.push(`<li class="step ms"><span class="mark" aria-hidden="true"></span><p class="milestone">${esc(ms[lang()] || ms.en)}</p></li>`);
  items.push(crossroads(ctx));
  if (branch) {
    const steps = shownSteps(branch.steps);
    const bctx = { ...ctx, kern: false, num: Object.fromEntries(steps.map((l, i) => [l.id, i + 1])) };
    const core = branch.steps.filter((l) => !l.extra);
    items.push(`<li class="step part"><span class="mark" aria-hidden="true"></span><div class="part-head">
      <span class="picbox small">${picture(branch.domein)}</span><div><h3>${esc(name(branch.title))}</h3>
      <p class="muted">${esc(t("wayBranch", { d: core.filter((l) => l.done).length, t: core.length, r: routeName(branch) }))}</p></div></div></li>`);
    const present = LEVELS.filter((lvl) => steps.some((l) => l.level === lvl));
    present.forEach((lvl, i) => {
      items.push(`<li class="step lvl-head"><h4>${esc(S.levels[lvl][lang()])}</h4></li>`);
      for (const m of modulesOf(steps.filter((l) => l.level === lvl))) items.push(moduleStep(m, bctx));
      const nextLvl = present[i + 1];
      if (nextLvl) {
        const done = (branch.milestones || []).find((x) => steps.some((l) => l.id === x.after && l.level === lvl));
        const n = steps.filter((l) => l.level === nextLvl).length;
        items.push(choiceStep("level", `<h3>${esc(t("afterLevel", { l: S.levels[lvl][lang()] }))}</h3>
          ${done ? `<p class="milestone">${esc(done[lang()] || done.en)}</p>` : ""}
          <p>${esc(t("afterLevelSub", { next: `${S.levels[nextLvl][lang()]} (${t("lvLessons", { n })})` }))}</p>`));
      }
    });
    const last = (branch.milestones || []).find((x) => x.after === steps[steps.length - 1]?.id);
    if (last) items.push(`<li class="step ms"><span class="mark" aria-hidden="true"></span><p class="milestone">${esc(last[lang()] || last.en)}</p></li>`);
  } else {
    items.push(levelsExample(ctx));
  }
  items.push(secondDomain(ctx));
  const wt = t("workingTitles") ? `<p class="muted small">${esc(t("workingTitles"))}</p>` : "";
  const legend = `<div class="legend">
    <span><span class="d done"></span>${esc(t("lgDone"))}</span><span><span class="d ready"></span>${esc(t("lgReady"))}</span>
    <span><span class="d soon"></span>${esc(t("lgSoon"))}</span><span><span class="d sq ready"></span>${esc(t("lgTaste"))}</span>
    <span><span class="choice-mini"></span>${esc(t("lgChoice"))}</span></div>`;
  return `<section class="way" id="route">
    <div class="route-head"><h2 class="section">${esc(t("way"))}</h2>
      <label class="toggle"><input type="checkbox" id="extras" ${S.settings.extras ? "checked" : ""}> ${esc(t("extras"))}</label></div>
    <p class="lede small">${esc(t("waySub"))}</p>${legend}${wt}
    <ol class="path">${items.join("")}</ol></section>`;
}

function render() {
  renderSwitches();
  const app = document.getElementById("app");
  app.innerHTML = renderHero() + renderWay() +
    `<footer class="foot"><p>${esc(t("agents"))}</p><p class="muted">${esc(t("footer"))}</p></footer>`;
  app.setAttribute("aria-busy", "false");
  document.getElementById("extras").onchange = (e) => saveSettings({ extras: e.target.checked });
  for (const c of app.querySelectorAll(".chip[data-goal]")) {
    c.onclick = async () => {
      const g = S.goals.find((x) => x.id === c.dataset.goal) || null;
      goal = g ? g.id : null;
      if (g && g.route !== (branchRoute() || S.kern).id) await saveSettings({ route: g.route });
      else render();
    };
  }
  for (const c of app.querySelectorAll("[data-route]")) {
    c.onclick = async () => {
      goal = null;
      const to = c.dataset.route === S.settings.route ? S.kern.id : c.dataset.route;   // a second press undoes the choice
      await saveSettings({ route: to });
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
