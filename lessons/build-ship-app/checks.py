# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-ship-app: the bank's assistant as an app that ships. Four settings files the course wrote for it, and
a log of two days that the course wrote from its runs of version 1 and version 2 (build-error-analysis and
build-eval-set, data/course-production-qwen.json), with the eval set and the course's notes on each reply (build-eval-online).
Three functions to write (settings from a file with defaults, a count per settings version in the log, the newest
settings that are not bad), a choice, and the pictures."""
import html
import importlib.util
import json
import types

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake


def _load(lesson_id: str):
    spec = importlib.util.spec_from_file_location(lesson_id.replace("-", "_") + "_checks_for_ship", paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


O = _load("build-eval-online")
K = _load("build-cost-latency-cache")   # the course's semantic cache over version 2's answers
V, E = O.V, O.E
QUESTIONS, Named = V.QUESTIONS, V.Named
PROMPTS = {"v1": E.SYSTEM, "v2": V.SYSTEM_V2}   # the two system prompts, by their version

# what the app uses for a setting that its settings file leaves out; temperature 0 is greedy, as the recorded runs were
# read-only, so that a box that changes DEFAULTS itself stops with an error instead of changing it for everyone
DEFAULTS = types.MappingProxyType({"model": "Qwen3-0.6B", "prompt": "v1", "max_tokens": 200, "temperature": 0, "cache": False, "cache_threshold": 0.85})

# the course's settings files for the bank's app, oldest first; each names its version by the day it went live
HISTORY = Named([
    {"version": "2026-11-02", "model": "Qwen3-0.6B", "prompt": "v1", "max_tokens": 200, "temperature": 0, "cache": False, "cache_threshold": 0.85},
    {"version": "2026-11-09", "model": "Qwen3-0.6B", "prompt": "v2", "max_tokens": 200, "temperature": 0, "cache": False, "cache_threshold": 0.85},
    {"version": "2026-11-12", "model": "Qwen3-0.6B", "prompt": "v2", "max_tokens": 200, "temperature": 0, "cache": True, "cache_threshold": 0.75},
], "HISTORY")
SETTINGS_TEXT = '{"version": "2026-11-12", "prompt": "v2", "cache": true, "cache_threshold": 0.75}'   # leaves the rest to DEFAULTS
DAYS = (("2026-11-02", "v1", "2026-11-04"), ("2026-11-09", "v2", "2026-11-10"))   # settings, prompt, the day of the log


def _log() -> list:
    """The log of two days: on each, the 120 test questions came in, four minutes apart, and the app answered them
    with that day's settings. The seconds and tokens are the recorded runs; passed is the eval set, made_up the
    course's notes. The times are the course's own."""
    lines = []
    for settings, prompt, day in DAYS:
        answers = O._answers(prompt)
        passed = V.run(prompt)["all"]
        for i, q in enumerate(QUESTIONS):
            minutes = 8 * 60 + i * 4
            entry = {"time": f"{day}T{minutes // 60:02d}:{minutes % 60:02d}:00", "settings": settings, "prompt": prompt,
                     "question": q, "seconds": answers[i]["seconds"], "prompt_tokens": answers[i]["prompt_tokens"],
                     "output_tokens": answers[i]["tokens"], "passed": bool(passed[i]), "made_up": "made-up" in O.NOTES[prompt][i]}
            lines.append(json.dumps(entry, ensure_ascii=False))
    return lines


LOG = Named(_log(), "LOG")


def settings_from(text):
    settings = json.loads(text)
    for key in DEFAULTS:
        if key not in settings:
            settings[key] = DEFAULTS[key]
    return settings


def count_by_settings(lines, field):
    counts = {}
    for line in lines:
        entry = json.loads(line)
        version = entry["settings"]
        if version not in counts:
            counts[version] = 0
        if entry[field]:
            counts[version] = counts[version] + 1
    return counts


def last_good(history, bad):
    good = None
    for settings in history:
        if settings["version"] not in bad:
            good = settings
    return good


def findings() -> dict:
    made = count_by_settings(LOG, "made_up")
    passed = count_by_settings(LOG, "passed")
    f = O.findings()
    return {
        "n": len(QUESTIONS), "lines": len(LOG), "made_up": made, "passed": passed, "eval": f["eval"], "helped": f["helped"],
        "seconds": {s: sum(json.loads(x)["seconds"] for x in LOG if json.loads(x)["settings"] == s) for s, _, _ in DAYS},
        "settings": settings_from(SETTINGS_TEXT),
        "other": {s: f["kinds"][p]["other-problem"] for s, p, _ in DAYS},
        "back": last_good(HISTORY, {"2026-11-12"})["version"], "back_far": last_good(HISTORY, {"2026-11-09", "2026-11-12"})["version"],
        "cache": {t: (len(r["hits"]), len(r["other"]), r["cached_pass"], r["fresh_pass"]) for t, r in ((t, K.run_cache(t)) for t in (0.75, 0.85))},
        "regressions": len(V.regressions(V.run("v1")["all"], V.run("v2")["all"])),
        "prompt_words": {k: len(p.split()) for k, p in PROMPTS.items()},
    }


# --- pictures ------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui)"
_MONO = "font-family:var(--ea-font-code)"

_WORDS = {
    "en": {"flow": ["settings file", "your code", "model", "log"],
           "flow_title": "An LLM app: settings, code, model, log",
           "flow_alt": "Four boxes: the settings file goes into your code, your code calls the model someone else trained, and every request goes into the log as one line. Above the settings file, a stack of older settings files with an arrow back: a rollback loads one of them again.",
           "older": "older settings", "back": "rollback", "trained": "trained by others",
           "set_title": "Your settings_from(text) on SETTINGS_TEXT",
           "set_alt": "A table with a row for each setting: what the file says, and what your function gives the app.",
           "set_head": ["setting", "in the file", "your settings"], "none": "—", "missing": "missing",
           "set_sum": "{k} of the {n} settings your app needs are there; {d} came from DEFAULTS.",
           "set_error": "Your function gave no dictionary.",
           "count_title": "Your count_by_settings(lines, field) on the log",
           "count_alt": "For each of two fields, made_up and passed, a bar for each settings version: the number your function counts, out of the 120 questions of that day.",
           "fields": {"made_up": "made something up", "passed": "passed the eval set"},
           "count_sum": "Your function counts {a} and {b} replies that made something up, and {c} and {d} that passed the eval set.",
           "count_failed": "For {k} of the four bars your function gives no count.",
           "hist_title": "Your last_good(history, bad) on HISTORY",
           "hist_alt": "Two rows with the three settings files in the order they went live; the bad ones are crossed out and the one your function goes back to is marked.",
           "bad": "bad", "yours": "your pick", "nothing": "Your function gives no settings here.",
           "hist_sum": "With {b} bad, your function goes back to {g}.",
           "and": " and "},
    "nl": {"flow": ["instellingenbestand", "jouw code", "model", "log"],
           "flow_title": "Een LLM-app: instellingen, code, model, log",
           "flow_alt": "Vier vakken: het instellingenbestand gaat jouw code in, jouw code roept het model aan dat anderen trainden, en elk request komt als één regel in de log. Boven het instellingenbestand een stapel oudere instellingenbestanden met een pijl terug: een rollback laadt er een van opnieuw.",
           "older": "oudere instellingen", "back": "rollback", "trained": "door anderen getraind",
           "set_title": "Jouw settings_from(text) op SETTINGS_TEXT",
           "set_alt": "Een tabel met een rij voor elke instelling: wat het bestand zegt, en wat je functie de app geeft.",
           "set_head": ["instelling", "in het bestand", "jouw instellingen"], "none": "—", "missing": "ontbreekt",
           "set_sum": "{k} van de {n} instellingen die je app nodig heeft, zijn er; {d} kwamen uit DEFAULTS.",
           "set_error": "Je functie gaf geen dictionary.",
           "count_title": "Jouw count_by_settings(lines, field) op de log",
           "count_alt": "Voor elk van twee velden, made_up en passed, een balk voor elke versie van de instellingen: het getal dat je functie telt, van de 120 vragen van die dag.",
           "fields": {"made_up": "verzon iets", "passed": "door de eval set"},
           "count_sum": "Je functie telt {a} en {b} antwoorden die iets verzonnen, en {c} en {d} die door de eval set kwamen.",
           "count_failed": "Bij {k} van de vier balken geeft je functie geen telling.",
           "hist_title": "Jouw last_good(history, bad) op HISTORY",
           "hist_alt": "Twee rijen met de drie instellingenbestanden in de volgorde waarin ze live gingen; de slechte zijn doorgestreept en het bestand waar je functie naar teruggaat, is gemarkeerd.",
           "bad": "slecht", "yours": "jouw keuze", "nothing": "Je functie geeft hier geen instellingen.",
           "hist_sum": "Met {b} slecht gaat je functie terug naar {g}.",
           "and": " en "},
}


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None


def _mine(seen, name: str):
    mine = seen.mine
    fn = mine.names.get(name) if mine is not None else None
    return fn if callable(fn) else None


def flow_picture(lang: str) -> str:
    """The idea: the settings file feeds the code, the code calls a model someone else trained, every request is logged;
    older settings files wait for a rollback."""
    w = _WORDS[lang]
    e = html.escape
    box_w, gap, y = 150, 34, 58
    xs = [6 + k * (box_w + gap) for k in range(4)]
    parts = []
    for k, (x, label) in enumerate(zip(xs, w["flow"])):
        style = "fill:var(--ea-data-wash);stroke:var(--ea-data)" if k == 1 else ("fill:var(--ea-inset);stroke:var(--ea-ink-2)" if k == 2 else "fill:none;stroke:var(--ea-ink-2)")
        parts.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="40" rx="6" style="{style};stroke-width:1"/>'
                     f'<text x="{x + box_w / 2}" y="{y + 25}" text-anchor="middle" style="{_UI};font-size:12px;fill:var(--ea-ink)">{e(label)}</text>')
        if k < 3:
            parts.append(f'<line x1="{x + box_w}" y1="{y + 20}" x2="{x + box_w + gap - 3}" y2="{y + 20}" style="stroke:var(--ea-ink-2);stroke-width:1"/>'
                         f'<path d="M{x + box_w + gap - 3},{y + 16} l4,4 l-4,4 z" style="fill:var(--ea-ink-2)"/>')
    parts.append(f'<text x="{xs[2] + box_w / 2}" y="{y + 56}" text-anchor="middle" style="{_UI};font-size:11px;fill:var(--ea-ink-2)">{e(w["trained"])}</text>')
    for k in range(3):   # the stack of older settings files above the first box
        parts.append(f'<rect x="{xs[0] + 8 + k * 4}" y="{6 + k * 4}" width="{box_w - 16}" height="26" rx="4" style="fill:var(--ea-page);stroke:var(--ea-ink-3);stroke-width:1"/>')
    parts.append(f'<text x="{xs[0] + box_w / 2 + 4}" y="{27}" text-anchor="middle" style="{_UI};font-size:11px;fill:var(--ea-ink-2)">{e(w["older"])}</text>'
                 f'<path d="M{xs[0] + box_w - 4},{24} C{xs[0] + box_w + 26},{24} {xs[0] + box_w + 26},{y + 6} {xs[0] + box_w + 2},{y + 6}" style="fill:none;stroke:var(--ea-data);stroke-width:1.2"/>'
                 f'<path d="M{xs[0] + box_w + 6},{y + 2} l-5,4 l5,4 z" style="fill:var(--ea-data)"/>'
                 f'<text x="{xs[0] + box_w + 26}" y="{40}" style="{_UI};font-size:11px;fill:var(--ea-data-text)">{e(w["back"])}</text>')
    width = xs[-1] + box_w + 6
    svg = (f'<div style="overflow-x:auto"><svg viewBox="0 0 {width} {y + 64}" width="100%" style="min-width:560px;max-width:{width}px" '
           f'role="img" aria-label="{e(w["flow_alt"])}">{"".join(parts)}</svg></div>')
    return yours.figure(w["flow_title"], svg, alt=w["flow_alt"])


def _table(head: list, rows: list) -> str:
    cell = f"{_UI};font-size:12px;color:var(--ea-ink);padding:0.25rem 0.5rem;vertical-align:top;border-top:1px solid var(--ea-line)"
    th = f"{_UI};font-size:12px;font-weight:400;color:var(--ea-ink-2);padding:0.2rem 0.5rem;text-align:left"
    out = "<tr>" + "".join(f'<th style="{th}">{html.escape(h)}</th>' for h in head) + "</tr>"
    for row in rows:
        out += "<tr>" + "".join(f'<td style="{cell}">{x}</td>' for x in row) + "</tr>"
    return f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;max-width:36rem">{out}</table></div>'


def settings_picture(seen) -> str:
    fn = _mine(seen, "settings_from")
    if fn is None:
        return ""
    w = _WORDS[seen.lang]
    e = html.escape
    got = _safe(fn, SETTINGS_TEXT)
    in_file = json.loads(SETTINGS_TEXT)
    keys = list(in_file) + [k for k in DEFAULTS if k not in in_file]
    if not isinstance(got, dict):
        body = f'<div style="{_UI};font-size:13px;color:var(--ea-loss-text)">{e(w["set_error"])}</div>'
        return yours.figure(w["set_title"], f'<div role="img" aria-label="{e(w["set_alt"])}">{body}</div>')
    rows, there, from_defaults = [], 0, 0
    for k in keys:
        file_value = f'<span style="{_MONO}">{e(json.dumps(in_file[k]))}</span>' if k in in_file else f'<span style="color:var(--ea-ink-3)">{e(w["none"])}</span>'
        if k in got:
            there += 1
            from_defaults += k not in in_file
            color = "var(--ea-data-text)" if k not in in_file else "var(--ea-ink)"
            mine = f'<span style="{_MONO};color:{color}">{e(json.dumps(got[k], default=str))}</span>'
        else:
            mine = f'<span style="color:var(--ea-loss-text)">{e(w["missing"])}</span>'
        rows.append([f'<span style="{_MONO}">{e(k)}</span>', file_value, mine])
    summary = f'<div style="{_UI};font-size:13px;color:{"var(--ea-ink)" if there == len(keys) else "var(--ea-loss-text)"};margin-top:0.4rem">{e(w["set_sum"].format(k=there, n=len(keys), d=from_defaults))}</div>'
    return yours.figure(w["set_title"], f'<div role="img" aria-label="{e(w["set_alt"])}">{_table(w["set_head"], rows)}{summary}</div>')


def counts_picture(seen) -> str:
    fn = _mine(seen, "count_by_settings")
    if fn is None:
        return ""
    w = _WORDS[seen.lang]
    e = html.escape
    n = len(QUESTIONS)
    rows, said = "", []
    for field in ("made_up", "passed"):
        got = _safe(fn, list(LOG), field)
        rows += f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2);margin:0.5rem 0 0.2rem"><span style="{_MONO}">{field}</span>: {e(w["fields"][field])}</div>'
        for settings, prompt, _ in DAYS:
            value = got.get(settings) if isinstance(got, dict) else None
            ok = isinstance(value, int) and not isinstance(value, bool)
            said.append(str(value) if ok else "?")
            share = min(max(value, 0), n) / n if ok else 0
            color = "var(--ea-loss)" if field == "made_up" else "var(--ea-data)"
            rows += (f'<div style="display:flex;align-items:center;gap:0.6rem;margin:0.2rem 0">'
                     f'<span style="{_MONO};font-size:12px;color:var(--ea-ink);flex:0 0 9rem">{e(settings)} ({prompt})</span>'
                     f'<span style="flex:1;background:var(--ea-inset);border-radius:3px;height:12px;position:relative">'
                     f'<span style="position:absolute;left:0;top:0;bottom:0;width:{share * 100:.1f}%;background:{color};border-radius:3px"></span></span>'
                     f'<span style="{_UI};font-size:12px;color:var(--ea-ink);flex:0 0 4.5rem;text-align:right">{e(said[-1])} / {n}</span></div>')
    failed = said.count("?")
    line = w["count_failed"].format(k=failed) if failed else w["count_sum"].format(a=said[0], b=said[1], c=said[2], d=said[3])
    summary = f'<div style="{_UI};font-size:13px;color:{"var(--ea-loss-text)" if failed else "var(--ea-ink)"};margin-top:0.5rem">{e(line)}</div>'
    return yours.figure(w["count_title"], f'<div role="img" aria-label="{e(w["count_alt"])}" style="overflow-x:auto"><div style="min-width:26rem">{rows}</div>{summary}</div>')


def history_picture(seen) -> str:
    fn = _mine(seen, "last_good")
    if fn is None:
        return ""
    w = _WORDS[seen.lang]
    e = html.escape
    out = ""
    for bad in ({"2026-11-12"}, {"2026-11-09", "2026-11-12"}):
        got = _safe(fn, [dict(s) for s in HISTORY], set(bad))
        pick = got.get("version") if isinstance(got, dict) else None
        boxes = ""
        for s in HISTORY:
            v = s["version"]
            if v in bad:
                style = "border:1px solid var(--ea-loss);color:var(--ea-loss-text);text-decoration:line-through"
            elif v == pick:
                style = "border:2px solid var(--ea-data);background:var(--ea-data-wash);color:var(--ea-ink)"
            else:
                style = "border:1px solid var(--ea-ink-3);color:var(--ea-ink-2)"
            cache = f' · cache {s["cache_threshold"]}' if s["cache"] else ""
            boxes += (f'<span style="{_MONO};font-size:12px;display:inline-block;padding:0.25rem 0.5rem;margin:0.2rem;border-radius:4px;{style}">'
                      f'{e(v)} · {s["prompt"]}{e(cache)}</span>')
        bad_text = w["and"].join(sorted(bad))
        line = w["hist_sum"].format(b=bad_text, g=pick) if pick else w["nothing"]
        out += (f'<div style="margin:0.4rem 0 0.8rem">{boxes}'
                f'<div style="{_UI};font-size:13px;color:{"var(--ea-ink)" if pick else "var(--ea-loss-text)"};margin-top:0.2rem">{e(line)}</div></div>')
    return yours.figure(w["hist_title"], f'<div role="img" aria-label="{e(w["hist_alt"])}">{out}</div>')


# --- exercises -----------------------------------------------------------------------------------------------------

START_SET = '''def settings_from(text):
    settings = json.loads(text)
    return settings
'''
SOLUTION_SET = '''def settings_from(text):
    settings = json.loads(text)
    for key in DEFAULTS:
        if key not in settings:
            settings[key] = DEFAULTS[key]
    return settings
'''
_SET_OVER = '''def settings_from(text):
    settings = json.loads(text)
    for key in DEFAULTS:
        settings[key] = DEFAULTS[key]
    return settings
'''
_SET_SAME = '''def settings_from(text):
    settings = DEFAULTS
    settings.update(json.loads(text))
    return settings
'''
_SET_ONLY = '''def settings_from(text):
    settings = json.loads(text)
    for key in DEFAULTS:
        if key not in settings:
            settings = DEFAULTS
    return settings
'''

START_COUNT = '''def count_by_settings(lines, field):
    counts = {}
    for line in lines:
        entry = json.loads(line)
        version = entry["settings"]
    return counts
'''
SOLUTION_COUNT = '''def count_by_settings(lines, field):
    counts = {}
    for line in lines:
        entry = json.loads(line)
        version = entry["settings"]
        if version not in counts:
            counts[version] = 0
        if entry[field]:
            counts[version] = counts[version] + 1
    return counts
'''
_COUNT_ALL = SOLUTION_COUNT.replace("        if entry[field]:\n            counts[version] = counts[version] + 1\n", "        counts[version] = counts[version] + 1\n")
_COUNT_SKIP = '''def count_by_settings(lines, field):
    counts = {}
    for line in lines:
        entry = json.loads(line)
        version = entry["settings"]
        if entry[field]:
            if version not in counts:
                counts[version] = 0
            counts[version] = counts[version] + 1
    return counts
'''
_COUNT_PROMPT = SOLUTION_COUNT.replace('version = entry["settings"]', 'version = entry["prompt"]')
_COUNT_RESET = SOLUTION_COUNT.replace("        if version not in counts:\n            counts[version] = 0\n", "        counts[version] = 0\n")

START_GOOD = '''def last_good(history, bad):
    good = None
    for settings in history:
        good = settings
    return good
'''
SOLUTION_GOOD = '''def last_good(history, bad):
    good = None
    for settings in history:
        if settings["version"] not in bad:
            good = settings
    return good
'''
_GOOD_FIRST = '''def last_good(history, bad):
    for settings in history:
        if settings["version"] not in bad:
            return settings
    return None
'''
_GOOD_BAD = SOLUTION_GOOD.replace("not in bad", "in bad")
_GOOD_PREVIOUS = '''def last_good(history, bad):
    good = None
    for settings in history:
        if settings["version"] not in bad:
            good = settings
    return history[-2]
'''

_NAMES = {"json": json, "DEFAULTS": DEFAULTS, "LOG": LOG, "HISTORY": HISTORY, "SETTINGS_TEXT": SETTINGS_TEXT}

_CLEARED = {"en": "Your box is empty. Click *Start over* under the box to get the start code back.",
            "nl": "Je vak is leeg. Klik op *Opnieuw beginnen* onder het vak om de begincode terug te krijgen."}


def _texts(name: str, empty: dict, wrong: dict, correct: dict) -> dict:
    return {
        "missing": {"en": f"Run the cell with `{name}` first.", "nl": f"Voer eerst de cel met `{name}` uit."},
        "cleared": _CLEARED,
        "empty": empty,
        "wrong": wrong,
        "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`.",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
        "no_function": {"en": f"Your box does not make a function `{name}`. Keep the `def` line as it was.",
                        "nl": f"Je vak maakt geen functie `{name}`. Laat de `def`-regel staan zoals hij was."},
        "correct": correct,
    }


def _line(settings: str, made_up: bool, passed: bool, prompt: str = "v1") -> str:
    return json.dumps({"time": "2026-11-06T08:00:00", "settings": settings, "prompt": prompt, "question": "Where is my card?",
                       "seconds": 7.0, "prompt_tokens": 300, "output_tokens": 40, "passed": passed, "made_up": made_up})


_LINES_A = Named([_line("a", True, False), _line("a", False, True), _line("b", False, True), _line("a", True, True)], "lines_a")
_LINES_B = Named([_line("x", False, True, "v2"), _line("y", True, False, "v2"), _line("x", False, False, "v2")], "lines_b")
_HIST_A = Named([{"version": "1", "prompt": "v1"}, {"version": "2", "prompt": "v1"}, {"version": "3", "prompt": "v2"}], "history_a")

class _SettingsBox(FunctionBox):
    """Writes the text of a settings file in a message by its name, or between single quotes, so its double quotes need
    no backslashes."""

    def _call_text(self, case) -> str:
        return f"settings_from({'SETTINGS_TEXT' if case[0] == SETTINGS_TEXT else repr(case[0])})"


EXERCISES = {
    "settings_from": _SettingsBox(
        name="settings_from",
        picture=settings_picture,
        function="settings_from",
        start=START_SET,
        solution=SOLUTION_SET,
        visible=[(SETTINGS_TEXT,)],
        hidden=[('{"version": "2026-11-09", "prompt": "v2", "max_tokens": 120, "cache": false}',), ('{"version": "x"}',),
                (json.dumps(HISTORY[2]),)],
        mistakes=[
            Mistake("over", wrong=_SET_OVER,
                    en="`{call}` gives `{got}`: every default overwrites what the file says. Fill in a default only for a key the file does not have.",
                    nl="`{call}` geeft `{got}`: elke standaardwaarde overschrijft wat het bestand zegt. Vul een standaardwaarde alleen in voor een key die het bestand niet heeft."),
            Mistake("same", error=lambda e: "mappingproxy" in str(e),
                    en="`{call}` stops: your code changes `DEFAULTS` itself, which every settings file shares. Start from what the file says, `json.loads(text)`, and add to that.",
                    nl="`{call}` stopt: je code verandert `DEFAULTS` zelf, dat elk instellingenbestand deelt. Begin bij wat het bestand zegt, `json.loads(text)`, en vul dat aan."),
            Mistake("only", wrong=_SET_ONLY,
                    en="`{call}` gives `{got}`: when one key is missing, you swap the whole file for DEFAULTS. Add only the missing key, with its value from DEFAULTS.",
                    nl="`{call}` geeft `{got}`: als één key ontbreekt, ruil je het hele bestand in voor DEFAULTS. Voeg alleen de ontbrekende key toe, met de waarde uit DEFAULTS."),
        ],
        texts=_texts("settings_from",
                     empty={"en": "Your `settings_from` gives only what the file says. Add the keys of `DEFAULTS` that the file leaves out.",
                            "nl": "Je `settings_from` geeft alleen wat het bestand zegt. Voeg de keys van `DEFAULTS` toe die het bestand weglaat."},
                     wrong={"en": "`{call}` gives `{got}`, and it should give `{want}`: what the file says, and a default for each key it leaves out.",
                            "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: wat het bestand zegt, en een standaardwaarde voor elke key die het weglaat."},
                     correct={"en": "Your `settings_from` is right in all {n} cases. Which settings in the picture came from `DEFAULTS`?",
                              "nl": "Je `settings_from` klopt in alle {n} gevallen. Welke instellingen in het plaatje kwamen uit `DEFAULTS`?"}),
        names=_NAMES,
    ),
    "count_by_settings": FunctionBox(
        name="count_by_settings",
        picture=counts_picture,
        function="count_by_settings",
        start=START_COUNT,
        solution=SOLUTION_COUNT,
        visible=[(LOG, "made_up")],
        hidden=[(LOG, "passed"), (_LINES_A, "made_up"), (_LINES_A, "passed"), (_LINES_B, "made_up"), (_LINES_B, "passed")],
        mistakes=[
            Mistake("all", wrong=_COUNT_ALL,
                    en="`{call}` gives `{got}`: every line of each version, whatever `field` says. Add 1 only when `entry[field]` is `True`.",
                    nl="`{call}` geeft `{got}`: elke regel van elke versie, wat `field` ook zegt. Tel alleen 1 op als `entry[field]` `True` is."),
            Mistake("skip", wrong=_COUNT_SKIP,
                    en="`{call}` gives `{got}`: a version without a single `True` is missing, while 0 is the answer for it. Start each version at 0 before the `if` on `field`.",
                    nl="`{call}` geeft `{got}`: een versie zonder één keer `True` ontbreekt, terwijl 0 daar het antwoord is. Zet elke versie op 0 voor de `if` op `field`."),
            Mistake("prompt", wrong=_COUNT_PROMPT,
                    en="`{call}` gives `{got}`: you count by the version of the prompt. Count by the version of the settings, `entry[\"settings\"]`: the prompt is one of its settings.",
                    nl="`{call}` geeft `{got}`: je telt per versie van de prompt. Tel per versie van de instellingen, `entry[\"settings\"]`: de prompt is een van de instellingen."),
            Mistake("reset", wrong=_COUNT_RESET,
                    en="`{call}` gives `{got}`: every line sets its version back to 0, so only the last line counts. Set a version to 0 only the first time it comes by.",
                    nl="`{call}` geeft `{got}`: elke regel zet zijn versie terug op 0, dus alleen de laatste regel telt. Zet een versie alleen op 0 de eerste keer dat hij langskomt."),
        ],
        texts=_texts("count_by_settings",
                     empty={"en": "Your `count_by_settings` reads each line but counts nothing yet. For each version, count the lines where `field` is `True`.",
                            "nl": "Je `count_by_settings` leest elke regel, maar telt nog niets. Tel voor elke versie de regels waar `field` `True` is."},
                     wrong={"en": "`{call}` gives `{got}`, and it should give `{want}`: for each settings version, the lines where `field` is `True`.",
                            "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: voor elke versie van de instellingen de regels waar `field` `True` is."},
                     correct={"en": "Your `count_by_settings` is right in all {n} cases. Compare the two pairs of bars: which version would you keep?",
                              "nl": "Je `count_by_settings` klopt in alle {n} gevallen. Vergelijk de twee paren balken: welke versie zou jij houden?"}),
        names=_NAMES,
    ),
    "last_good": FunctionBox(
        name="last_good",
        picture=history_picture,
        function="last_good",
        start=START_GOOD,
        solution=SOLUTION_GOOD,
        visible=[(HISTORY, {"2026-11-12"})],
        hidden=[(HISTORY, {"2026-11-09", "2026-11-12"}), (HISTORY, {"2026-11-09"}), (HISTORY, set()), (_HIST_A, {"3", "2"}), (_HIST_A, {"3"}),
                (_HIST_A, {"1", "2", "3"})],
        mistakes=[
            Mistake("first", wrong=_GOOD_FIRST,
                    en="`{call}` gives `{got}`: the oldest settings that are not bad. Go back as little as you can: keep the newest that is not bad.",
                    nl="`{call}` geeft `{got}`: de oudste instellingen die niet slecht zijn. Ga zo weinig mogelijk terug: houd de nieuwste die niet slecht is."),
            Mistake("bad", wrong=_GOOD_BAD,
                    en="`{call}` gives `{got}`: settings that are bad. Keep only the settings whose version is not in `bad`.",
                    nl="`{call}` geeft `{got}`: instellingen die slecht zijn. Houd alleen de instellingen waarvan de versie niet in `bad` zit."),
            Mistake("previous", wrong=_GOOD_PREVIOUS,
                    en="`{call}` gives `{got}`: always the settings before the newest. When that one is bad too, go back further.",
                    nl="`{call}` geeft `{got}`: altijd de instellingen voor de nieuwste. Als die ook slecht zijn, ga dan verder terug."),
        ],
        texts=_texts("last_good",
                     empty={"en": "Your `last_good` gives the newest settings, bad or not. Skip the settings whose version is in `bad`.",
                            "nl": "Je `last_good` geeft de nieuwste instellingen, slecht of niet. Sla de instellingen over waarvan de versie in `bad` zit."},
                     wrong={"en": "`{call}` gives `{got}`, and it should give `{want}`: the newest settings whose version is not in `bad`.",
                            "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de nieuwste instellingen waarvan de versie niet in `bad` zit."},
                     correct={"en": "Your `last_good` is right in all {n} cases. In the second row, what did the bank lose by going back that far?",
                              "nl": "Je `last_good` klopt in alle {n} gevallen. Wat raakte de bank in de tweede rij kwijt door zo ver terug te gaan?"}),
        names=_NAMES,
    ),
}

REQUIRED = ["settings_from", "count_by_settings", "last_good", "first-step"]

CASES = [
    ("settings_from", START_SET, "empty", None),
    ("settings_from", "", "empty", None),
    ("settings_from", SOLUTION_SET, "correct", None),
    ("settings_from", '''def settings_from(text):
    settings = dict(DEFAULTS)
    settings.update(json.loads(text))
    return settings
''', "correct", None),
    ("settings_from", _SET_OVER, "wrong", "over"),
    ("settings_from", _SET_ONLY, "wrong", "only"),
    ("settings_from", _SET_SAME, "error", "same"),
    ("count_by_settings", START_COUNT, "empty", None),
    ("count_by_settings", "", "empty", None),
    ("count_by_settings", SOLUTION_COUNT, "correct", None),
    ("count_by_settings", '''def count_by_settings(lines, field):
    entries = [json.loads(line) for line in lines]
    counts = {e["settings"]: 0 for e in entries}
    for e in entries:
        counts[e["settings"]] += int(bool(e[field]))
    return counts
''', "correct", None),
    ("count_by_settings", _COUNT_ALL, "wrong", "all"),
    ("count_by_settings", _COUNT_SKIP, "wrong", "skip"),
    ("count_by_settings", _COUNT_PROMPT, "wrong", "prompt"),
    ("count_by_settings", _COUNT_RESET, "wrong", "reset"),
    ("last_good", START_GOOD, "empty", None),
    ("last_good", "", "empty", None),
    ("last_good", SOLUTION_GOOD, "correct", None),
    ("last_good", '''def last_good(history, bad):
    good = [s for s in history if s["version"] not in bad]
    return good[-1] if good else None
''', "correct", None),
    ("last_good", _GOOD_FIRST, "wrong", "first"),
    ("last_good", _GOOD_BAD, "wrong", "bad"),
    ("last_good", _GOOD_PREVIOUS, "wrong", "previous"),
]
