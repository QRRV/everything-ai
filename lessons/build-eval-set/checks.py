# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-eval-set: the 120 test questions of build-error-analysis as an eval set, each with the topic it is
about, and checks in code for the kinds of mistake that code can see. Version 1 of the bank's assistant is the run of
build-error-analysis; version 2 ties each fact to its topic (data/course-production-qwen.json). How well each check
finds the kind it is for, measured against the course's notes on version 1; the eval set run on both versions, and the
regression test between them. Two checks to write, a function that finds the regressions, the pictures."""
import functools
import html
import importlib.util
import json
import re

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake


def _error_analysis():
    spec = importlib.util.spec_from_file_location("build_error_analysis_checks", paths.LESSONS / "build-error-analysis" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _error_analysis()
DATA = E.DATA
QUESTIONS, LABELS, NOTES, Named = E.QUESTIONS, E.LABELS, E.NOTES, E.Named
FACTS_BY_TOPIC = {
    "card_arrival": "A new card arrives within 5 working days. Its status is in the app under Cards. There is no tracking number.",
    "lost_or_stolen_card": "Freeze the card at once in the app under Cards > Freeze, then order a new card there. A new card is free.",
    "exchange_rate": "A payment in another currency gets the exchange rate at the moment of payment; on weekends 0.5% is added. "
                     "The app shows the rate before you pay.",
    "card_payment_wrong_exchange_rate": "Send the date and the amount of the payment in this chat, and the team checks it "
                                        "within 2 working days.",
    "pin_blocked": "The PIN is blocked after 3 wrong tries. Unblock it in the app under Cards > PIN, 24 hours after the last try.",
    "request_refund": "Ask the shop first. If the shop has not helped within 14 days, the team can open a dispute.",
}
SYSTEM_V2 = ("You answer customers of an online bank in the chat of its app. These are the topics a message can be about, "
             "each with its facts:\n" + "\n".join(f"{t}: {f}" for t, f in FACTS_BY_TOPIC.items()) + "\n"
             "First choose the topic of the customer's message, then answer with the facts of that topic only. "
             "If the facts do not answer the question, say that a person from the team will reply.\n"
             'Answer with JSON only: {"topic": "...", "reply": "..."}. reply is your answer to the customer, in at most '
             "three sentences.")
RECORD = {"v2": {"labels": "test", "system": SYSTEM_V2, "max_tokens": 200}}
RATE = ("card_payment_wrong_exchange_rate", "exchange_rate")   # the topics where asking for a payment's date and amount fits
ADMIT_WORDS = ("incorrect", "wrong", "too low")


@functools.cache
def texts(version: str) -> tuple:
    """The answers of a version, "v1" (the run of build-error-analysis) or "v2" (this lesson's), as texts."""
    if version == "v1":
        return tuple(a["text"] for a in E.answers())
    return tuple(a["text"] for a in E.answers("build-eval-set", "v2"))


TEXTS_V1 = Named(texts("v1"), "TEXTS_V1")


def reply_of(text):
    """The answer as a dict with "topic" and "reply", or None when it is not JSON of that form."""
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("topic"), str) or not isinstance(value.get("reply"), str):
        return None
    return value


# --- the course's checks: one for the form, and one for each kind of mistake that code can see --------------------------

def json_ok(text, topic):
    return reply_of(text) is not None


def topic_ok(text, topic):
    answer = reply_of(text)
    if answer is None:
        return False
    return answer["topic"] == topic


def steps_ok(text, topic):
    answer = reply_of(text)
    if answer is None:
        return False
    return "date and amount" not in answer["reply"] or topic in RATE


def admits_ok(text, topic):
    answer = reply_of(text)
    if answer is None:
        return False
    return not any(w in answer["reply"].lower() for w in ADMIT_WORDS)


def short_ok(text, topic):
    answer = reply_of(text)
    if answer is None:
        return False
    return E.sentences(answer["reply"]) <= 3


CHECKS = {"json_ok": json_ok, "topic_ok": topic_ok, "steps_ok": steps_ok, "short_ok": short_ok}   # the eval set
TRIED = {**CHECKS, "admits_ok": admits_ok}   # every check the course tried against its notes; admits_ok did not pass that test
FOR_KIND = {"topic_ok": "wrong-topic", "steps_ok": "other-problem", "admits_ok": "admits", "short_ok": "too-long"}


def run(version: str, checks: dict = CHECKS) -> dict:
    """The eval set on a version: for each check, a list with True for each input that passes it; "all" for passing
    every check."""
    out = {name: [bool(fn(t, topic)) for t, topic in zip(texts(version), LABELS)] for name, fn in checks.items()}
    out["all"] = [all(out[name][i] for name in checks) for i in range(len(LABELS))]
    return out


def against_notes(name: str, fn=None) -> dict:
    """How a check does on version 1 against the course's notes for its kind: the inputs it fails, those of them the
    course noted, the noted ones it missed, and its false alarms."""
    fn = fn or TRIED[name]
    kind = FOR_KIND[name]
    fails = [i for i, (t, topic) in enumerate(zip(texts("v1"), LABELS)) if not fn(t, topic)]
    noted = [i for i, k in enumerate(NOTES) if kind in k]
    return {"fails": fails, "noted": noted, "caught": [i for i in fails if i in noted],
            "missed": [i for i in noted if i not in fails], "alarms": [i for i in fails if i not in noted]}


def regressions(old: list, new: list) -> list:
    return [i for i in range(len(old)) if old[i] and not new[i]]


def findings() -> dict:
    """The numbers the lesson gives: the checks against the notes, and the eval set on both versions."""
    v1, v2 = run("v1"), run("v2")
    return {
        "n": len(LABELS),
        "notes": {name: against_notes(name) for name in FOR_KIND},
        "v1": {name: sum(v) for name, v in v1.items()},
        "v2": {name: sum(v) for name, v in v2.items()},
        "fixed": [i for i in range(len(LABELS)) if not v1["all"][i] and v2["all"][i]],
        "broke": regressions(v1["all"], v2["all"]),
        "topic_broke": regressions(v1["topic_ok"], v2["topic_ok"]),
        "topic_fixed": [i for i in range(len(LABELS)) if not v1["topic_ok"][i] and v2["topic_ok"][i]],
        "v2_topics": {t: sum(reply_of(x) is not None and reply_of(x)["topic"] == t for x in texts("v2")) for t in E.TOPICS},
        "right": {version: {t: sum(run(version)["topic_ok"][i] for i in range(len(LABELS)) if LABELS[i] == t) for t in E.TOPICS}
                  for version in ("v1", "v2")},
        "arrival_lost": sum(LABELS[i] == "card_arrival" and reply_of(texts("v2")[i])["topic"] == "lost_or_stolen_card"
                            for i in range(len(LABELS))),
        # replies that tell the customer to freeze the card ("freeze your card", "Freeze the card", "freeze it"), not the
        # ones that only name the menu Cards > Freeze
        "arrival_freeze": sum(LABELS[i] == "card_arrival" and re.search(r"\bfreeze (your|the|it)\b", reply_of(texts("v2")[i])["reply"].lower()) is not None
                              for i in range(len(LABELS))),
        "broke_arrival": sum(LABELS[i] == "card_arrival" for i in regressions(v1["all"], v2["all"])),
        "broke_rate": sum(LABELS[i] == "card_payment_wrong_exchange_rate" for i in regressions(v1["all"], v2["all"])),
    }


# --- pictures ---------------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui)"
_MONO = "font-family:var(--ea-font-code)"
_MORE = {
    "en": {"flow": ["120 inputs, each with its topic", "the app, version 1 or 2", "checks in code", "passes per check"],
           "compare": "compare the versions: fixed and regressions",
           "flow_title": "An eval set",
           "flow_alt": "Four steps in a row, joined by arrows: 120 inputs, each with the topic it is about, go into a version of the app; checks in code look at each answer; for each check, the number of answers that pass. Under them: compare version 1 with version 2, the inputs it fixed and its regressions.",
           "tested_title": "The course's checks on the 120 replies of version 1, against its notes",
           "tested_alt": "A table with a row for each check the course tried: the kind it is for, how many replies it fails, how many of those the course noted with that kind, the noted ones it missed, and its false alarms.",
           "check": "check", "for": "for the kind", "fails": "fails", "caught": "noted too", "missed": "missed", "alarms": "false alarms",
           "out": "not in the eval set",
           "versions_title": "The eval set on version 1 and version 2",
           "versions_alt": "A table with a row for each check of the eval set, and a last row for passing every check: how many of the 120 answers pass it with version 1, and how many with version 2.",
           "v1": "version 1", "v2": "version 2", "every": "every check",
           "input": "Input {i} of {n}", "customer": "customer", "expects": "the eval set expects",
           "groups": ("fixed by version 2: {n}", "regressions of version 2: {n}", "pass with both: {n}", "fail with both: {n}", "all {n} inputs")},
    "nl": {"flow": ["120 inputs, elk met zijn onderwerp", "de app, versie 1 of 2", "checks in code", "geslaagd per check"],
           "compare": "de versies vergelijken: verbeterd en regressions",
           "flow_title": "Een eval set",
           "flow_alt": "Vier stappen op een rij, verbonden door pijlen: 120 inputs, elk met het onderwerp waar ze over gaan, gaan naar een versie van de app; checks in code bekijken elk antwoord; per check het aantal antwoorden dat slaagt. Daaronder: versie 1 met versie 2 vergelijken, de inputs die beter werden en de regressions.",
           "tested_title": "De checks van de course op de 120 antwoorden van versie 1, naast de notities",
           "tested_alt": "Een tabel met een rij voor elke check die de course probeerde: de soort waar hij voor is, hoeveel antwoorden hij afkeurt, hoeveel daarvan de course met die soort noteerde, de genoteerde die hij miste, en zijn vals alarm.",
           "check": "check", "for": "voor de soort", "fails": "keurt af", "caught": "ook genoteerd", "missed": "gemist", "alarms": "vals alarm",
           "out": "niet in de eval set",
           "versions_title": "De eval set op versie 1 en versie 2",
           "versions_alt": "Een tabel met een rij voor elke check van de eval set, en een laatste rij voor door elke check komen: hoeveel van de 120 antwoorden erdoor komen met versie 1, en hoeveel met versie 2.",
           "v1": "versie 1", "v2": "versie 2", "every": "elke check",
           "input": "Input {i} van {n}", "customer": "klant", "expects": "de eval set verwacht",
           "groups": ("verbeterd door versie 2: {n}", "regressions van versie 2: {n}", "slagen met allebei: {n}", "falen met allebei: {n}", "alle {n} inputs")},
}
_WORDS = {
    "en": {"fails": "your check fails it", "noted": "the course noted {kind}", "pass": "passes",
           "strip": "Your {name} on the 120 replies of version 1",
           "sum": "Your check fails {f}; the course noted {kind} in {k}; both: {b}.",
           "alt": "A row of 120 small squares, one for each reply of version 1, in the order of the questions. A filled square is a reply your check fails; a square with a thick border is one the course noted as {kind}.",
           "v1": "version 1", "v2": "version 2", "reg": "regression", "pass_all": "passes every check",
           "reg_title": "Your regressions(PASSED_V1, PASSED_V2)",
           "reg_sum": "Your function gives {n} inputs, at positions {items} of the list (counting from 0).",
           "reg_one": "Your function gives 1 input, at position {items} of the list (counting from 0).",
           "reg_none": "Your function gives no inputs.", "reg_odd": "Your function gives {items}.",
           "reg_alt": "Two rows of 120 small squares, one for each input of the eval set: the upper row for version 1, the lower for version 2. A filled square passes every check. The inputs your function gives are marked under the lower row."},
    "nl": {"fails": "je check keurt het af", "noted": "de course noteerde {kind}", "pass": "komt erdoor",
           "strip": "Je {name} op de 120 antwoorden van versie 1",
           "sum": "Je check keurt er {f} af; de course noteerde {kind} bij {k}; allebei: {b}.",
           "alt": "Een rij van 120 kleine vierkantjes, één voor elk antwoord van versie 1, in de volgorde van de vragen. Een gevuld vierkantje is een antwoord dat je check afkeurt; een vierkantje met een dikke rand is er een dat de course als {kind} noteerde.",
           "v1": "versie 1", "v2": "versie 2", "reg": "regression", "pass_all": "komt door elke check",
           "reg_title": "Jouw regressions(PASSED_V1, PASSED_V2)",
           "reg_sum": "Je functie geeft {n} inputs, op plaats {items} in de lijst (geteld vanaf 0).",
           "reg_one": "Je functie geeft 1 input, op plaats {items} in de lijst (geteld vanaf 0).",
           "reg_none": "Je functie geeft geen inputs.", "reg_odd": "Je functie geeft {items}.",
           "reg_alt": "Twee rijen van 120 kleine vierkantjes, één voor elke input van de eval set: de bovenste rij voor versie 1, de onderste voor versie 2. Een gevuld vierkantje komt door elke check. De inputs die je functie geeft, staan onder de onderste rij gemarkeerd."},
}


def _squares(marks: list, ring: list | None = None, under: list | None = None, fill: str = "var(--ea-loss)") -> str:
    size, gap, per_row = 11, 3, 40
    rows = (len(marks) + per_row - 1) // per_row
    parts = []
    for i, filled in enumerate(marks):
        x, y = (i % per_row) * (size + gap) + 2, (i // per_row) * (size + gap + (6 if under is not None else 0)) + 2
        border = "var(--ea-ink)" if ring and ring[i] else "var(--ea-line)"
        width = 2.2 if ring and ring[i] else 1
        parts.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="2" style="fill:{fill if filled else "var(--ea-page)"};'
                     f'stroke:{border};stroke-width:{width}"/>')
        if under is not None and under[i]:
            parts.append(f'<path d="M{x + size / 2} {y + size + 1} l-3.5 5 h7 z" style="fill:var(--ea-ink)"/>')
    w = per_row * (size + gap) + 4
    h = rows * (size + gap + (6 if under is not None else 0)) + 4
    return f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" aria-hidden="true">{"".join(parts)}</svg>'


def _key(items: list) -> str:
    out = []
    for fill, border, text in items:
        out.append(f'<span><svg width="14" height="14" aria-hidden="true"><rect x="1.5" y="1.5" width="11" height="11" rx="2" '
                   f'style="fill:{fill};stroke:{border};stroke-width:{2.2 if border == "var(--ea-ink)" else 1}"/></svg> {html.escape(text)}</span>')
    return (f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2);display:flex;gap:1rem;flex-wrap:wrap;margin-top:0.3rem">'
            f'{"".join(out)}</div>')


def strip_picture(name: str):
    """What the learner's check says about each reply of version 1, beside the course's notes for its kind."""
    def picture(seen) -> str:
        mine = seen.mine
        fn = mine.names.get(name) if mine is not None else None
        if not callable(fn):
            return ""
        try:
            fails = [not fn(t, topic) for t, topic in zip(texts("v1"), LABELS)]
        except Exception:
            return ""
        w = _WORDS[seen.lang]
        kind = E.kind_name(FOR_KIND[name], seen.lang)
        noted = [FOR_KIND[name] in k for k in NOTES]
        body = (_squares(fails, ring=noted)
                + _key([("var(--ea-loss)", "var(--ea-line)", w["fails"]), ("var(--ea-page)", "var(--ea-ink)", w["noted"].format(kind=kind))])
                + f'<div style="{_UI};font-size:13px;color:var(--ea-ink);margin-top:0.35rem">'
                + html.escape(w["sum"].format(f=sum(fails), kind=kind, k=sum(noted), b=sum(a and b for a, b in zip(fails, noted)))) + "</div>")
        return yours.figure(w["strip"].format(name=name), f'<div role="img" aria-label="{html.escape(w["alt"].format(kind=kind))}">{body}</div>')
    return picture


def _reg_sum(w: dict, got, shown: str) -> str:
    if not isinstance(got, list):
        return w["reg_odd"].format(items=shown)
    if not got:
        return w["reg_none"]
    return (w["reg_one"] if len(got) == 1 else w["reg_sum"]).format(n=len(got), items=shown)


def regressions_picture(seen) -> str:
    """The inputs that pass every check in version 1 and in version 2, and the ones the learner's function gives."""
    mine = seen.mine
    fn = mine.names.get("regressions") if mine is not None else None
    if not callable(fn):
        return ""
    old, new = run("v1")["all"], run("v2")["all"]
    try:
        got = fn(list(old), list(new))
        marked = [i in got for i in range(len(old))]
    except Exception:
        return ""
    w = _WORDS[seen.lang]
    label = f'style="{_UI};font-size:12px;color:var(--ea-ink-2);margin:0.3rem 0 0.1rem"'
    shown = ", ".join(str(i) for i in got[:12]) + (" …" if len(got) > 12 else "") if isinstance(got, list) else html.escape(repr(got))
    body = (f'<div {label}>{html.escape(w["v1"])}</div>' + _squares(old, fill="var(--ea-data)")
            + f'<div {label}>{html.escape(w["v2"])}</div>' + _squares(new, under=marked, fill="var(--ea-data)")
            + _key([("var(--ea-data)", "var(--ea-line)", w["pass_all"])])
            + f'<div style="{_UI};font-size:13px;color:var(--ea-ink);margin-top:0.35rem">'
            + html.escape(_reg_sum(w, got, shown)) + "</div>")
    return yours.figure(w["reg_title"], f'<div role="img" aria-label="{html.escape(w["reg_alt"])}">{body}</div>')


def idea_picture(lang: str) -> str:
    """An eval set: inputs with what they should be about, a version of the app, checks in code, and the passes; then the
    two versions compared."""
    w = _MORE[lang]
    e = html.escape
    boxes = [f'<span style="{_UI};font-size:14px;color:var(--ea-ink);border:1.4px solid var(--ea-ink-2);border-radius:6px;'
             f'padding:0.35rem 0.8rem">{e(step)}</span>' for step in w["flow"]]
    arrow = f'<span aria-hidden="true" style="{_UI};font-size:16px;color:var(--ea-ink-2)">→</span>'
    row = (f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:0.45rem 0.5rem">{arrow.join(boxes)}</div>'
           f'<div style="{_UI};font-size:13px;color:var(--ea-ink-2);margin-top:0.5rem">⇄ {e(w["compare"])}</div>')
    return yours.figure(w["flow_title"], f'<div role="img" aria-label="{e(w["flow_alt"])}">{row}</div>', alt=w["flow_alt"])


def _table(head: list, rows: list, right_from: int = 1) -> str:
    cell = f"{_UI};font-size:13px;color:var(--ea-ink);padding:0.25rem 0.6rem;border-top:1px solid var(--ea-line)"
    th = f"{_UI};font-size:12px;font-weight:400;color:var(--ea-ink-2);padding:0.2rem 0.6rem"
    out = "<tr>" + "".join(f'<th style="{th};text-align:{"right" if k >= right_from else "left"}">{html.escape(h)}</th>'
                           for k, h in enumerate(head)) + "</tr>"
    for row in rows:
        out += "<tr>" + "".join(f'<td style="{cell};text-align:{"right" if k >= right_from else "left"}">{v}</td>'
                                for k, v in enumerate(row)) + "</tr>"
    return f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;max-width:44rem">{out}</table></div>'


def tested_picture(lang: str) -> str:
    """Each check the course tried, against its notes on version 1: what it fails, catches, misses and fails wrongly."""
    w = _MORE[lang]
    e = html.escape
    rows = []
    for name, kind in FOR_KIND.items():
        r = against_notes(name)
        label = f'<span style="{_MONO}">{e(name)}</span>' + (
            f'<div style="font-size:12px;color:var(--ea-ink-2)">{e(w["out"])}</div>' if name not in CHECKS else "")
        rows.append([label, e(E.kind_name(kind, lang)), len(r["fails"]), len(r["caught"]), len(r["missed"]),
                     f'<b>{len(r["alarms"])}</b>' if r["alarms"] else "0"])
    table = _table([w["check"], w["for"], w["fails"], w["caught"], w["missed"], w["alarms"]], rows, right_from=2)
    return yours.figure(w["tested_title"], table, alt=w["tested_alt"])


def versions_picture(lang: str) -> str:
    """The eval set on both versions: the passes of each check, and of every check together."""
    w = _MORE[lang]
    e = html.escape
    v1, v2 = run("v1"), run("v2")
    rows = [[f'<span style="{_MONO}">{e(name)}</span>', sum(v1[name]), sum(v2[name])] for name in CHECKS]
    rows.append([f'<b>{e(w["every"])}</b>', f'<b>{sum(v1["all"])}</b>', f'<b>{sum(v2["all"])}</b>'])
    return yours.figure(w["versions_title"], _table([w["check"], w["v1"], w["v2"]], rows), alt=w["versions_alt"])


def _marks(text: str, topic: str) -> str:
    out = []
    for name, fn in CHECKS.items():
        ok = fn(text, topic)
        out.append(f'<span style="{_MONO};font-size:12px;color:{"var(--ea-ink)" if ok else "var(--ea-loss-text)"};'
                   f'margin-right:0.8rem;white-space:nowrap">{"✓" if ok else "✗"} {html.escape(name)}</span>')
    return f'<div style="margin-top:0.2rem;line-height:1.7">{"".join(out)}</div>'


def compare_card(i: int, lang: str) -> str:
    """Input i of the eval set: the customer's message with the topic the eval set expects, and the answer of each
    version with what each check says about it."""
    w = _MORE[lang]
    e = html.escape
    rows = (f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2)">{e(w["customer"])} · {e(w["expects"])} '
            f'<span style="{_MONO}">{e(LABELS[i])}</span></div>'
            f'<div style="font-size:14px;color:var(--ea-ink);margin:0.15rem 0 0.5rem">{e(QUESTIONS[i])}</div>')
    for version in ("v1", "v2"):
        text = texts(version)[i]
        rows += (f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2)">{e(w[version])}</div>'
                 f'<pre style="{_MONO};font-size:12px;color:var(--ea-ink);background:var(--ea-inset);padding:0.4rem 0.6rem;'
                 f'margin:0.15rem 0 0.1rem;white-space:pre-wrap;word-break:break-word">{e(text)}</pre>'
                 f'{_marks(text, LABELS[i])}<div style="height:0.5rem"></div>')
    return yours.figure(w["input"].format(i=i + 1, n=len(LABELS)), rows)


def group(which: str) -> list:
    """The numbers of the inputs in a group of group_options."""
    v1, v2 = run("v1")["all"], run("v2")["all"]
    pick = {"fixed": lambda a, b: not a and b, "broke": lambda a, b: a and not b, "both": lambda a, b: a and b,
            "neither": lambda a, b: not a and not b, "all": lambda a, b: True}[which]
    return [i for i, (a, b) in enumerate(zip(v1, v2)) if pick(a, b)]


def group_options(lang: str) -> dict:
    """The groups of inputs to compare in the demo, each with its size; a group without inputs is left out."""
    words = _MORE[lang]["groups"]
    out = {}
    for which, word in zip(("fixed", "broke", "both", "neither", "all"), words):
        if group(which):
            out[word.format(n=len(group(which)))] = which
    return out


# --- exercise 1: the topic check ---------------------------------------------------------------------------------------------

START_TOPIC = '''def topic_ok(text, topic):
    answer = reply_of(text)
    if answer is None:
        return False
    return True
'''
SOLUTION_TOPIC = START_TOPIC.replace("    return True\n", '''    return answer["topic"] == topic
''')
_TOPIC_IN = '''def topic_ok(text, topic):
    return topic in text
'''
_TOPIC_NO_NONE = '''def topic_ok(text, topic):
    answer = reply_of(text)
    return answer["topic"] == topic
'''
_SHORT = '{"topic": "pin_blocked", "reply": "Unblock it in the app under Cards > PIN."}'
_NOT_JSON = "Sure! Your PIN is blocked after 3 wrong tries."
_RATE_FOR_EXCHANGE = '{"topic": "card_payment_wrong_exchange_rate", "reply": "Send the date and amount of the payment."}'


def _type_error(e) -> bool:
    """The TypeError of looking up a key in None, not any TypeError."""
    return isinstance(e, TypeError) and "NoneType" in str(e)


_CLEARED = E._CLEARED

# --- exercise 2: the steps check ---------------------------------------------------------------------------------------------

START_STEPS = START_TOPIC.replace("topic_ok", "steps_ok")
SOLUTION_STEPS = START_STEPS.replace("    return True\n", '''    return "date and amount" not in answer["reply"] or topic in RATE
''')
_STEPS_MODEL = SOLUTION_STEPS.replace("or topic in RATE", 'or answer["topic"] in RATE')
_STEPS_AND = SOLUTION_STEPS.replace('"date and amount" not in answer["reply"] or topic in RATE',
                                    '"date and amount" not in answer["reply"] and topic in RATE')
_STEPS_ONE = SOLUTION_STEPS.replace("or topic in RATE", 'or topic == "card_payment_wrong_exchange_rate"')
_STEPS_ASKS = SOLUTION_STEPS.replace('"date and amount" not in answer["reply"] or topic in RATE',
                                     '"date and amount" not in answer["reply"]')
_REFUND_ASKS = '{"topic": "card_payment_wrong_exchange_rate", "reply": "Please send the date and amount of the payment."}'
_RATE_ASKS = '{"topic": "exchange_rate", "reply": "If the rate looks wrong, send the date and amount of the payment."}'

# --- exercise 3: regressions -------------------------------------------------------------------------------------------------

START_REGRESSIONS = '''def regressions(old, new):
    found = []
    for i in range(len(old)):
        if not new[i]:
            found.append(i)
    return found
'''
SOLUTION_REGRESSIONS = START_REGRESSIONS.replace("        if not new[i]:", "        if old[i] and not new[i]:")
_REG_FIXED = START_REGRESSIONS.replace("        if not new[i]:", "        if new[i] and not old[i]:")
_REG_CHANGED = START_REGRESSIONS.replace("        if not new[i]:", "        if old[i] != new[i]:")
_REG_OLD = START_REGRESSIONS.replace("        if not new[i]:", "        if not old[i]:")


def passed(version: str) -> Named:
    """For each input, whether its answer passes every check. Empty while a version is not recorded yet, so that the
    recorder can load this file before it records version 2."""
    try:
        return Named(run(version)["all"], f"PASSED_{version.upper()}")
    except (OSError, KeyError):
        return Named([], f"PASSED_{version.upper()}")


EXERCISES = {
    "topic_ok": FunctionBox(
        name="topic_ok",
        picture=strip_picture("topic_ok"),
        function="topic_ok",
        start=START_TOPIC,
        solution=SOLUTION_TOPIC,
        names={"reply_of": reply_of},
        visible=[(_SHORT, "pin_blocked")],
        hidden=[(_SHORT, "request_refund"), (_NOT_JSON, "pin_blocked"), (_RATE_FOR_EXCHANGE, "exchange_rate"),
                (texts("v1")[0], LABELS[0]), (texts("v1")[1], LABELS[1])],
        mistakes=[
            Mistake("no-none", error=_type_error,
                    en="With `{call}`, your function stops with a `TypeError`: for a text that is not JSON, `reply_of` gives `None`, and `None` has no `[\"topic\"]`. Give `False` when `answer` is `None`, before you look at its topic.",
                    nl="Met `{call}` stopt je functie met een `TypeError`: voor een tekst die geen JSON is, geeft `reply_of` `None`, en `None` heeft geen `[\"topic\"]`. Geef `False` als `answer` `None` is, voordat je naar het onderwerp kijkt."),
            Mistake("in-text", wrong=_TOPIC_IN, state="almost",
                    en="Almost. `{call}` gives `{got}`: `topic in text` is also `True` when the topic is only a piece of another one, such as `exchange_rate` in `card_payment_wrong_exchange_rate`. Compare the topic of the answer with `topic` as a whole: `answer[\"topic\"] == topic`.",
                    nl="Bijna. `{call}` geeft `{got}`: `topic in text` is ook `True` als het onderwerp maar een stuk van een ander is, zoals `exchange_rate` in `card_payment_wrong_exchange_rate`. Vergelijk het onderwerp van het antwoord als geheel met `topic`: `answer[\"topic\"] == topic`."),
        ],
        texts={
            "missing": {"en": "Run the cell with `topic_ok` first.", "nl": "Voer eerst de cel met `topic_ok` uit."},
            "cleared": _CLEARED,
            "empty": {"en": "Your `topic_ok` passes every answer that is JSON, whatever its topic. Compare the topic of the answer with `topic`, the topic the question is about.",
                      "nl": "Je `topic_ok` laat elk antwoord door dat JSON is, wat het onderwerp ook is. Vergelijk het onderwerp van het antwoord met `topic`, het onderwerp waar de vraag over gaat."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: `True` only when the answer is JSON and its topic is `topic`.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: `True` alleen als het antwoord JSON is en het onderwerp `topic` is."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for an answer with another topic, and for a text that is not JSON?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie voor een antwoord met een ander onderwerp, en voor een tekst die geen JSON is?"},
            "no_function": {"en": "Your box does not make a function `topic_ok`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `topic_ok`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `topic_ok` is right in all {n} cases: it fails every answer with another topic than the eval set expects.",
                        "nl": "Je `topic_ok` klopt in alle {n} gevallen: hij keurt elk antwoord af met een ander onderwerp dan de eval set verwacht."},
        },
    ),
    "steps_ok": FunctionBox(
        name="steps_ok",
        picture=strip_picture("steps_ok"),
        function="steps_ok",
        start=START_STEPS,
        solution=SOLUTION_STEPS,
        names={"reply_of": reply_of, "RATE": RATE},
        visible=[(_REFUND_ASKS, "request_refund")],
        hidden=[(_REFUND_ASKS, "card_payment_wrong_exchange_rate"), (_RATE_ASKS, "exchange_rate"), (_SHORT, "pin_blocked"),
                (_NOT_JSON, "pin_blocked"), (texts("v1")[0], LABELS[0]), (texts("v1")[5], LABELS[5])],
        mistakes=[
            Mistake("model-topic", wrong=_STEPS_MODEL,
                    en="`{call}` gives `{got}`: you look at the topic the model chose, and for this refund it chose a wrong exchange rate, so its own steps fit. Look at `topic`, the topic the eval set says the question is about.",
                    nl="`{call}` geeft `{got}`: je kijkt naar het onderwerp dat het model koos, en voor deze terugbetaling koos het een foute wisselkoers, dus passen zijn eigen stappen. Kijk naar `topic`, het onderwerp waarvan de eval set zegt dat de vraag erover gaat."),
            Mistake("one-topic", wrong=_STEPS_ONE, state="almost",
                    en="Almost. `{call}` gives `{got}`: for a question about exchange rates, asking for a payment's date and amount fits too. Allow both topics in `RATE`: `topic in RATE`.",
                    nl="Bijna. `{call}` geeft `{got}`: bij een vraag over wisselkoersen past vragen om de datum en het bedrag van een betaling ook. Sta beide onderwerpen in `RATE` toe: `topic in RATE`."),
            Mistake("asks", wrong=_STEPS_ASKS, state="almost",
                    en="Almost. `{call}` gives `{got}`: you fail every answer that asks for the date and amount, also where it fits. Let it pass when `topic` is in `RATE`.",
                    nl="Bijna. `{call}` geeft `{got}`: je keurt elk antwoord af dat om de datum en het bedrag vraagt, ook waar dat past. Laat het door als `topic` in `RATE` staat."),
            Mistake("and", wrong=_STEPS_AND,
                    en="`{call}` gives `{got}`: with `and`, an answer passes only when it does not ask for the date and amount and the topic is about exchange rates. It should pass when it does not ask for them, or when the topic is about exchange rates: `or`.",
                    nl="`{call}` geeft `{got}`: met `and` komt een antwoord er alleen door als het niet om de datum en het bedrag vraagt en het onderwerp over wisselkoersen gaat. Het moet erdoor komen als het er niet om vraagt, of als het onderwerp over wisselkoersen gaat: `or`."),
        ],
        texts={
            "missing": {"en": "Run the cell with `steps_ok` first.", "nl": "Voer eerst de cel met `steps_ok` uit."},
            "cleared": _CLEARED,
            "empty": {"en": "Your `steps_ok` passes every answer that is JSON. Fail an answer that asks for the date and amount of a payment, unless `topic` is in `RATE`.",
                      "nl": "Je `steps_ok` laat elk antwoord door dat JSON is. Keur een antwoord af dat om de datum en het bedrag van een betaling vraagt, tenzij `topic` in `RATE` staat."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: `False` for an answer that asks for the date and amount while the question is not about exchange rates.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: `False` voor een antwoord dat om de datum en het bedrag vraagt terwijl de vraag niet over wisselkoersen gaat."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function let an answer about exchange rates ask for the date and amount, and does it give `False` for a text that is not JSON?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Laat je functie een antwoord over wisselkoersen om de datum en het bedrag vragen, en geeft hij `False` voor een tekst die geen JSON is?"},
            "no_function": {"en": "Your box does not make a function `steps_ok`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `steps_ok`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `steps_ok` is right in all {n} cases: it fails an answer that asks for what belongs to a wrong exchange rate, where the question is about something else.",
                        "nl": "Je `steps_ok` klopt in alle {n} gevallen: hij keurt een antwoord af dat vraagt om wat bij een foute wisselkoers hoort, terwijl de vraag over iets anders gaat."},
        },
    ),
    "regressions": FunctionBox(
        name="regressions",
        picture=regressions_picture,
        function="regressions",
        start=START_REGRESSIONS,
        solution=SOLUTION_REGRESSIONS,
        visible=[(passed("v1"), passed("v2"))],
        hidden=[([True, True, False, False], [True, False, True, False]), ([False, False], [True, True]),
                ([True, False, True], [False, False, False]), ([], [])],
        mistakes=[
            Mistake("fixed", wrong=_REG_FIXED,
                    en="`{call}` gives `{got}`: those are the inputs version 2 fixed, which failed before and pass now. A regression is the other way round: it passed with `old`, and fails with `new`.",
                    nl="`{call}` geeft `{got}`: dat zijn de inputs die versie 2 verbeterde, die eerst faalden en nu slagen. Een regression is andersom: hij slaagde met `old`, en faalt met `new`."),
            Mistake("changed", wrong=_REG_CHANGED,
                    en="`{call}` gives `{got}`: you take every input whose outcome changed, also the ones that got better. Keep only those that passed with `old` and fail with `new`.",
                    nl="`{call}` geeft `{got}`: je neemt elke input waarvan de uitkomst veranderde, ook die beter werden. Houd alleen die over die slaagden met `old` en falen met `new`."),
            Mistake("old", wrong=_REG_OLD,
                    en="`{call}` gives `{got}`: those are the inputs that failed with `old`. A regression passed with `old`, and fails with `new`.",
                    nl="`{call}` geeft `{got}`: dat zijn de inputs die faalden met `old`. Een regression slaagde met `old`, en faalt met `new`."),
        ],
        texts={
            "missing": {"en": "Run the cell with `regressions` first.", "nl": "Voer eerst de cel met `regressions` uit."},
            "cleared": _CLEARED,
            "empty": {"en": "Your `regressions` gives every input that fails with `new`, also the ones that failed with `old` already. Keep only those that passed with `old`.",
                      "nl": "Je `regressions` geeft elke input die faalt met `new`, ook die al faalden met `old`. Houd alleen die over die slaagden met `old`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the numbers of the inputs that passed with `old` and fail with `new`.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de nummers van de inputs die slaagden met `old` en falen met `new`."},
            "almost": {"en": "Right for the course's two versions. But `{call}` gives `{got}`, and it should give `{want}`.",
                       "nl": "Goed voor de twee versies van de course. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
            "no_function": {"en": "Your box does not make a function `regressions`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `regressions`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `regressions` is right in all {n} cases: the inputs that version 2 broke.",
                        "nl": "Je `regressions` klopt in alle {n} gevallen: de inputs die versie 2 brak."},
        },
    ),
}

REQUIRED = ["topic_ok", "steps_ok", "regressions", "keep-v2"]

CASES = [
    ("topic_ok", START_TOPIC, "empty", None),
    ("topic_ok", "", "empty", None),
    ("topic_ok", SOLUTION_TOPIC, "correct", None),
    ("topic_ok", '''def topic_ok(text, topic):
    answer = reply_of(text)
    return answer is not None and answer["topic"] == topic
''', "correct", None),
    ("topic_ok", _TOPIC_NO_NONE, "error", "no-none"),
    ("topic_ok", _TOPIC_IN, "almost", "in-text"),
    ("topic_ok", SOLUTION_TOPIC.replace('answer["topic"] == topic', 'answer["topic"] != topic'), "wrong", None),
    ("steps_ok", START_STEPS, "empty", None),
    ("steps_ok", "", "empty", None),
    ("steps_ok", SOLUTION_STEPS, "correct", None),
    ("steps_ok", '''def steps_ok(text, topic):
    answer = reply_of(text)
    if answer is None:
        return False
    if topic in RATE:
        return True
    return "date and amount" not in answer["reply"]
''', "correct", None),
    ("steps_ok", _STEPS_MODEL, "wrong", "model-topic"),
    ("steps_ok", _STEPS_ONE, "almost", "one-topic"),
    ("steps_ok", _STEPS_ASKS, "almost", "asks"),
    ("steps_ok", _STEPS_AND, "wrong", "and"),
    ("regressions", START_REGRESSIONS, "empty", None),
    ("regressions", "", "empty", None),
    ("regressions", SOLUTION_REGRESSIONS, "correct", None),
    ("regressions", '''def regressions(old, new):
    return [i for i in range(len(old)) if old[i] and not new[i]]
''', "correct", None),
    ("regressions", _REG_FIXED, "wrong", "fixed"),
    ("regressions", _REG_CHANGED, "wrong", "changed"),
    ("regressions", _REG_OLD, "wrong", "old"),
]
