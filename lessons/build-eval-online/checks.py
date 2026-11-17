# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-eval-online: version 1 and version 2 of the bank's assistant (build-error-analysis and build-eval-set,
data/course-production-qwen.json) after the eval set. The course's notes on the replies of version 2, read with the
rules of build-error-analysis; a week of chats in which the course plays the customers from those notes, as an A/B test
of the two versions; what each chat cost. Three functions to write (the share of thumbs up, the share of tasks done for
a version, the thumbs-down chats that the eval set passed), the pictures."""
import functools
import html
import importlib.util
import random

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake


def _eval_set():
    spec = importlib.util.spec_from_file_location("build_eval_set_checks_for_online", paths.LESSONS / "build-eval-set" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _eval_set()
E = V.E
QUESTIONS, LABELS, Named = V.QUESTIONS, V.LABELS, V.Named
KIND = E.KIND
PRICE_IN, PRICE_OUT = 3, 15   # made-up prices, in dollars per million tokens, as in build-tokens-prices
THUMB = 4                     # in the week, one customer in four clicks a thumb
SEED = 1                      # the coin flips of the week in the lesson; the demo tries 1 to RUNS
RUNS = 20
HELPED_NOT = ("other-problem", "made-up", "elsewhere")   # what a customer notices: a reply with none of these helped them

# The course read every reply of version 2 against its facts with the rules of build-error-analysis (E.RULE), as it did
# for version 1, and wrote down the kinds that apply; a reply without an entry is right. wrong-topic is the topic field,
# which the customer never sees.
_NOTED_V2 = {1: "T", 5: "TM", 7: "M", 9: "T", 10: "TM", 12: "M", 13: "TO", 14: "A", 15: "TE", 16: "M", 17: "TM",
             18: "M", 22: "M", 23: "T", 25: "M", 26: "TO", 28: "TM", 30: "TA", 31: "TE", 32: "T", 35: "TO", 37: "M",
             38: "M", 39: "TOM", 42: "T", 44: "M", 45: "M", 48: "M", 49: "M", 50: "M", 51: "ME", 52: "TM", 56: "TM",
             57: "TO", 58: "TO", 59: "M", 63: "TO", 66: "TM", 68: "T", 70: "TO", 71: "TM", 73: "A", 78: "TOM",
             80: "M", 81: "T", 83: "M", 86: "M", 89: "TM", 90: "T", 92: "TO", 94: "TOM", 98: "M", 99: "T", 101: "M",
             103: "TOM", 104: "TM", 105: "TO", 106: "M", 108: "M", 111: "TME", 112: "TOM", 115: "M", 117: "M",
             118: "TM"}
NOTES = {"v1": [list(k) for k in E.NOTES], "v2": [[E._LETTER[c] for c in _NOTED_V2.get(i, "")] for i in range(len(QUESTIONS))]}


def helped(version: str, i: int) -> bool:
    """Whether reply i of a version helped its customer: the course's note on it has no other problem, nothing made up
    and no sending elsewhere."""
    return not any(k in NOTES[version][i] for k in HELPED_NOT)


def _answers(version: str) -> tuple:
    return E.answers() if version == "v1" else E.answers("build-eval-set", "v2")


@functools.cache
def _week(seed: int) -> tuple:
    """A week of chats: the 120 test questions in their order, each customer given version 1 or version 2 by a coin flip,
    and one in THUMB clicking a thumb, up when the reply helped and down when it did not. As tuples, so that every caller
    gets its own dictionaries."""
    r = random.Random(seed)
    passed = {v: V.run(v)["all"] for v in ("v1", "v2")}
    chats = []
    for i, message in enumerate(QUESTIONS):
        version = r.choice(("v1", "v2"))
        clicks = r.randrange(THUMB) == 0
        a = _answers(version)[i]
        done = helped(version, i)
        chats.append((("message", message), ("version", version), ("reply", V.reply_of(a["text"])["reply"]),
                      ("prompt_tokens", a["prompt_tokens"]), ("output_tokens", a["tokens"]), ("seconds", a["seconds"]),
                      ("thumb", ("up" if done else "down") if clicks else None), ("done", done), ("passed", passed[version][i])))
    return tuple(chats)


def week(seed: int = SEED) -> list:
    return [dict(c) for c in _week(seed)]


WEEK = Named(week(), "WEEK")


def up_share(chats):
    ups = 0
    downs = 0
    for chat in chats:
        if chat["thumb"] == "up":
            ups = ups + 1
        if chat["thumb"] == "down":
            downs = downs + 1
    return ups / (ups + downs)


def done_share(chats, version):
    done = 0
    count = 0
    for chat in chats:
        if chat["version"] == version:
            if chat["done"]:
                done = done + 1
            count = count + 1
    return done / count


def passed_down(chats):
    messages = []
    for chat in chats:
        if chat["thumb"] == "down":
            if chat["passed"]:
                messages.append(chat["message"])
    return messages


def cost(chat: dict) -> float:
    """What a chat cost at the made-up prices, in dollars."""
    return chat["prompt_tokens"] * PRICE_IN / 1_000_000 + chat["output_tokens"] * PRICE_OUT / 1_000_000


def cost_per_1000(version: str) -> float:
    """What 1,000 chats of a version cost, from the average of its 120 replies, in dollars."""
    a = _answers(version)
    return sum(x["prompt_tokens"] * PRICE_IN / 1_000_000 + x["tokens"] * PRICE_OUT / 1_000_000 for x in a) / len(a) * 1000


def kinds(version: str) -> dict:
    return {k: sum(k in n for n in NOTES[version]) for k in E.KINDS}


def findings() -> dict:
    """The numbers the lesson gives: the notes on both versions, the eval set, the week, the twenty coin-flip runs and
    the costs."""
    n = len(QUESTIONS)
    w = week()
    ev = {v: sum(V.run(v)["all"]) for v in ("v1", "v2")}
    helps = {v: sum(helped(v, i) for i in range(n)) for v in ("v1", "v2")}
    runs = []
    for seed in range(1, RUNS + 1):
        c = week(seed)
        runs.append({v: done_share(c, v) for v in ("v1", "v2")} | {"n1": sum(x["version"] == "v1" for x in c)})
    thumbs = [x for x in w if x["thumb"] is not None]
    down = passed_down(w)
    down_i = [i for i, x in enumerate(w) if x["thumb"] == "down" and x["passed"]]
    return {
        "n": n,
        "eval": ev,
        "good": {v: sum(not k for k in NOTES[v]) for v in ("v1", "v2")},
        "kinds": {v: kinds(v) for v in ("v1", "v2")},
        "helped": helps,
        "only": {v: sum(helped(v, i) and not helped(o, i) for i in range(n)) for v, o in (("v1", "v2"), ("v2", "v1"))},
        "week_n": {v: sum(x["version"] == v for x in w) for v in ("v1", "v2")},
        "thumbs": len(thumbs), "ups": sum(x["thumb"] == "up" for x in thumbs),
        "up_share": up_share(w), "up_share_v": {v: up_share([x for x in w if x["version"] == v]) for v in ("v1", "v2")},
        "thumbs_v": {v: sum(x["thumb"] is not None for x in w if x["version"] == v) for v in ("v1", "v2")},
        "done": {v: done_share(w, v) for v in ("v1", "v2")},
        "done_all": sum(x["done"] for x in w) / n,
        "runs": runs,
        "v1_ahead": sum(r["v1"] > r["v2"] for r in runs), "v2_ahead": sum(r["v2"] > r["v1"] for r in runs),
        "spread": (min(round((r["v2"] - r["v1"]) * 100) for r in runs), max(round((r["v2"] - r["v1"]) * 100) for r in runs)),
        "passed_down": down, "passed_down_kinds": {k: sum(k in NOTES[w[i]["version"]][i] for i in down_i) for k in E.KINDS},
        "downs": sum(x["thumb"] == "down" for x in w),
        "cost_1000": {v: cost_per_1000(v) for v in ("v1", "v2")},
        "cost_done": {v: cost_per_1000(v) / (helps[v] / n) for v in ("v1", "v2")},
        "seconds": {v: sum(x["seconds"] for x in _answers(v)) / n for v in ("v1", "v2")},
        "example": 0,
    }


# --- pictures ---------------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui)"
_MONO = "font-family:var(--ea-font-code)"
_COLOR = {"v1": "var(--ea-ink-3)", "v2": "var(--ea-data)"}
_WORDS = {
    "en": {"v1": "version 1", "v2": "version 2", "all": "all chats", "pct": "{p}%",
           "up_title": "Your up_share(chats) on the week",
           "up_alt": "For all chats of the week, the chats of version 1 and those of version 2: a bar with the share your function gives, and below it a bar with the share of tasks done, with how many customers clicked a thumb.",
           "up_row": "{t} thumbs of {n} chats", "yours": "your up_share", "done": "tasks done",
           "up_sum": "Thumbs from {t} of the {n} customers; tasks done in {d} of the {n} chats.",
           "done_title": "Your done_share(chats, version) beside the eval set",
           "done_alt": "Two groups of two bars, one for each version: on the left the share of the {n} inputs that pass every check of the eval set, on the right the share of tasks done that your function gives for the week.",
           "offline": "offline: passes the eval set", "online": "online: your tasks done in the week",
           "ahead": "The eval set puts {e} ahead; your tasks done put {d} ahead.", "tie": "no version", "group": "{n} customers",
           "pd_title": "Your passed_down(chats): thumbs down that the eval set passed",
           "pd_alt": "A card for each message your function gives: the customer's message, the version that answered, its reply, and the kinds in the course's note on that reply.",
           "pd_sum": "{n} chats, of which the eval set passed {p}; the course's note says {kind} for {m} of those {p}, and the eval set has no check for that.",
           "pd_none": "Your function gives no chats.", "note": "the course's note", "no_note": "no mistake",
           "kind": "kind", "both": "if every customer got it",
           "loop": ["eval set", "A/B test", "feedback", "read", "new inputs"],
           "loop_sub": ["offline, before users", "some customers get it", "thumbs, tasks done, cost", "the chats that went wrong", "for the eval set"],
           "loop_title": "Offline and online, around and around",
           "loop_alt": "Five steps in a loop: the eval set tests a change offline; an A/B test gives it to some of the customers; their feedback, thumbs, tasks done and cost, is measured; the chats that went wrong are read; and they become new inputs for the eval set, where the loop starts again.",
           "week_title": "The first ten chats of the week",
           "week_head": ["", "message", "version", "thumb", "task done", "eval set"],
           "yes": "yes", "no": "no", "pass": "passes", "fail": "fails", "none": "–",
           "notes_title": "The course's notes on both versions",
           "notes_head": ["", "version 1", "version 2"],
           "notes_rows": {"good": "no mistake", "eval": "passes every check of the eval set", "helped": "helped the customer"},
           "runs_title": "Week {s}: which version came out ahead",
           "runs_alt": "Two bars: the share of tasks done for the customers who got version 1 and for those who got version 2 in this week. Below, a dot for each of the {r} weeks with other coin flips: how many points version 2 did better than version 1, left of the middle line when it did worse.",
           "runs_strip": "version 2 minus version 1, in points, for weeks 1 to {r}",
           "runs_sum": "Version 1 came out ahead in {a} of the {r} weeks, version 2 in {b}. If every customer got version 1, it helped {h1} of the {n}; version 2, {h2}."},
    "nl": {"v1": "versie 1", "v2": "versie 2", "all": "alle chats", "pct": "{p}%",
           "up_title": "Jouw up_share(chats) op de week",
           "up_alt": "Voor alle chats van de week, de chats van versie 1 en die van versie 2: een balk met het aandeel dat je functie geeft, en daaronder een balk met het aandeel taken dat gedaan is, met hoeveel klanten op een duim klikten.",
           "up_row": "{t} duimen van {n} chats", "yours": "jouw up_share", "done": "gedane taken",
           "up_sum": "Duimen van {t} van de {n} klanten; de taak gedaan in {d} van de {n} chats.",
           "done_title": "Jouw done_share(chats, version) naast de eval set",
           "done_alt": "Twee groepen van twee balken, een voor elke versie: links het aandeel van de {n} inputs dat elke check van de eval set doorstaat, rechts het aandeel taken dat gedaan is volgens je functie voor de week.",
           "offline": "offline: doorstaat de eval set", "online": "online: jouw gedane taken in de week",
           "ahead": "Volgens de eval set ligt {e} voor; volgens jouw gedane taken ligt {d} voor.", "tie": "geen van beide", "group": "{n} klanten",
           "pd_title": "Jouw passed_down(chats): duim omlaag, maar door de eval set goedgekeurd",
           "pd_alt": "Een kaart voor elk bericht dat je functie geeft: het bericht van de klant, de versie die antwoordde, het antwoord, en de soorten in de notitie van de course bij dat antwoord.",
           "pd_sum": "{n} chats, waarvan de eval set er {p} doorliet; de notitie van de course zegt {kind} bij {m} van die {p}, en daar heeft de eval set geen check voor.",
           "pd_none": "Je functie geeft geen chats.", "note": "de notitie van de course", "no_note": "geen fout",
           "kind": "soort", "both": "als elke klant hem kreeg",
           "loop": ["eval set", "A/B test", "feedback", "lezen", "nieuwe inputs"],
           "loop_sub": ["offline, vóór gebruikers", "een deel van de klanten", "duimen, taken, kosten", "de chats die misgingen", "voor de eval set"],
           "loop_title": "Offline en online, rond en rond",
           "loop_alt": "Vijf stappen in een kringloop: de eval set test een wijziging offline; een A/B test geeft haar aan een deel van de klanten; hun feedback, duimen, gedane taken en kosten, wordt gemeten; de chats die misgingen worden gelezen; en die worden nieuwe inputs voor de eval set, waar de kringloop opnieuw begint.",
           "week_title": "De eerste tien chats van de week",
           "week_head": ["", "bericht", "versie", "duim", "taak gedaan", "eval set"],
           "yes": "ja", "no": "nee", "pass": "door", "fail": "fout", "none": "–",
           "notes_title": "De notities van de course bij beide versies",
           "notes_head": ["", "versie 1", "versie 2"],
           "notes_rows": {"good": "geen fout", "eval": "doorstaat elke check van de eval set", "helped": "hielp de klant"},
           "runs_title": "Week {s}: welke versie voorlag",
           "runs_alt": "Twee balken: het aandeel taken dat gedaan is voor de klanten die versie 1 kregen en voor die versie 2 kregen in deze week. Daaronder een stip voor elk van de {r} weken met andere muntworpen: hoeveel punten versie 2 het beter deed dan versie 1, links van de middellijn als hij het slechter deed.",
           "runs_strip": "versie 2 min versie 1, in punten, voor week 1 tot en met {r}",
           "runs_sum": "Versie 1 lag voor in {a} van de {r} weken, versie 2 in {b}. Als elke klant versie 1 kreeg, hielp die er {h1} van de {n}; versie 2 {h2}."},
}


def pct(x: float) -> int:
    """A share as a whole percentage, as the lesson writes it: a half rounds up, so 75 of 120 is 63%."""
    return int(x * 100 + 0.5)


def money(x: float, lang: str) -> str:
    """Dollars with two decimals, with a decimal comma in Dutch."""
    text = f"{x:.2f}"
    return text.replace(".", ",") if lang == "nl" else text


def _bar(share, color: str, label: str, width: str = "100%") -> str:
    known = isinstance(share, (int, float)) and not isinstance(share, bool)
    w = max(0.0, min(1.0, float(share))) * 100 if known else 0
    text = f"{pct(share)}%" if known else html.escape(repr(share))
    return (f'<div style="display:flex;align-items:center;gap:0.5rem;margin:0.12rem 0;width:{width}">'
            f'<span style="{_UI};font-size:12px;color:var(--ea-ink-2);flex:0 0 9rem">{html.escape(label)}</span>'
            f'<span style="flex:1;background:var(--ea-inset);border-radius:3px;height:12px;position:relative">'
            f'<span style="position:absolute;left:0;top:0;bottom:0;width:{w:.1f}%;background:{color};border-radius:3px"></span></span>'
            f'<span style="{_UI};font-size:12px;color:var(--ea-ink);flex:0 0 3rem;text-align:right">{text}</span></div>')


def _mine(seen, name: str):
    mine = seen.mine
    fn = mine.names.get(name) if mine is not None else None
    return fn if callable(fn) else None


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None


def up_picture(seen) -> str:
    fn = _mine(seen, "up_share")
    if fn is None:
        return ""
    w = _WORDS[seen.lang]
    e = html.escape
    rows = ""
    for key in ("all", "v1", "v2"):
        chats = WEEK if key == "all" else [c for c in WEEK if c["version"] == key]
        got = _safe(fn, [dict(c) for c in chats])
        t = sum(c["thumb"] is not None for c in chats)
        rows += (f'<div style="margin:0.35rem 0"><div style="{_UI};font-size:12px;color:var(--ea-ink)">{e(w[key])} · '
                 f'{e(w["up_row"].format(t=t, n=len(chats)))}</div>'
                 f'{_bar(got, "var(--ea-data)", w["yours"])}{_bar(sum(c["done"] for c in chats) / len(chats), "var(--ea-ink-3)", w["done"])}</div>')
    t = sum(c["thumb"] is not None for c in WEEK)
    summary = (f'<div style="{_UI};font-size:13px;color:var(--ea-ink);margin-top:0.3rem">'
               f'{e(w["up_sum"].format(t=t, n=len(WEEK), d=sum(c["done"] for c in WEEK)))}</div>')
    return yours.figure(w["up_title"], f'<div role="img" aria-label="{e(w["up_alt"])}">{rows}{summary}</div>')


def done_picture(seen) -> str:
    fn = _mine(seen, "done_share")
    if fn is None:
        return ""
    w = _WORDS[seen.lang]
    e = html.escape
    got = {v: _safe(fn, [dict(c) for c in WEEK], v) for v in ("v1", "v2")}
    ev = {v: sum(V.run(v)["all"]) / len(QUESTIONS) for v in ("v1", "v2")}
    sizes = {v: sum(c["version"] == v for c in WEEK) for v in ("v1", "v2")}
    left = "".join(_bar(ev[v], _COLOR[v], w[v]) for v in ("v1", "v2"))
    right = "".join(_bar(got[v], _COLOR[v], f'{w[v]} ({w["group"].format(n=sizes[v])})') for v in ("v1", "v2"))
    col = "flex:1 1 16rem;min-width:0"
    body = (f'<div style="display:flex;flex-wrap:wrap;gap:1rem">'
            f'<div style="{col}"><div style="{_UI};font-size:12px;color:var(--ea-ink-2);margin-bottom:0.2rem">{e(w["offline"])}</div>{left}</div>'
            f'<div style="{col}"><div style="{_UI};font-size:12px;color:var(--ea-ink-2);margin-bottom:0.2rem">{e(w["online"])}</div>{right}</div></div>')

    def ahead(d):
        a, b = d["v1"], d["v2"]
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a, b)) or a == b:
            return w["tie"]
        return w["v1"] if a > b else w["v2"]

    summary = (f'<div style="{_UI};font-size:13px;color:var(--ea-ink);margin-top:0.4rem">'
               f'{e(w["ahead"].format(e=ahead(ev), d=ahead(got)))}</div>')
    return yours.figure(w["done_title"], f'<div role="img" aria-label="{e(w["done_alt"].format(n=len(QUESTIONS)))}">{body}{summary}</div>')


def _kinds_text(kinds: list, lang: str) -> str:
    return ", ".join(KIND[k][lang] for k in kinds) if kinds else _WORDS[lang]["no_note"]


def passed_down_picture(seen) -> str:
    fn = _mine(seen, "passed_down")
    if fn is None:
        return ""
    w = _WORDS[seen.lang]
    e = html.escape
    got = _safe(fn, [dict(c) for c in WEEK])
    got = got if isinstance(got, list) else []
    cards, made, passed = "", 0, 0
    for message in got:
        i = next((k for k, c in enumerate(WEEK) if c["message"] == message), None)
        if i is None:
            cards += f'<div style="{_MONO};font-size:12px;color:var(--ea-ink);margin:0.3rem 0">{e(repr(message)[:120])}</div>'
            continue
        c = WEEK[i]
        kinds = NOTES[c["version"]][i]
        passed += c["passed"]
        made += c["passed"] and "made-up" in kinds
        cards += (f'<div style="border:1px solid var(--ea-line);border-radius:6px;padding:0.4rem 0.6rem;margin:0.35rem 0">'
                  f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2)">{e(w[c["version"]])}</div>'
                  f'<div style="{_UI};font-size:13px;color:var(--ea-ink);font-weight:600;margin:0.1rem 0">{e(c["message"])}</div>'
                  f'<div style="{_UI};font-size:13px;color:var(--ea-ink)">{e(c["reply"])}</div>'
                  f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2);margin-top:0.2rem">{e(w["note"])}: '
                  f'<span style="color:var(--ea-loss-text)">{e(_kinds_text(kinds, seen.lang))}</span></div></div>')
    if not got:
        cards = f'<div style="{_UI};font-size:13px;color:var(--ea-ink-2)">{e(w["pd_none"])}</div>'
    summary = (f'<div style="{_UI};font-size:13px;color:var(--ea-ink);margin-top:0.3rem">'
               f'{e(w["pd_sum"]).format(n=len(got), p=passed, m=made, kind="<i>" + e(KIND["made-up"][seen.lang]) + "</i>")}</div>') if got else ""
    return yours.figure(w["pd_title"], f'<div role="img" aria-label="{e(w["pd_alt"])}">{cards}{summary}</div>')


def loop_picture(lang: str) -> str:
    """The loop of offline and online evaluation: five steps, and back to the eval set."""
    w = _WORDS[lang]
    e = html.escape
    box_w, gap, h = 118, 22, 52
    parts = []
    for k, (name, sub) in enumerate(zip(w["loop"], w["loop_sub"])):
        x = 6 + k * (box_w + gap)
        parts.append(f'<rect x="{x}" y="8" width="{box_w}" height="{h}" rx="6" style="fill:var(--ea-page);stroke:var(--ea-ink-2);stroke-width:1"/>'
                     f'<text x="{x + box_w / 2}" y="29" text-anchor="middle" style="{_UI};font-size:13px;font-weight:600;fill:var(--ea-ink)">{e(name)}</text>'
                     f'<text x="{x + box_w / 2}" y="47" text-anchor="middle" style="{_UI};font-size:11px;fill:var(--ea-ink-2)">{e(sub)}</text>')
        if k < 4:
            ax = x + box_w + 3
            parts.append(f'<line x1="{ax}" y1="34" x2="{ax + gap - 8}" y2="34" style="stroke:var(--ea-ink-2);stroke-width:1.2"/>'
                         f'<path d="M{ax + gap - 8},30 L{ax + gap - 3},34 L{ax + gap - 8},38 Z" style="fill:var(--ea-ink-2)"/>')
    width = 6 + 5 * box_w + 4 * gap + 6
    last = 6 + 4 * (box_w + gap) + box_w / 2
    first = 6 + box_w / 2
    parts.append(f'<path d="M{last},{8 + h} L{last},{h + 30} L{first},{h + 30} L{first},{8 + h + 6}" '
                 f'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.2"/>'
                 f'<path d="M{first - 4},{8 + h + 8} L{first},{8 + h + 2} L{first + 4},{8 + h + 8} Z" style="fill:var(--ea-ink-2)"/>')
    svg = (f'<div style="overflow-x:auto"><svg viewBox="0 0 {width} {h + 38}" width="100%" style="min-width:560px;max-width:{width}px" '
           f'role="img" aria-label="{e(w["loop_alt"])}">{"".join(parts)}</svg></div>')
    return yours.figure(w["loop_title"], svg, alt=w["loop_alt"])


def _table(head: list, rows: list) -> str:
    cell = f"{_UI};font-size:12px;color:var(--ea-ink);padding:0.25rem 0.5rem;vertical-align:top;border-top:1px solid var(--ea-line)"
    th = f"{_UI};font-size:12px;font-weight:400;color:var(--ea-ink-2);padding:0.2rem 0.5rem;text-align:left"
    out = "<tr>" + "".join(f'<th style="{th}">{html.escape(h)}</th>' for h in head) + "</tr>"
    for row in rows:
        out += "<tr>" + "".join(f'<td style="{cell}">{html.escape(str(x))}</td>' for x in row) + "</tr>"
    return f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%">{out}</table></div>'


def week_picture(lang: str) -> str:
    w = _WORDS[lang]
    rows = []
    for i, c in enumerate(WEEK[:10]):
        rows.append([i, c["message"][:60] + ("…" if len(c["message"]) > 60 else ""), w[c["version"]], c["thumb"] or w["none"],
                     w["yes"] if c["done"] else w["no"], w["pass"] if c["passed"] else w["fail"]])
    return yours.figure(w["week_title"], _table(w["week_head"], rows))


def notes_picture(lang: str) -> str:
    w = _WORDS[lang]
    f = findings()
    n = f["n"]
    rows = [[KIND[k][lang], f["kinds"]["v1"][k], f["kinds"]["v2"][k]] for k in E.KINDS]
    rows.append([w["notes_rows"]["good"], f["good"]["v1"], f["good"]["v2"]])
    rows.append([w["notes_rows"]["eval"], f["eval"]["v1"], f["eval"]["v2"]])
    rows.append([w["notes_rows"]["helped"], f["helped"]["v1"], f["helped"]["v2"]])
    head = [w["kind"] + f" ({n})"] + w["notes_head"][1:]
    return yours.figure(w["notes_title"], _table(head, rows))


def runs_picture(seed: int, lang: str) -> str:
    """The A/B test of the week with coin flips `seed`, and the twenty weeks side by side."""
    w = _WORDS[lang]
    e = html.escape
    f = findings()
    chats = week(seed)
    sizes = {v: sum(c["version"] == v for c in chats) for v in ("v1", "v2")}
    bars = "".join(_bar(done_share(chats, v), _COLOR[v], f'{w[v]} ({w["group"].format(n=sizes[v])})') for v in ("v1", "v2"))
    both = "".join(_bar(f["helped"][v] / f["n"], _COLOR[v], f'{w[v]}, {w["both"]}') for v in ("v1", "v2"))
    diffs = [round((r["v2"] - r["v1"]) * 100) for r in f["runs"]]
    span = max(20, max(abs(d) for d in diffs) + 2)
    left, width, mid = 20, 520, 20 + 260
    dots = ""
    for k, d in enumerate(diffs):
        x = mid + d / span * 260
        on = k + 1 == seed
        dots += (f'<circle cx="{x:.1f}" cy="{22 + (k % 4) * 9}" r="{5.5 if on else 4}" '
                 f'style="fill:{"var(--ea-data)" if on else "var(--ea-ink-3)"};stroke:var(--ea-page);stroke-width:1"/>')
    ticks = ""
    for t in (-span // 10 * 10, 0, span // 10 * 10):
        x = mid + t / span * 260
        ticks += f'<text x="{x:.1f}" y="70" text-anchor="middle" style="{_UI};font-size:11px;fill:var(--ea-ink-2)">{t:+d}</text>' if t else \
                 f'<text x="{x:.1f}" y="70" text-anchor="middle" style="{_UI};font-size:11px;fill:var(--ea-ink-2)">0</text>'
    strip = (f'<svg viewBox="0 0 {left * 2 + width} 92" width="100%" style="max-width:{left * 2 + width}px" aria-hidden="true">'
             f'<line x1="{mid}" y1="10" x2="{mid}" y2="58" style="stroke:var(--ea-ink-2);stroke-width:1"/>{dots}{ticks}'
             f'<text x="{mid}" y="86" text-anchor="middle" style="{_UI};font-size:11px;fill:var(--ea-ink-2)">{e(w["runs_strip"].format(r=RUNS))}</text></svg>')
    summary = (f'<div style="{_UI};font-size:13px;color:var(--ea-ink);margin-top:0.3rem">'
               f'{e(w["runs_sum"].format(a=f["v1_ahead"], b=f["v2_ahead"], r=RUNS, h1=f["helped"]["v1"], h2=f["helped"]["v2"], n=f["n"]))}</div>')
    body = f'<div role="img" aria-label="{e(w["runs_alt"].format(r=RUNS))}">{bars}<div style="margin-top:0.5rem">{both}</div>{strip}{summary}</div>'
    return yours.figure(w["runs_title"].format(s=seed), body)


# --- the exercises --------------------------------------------------------------------------------------------------------

START_UP = '''def up_share(chats):
    ups = 0
    downs = 0
    for chat in chats:
        if chat["thumb"] == "up":
            ups = ups + 1
        if chat["thumb"] == "down":
            downs = downs + 1
    return ups / len(chats)
'''
SOLUTION_UP = START_UP.replace("return ups / len(chats)", "return ups / (ups + downs)")
_UP_DOWNS = START_UP.replace("return ups / len(chats)", "return downs / (ups + downs)")
_UP_RATIO = START_UP.replace("return ups / len(chats)", "return ups / downs")
_UP_ALL = '''def up_share(chats):
    ups = 0
    for chat in chats:
        if chat["thumb"] == "up":
            ups = ups + 1
    return ups / len(chats)
'''

START_DONE = '''def done_share(chats, version):
    done = 0
    count = 0
    for chat in chats:
        if chat["done"]:
            done = done + 1
        count = count + 1
    return done / count
'''
SOLUTION_DONE = START_DONE.replace('''        if chat["done"]:
            done = done + 1
        count = count + 1''', '''        if chat["version"] == version:
            if chat["done"]:
                done = done + 1
            count = count + 1''')
_DONE_COUNT_ALL = START_DONE.replace('''        if chat["done"]:
            done = done + 1''', '''        if chat["version"] == version:
            if chat["done"]:
                done = done + 1''')
_DONE_ALL = '''def done_share(chats, version):
    done = 0
    for chat in chats:
        if chat["done"]:
            done = done + 1
    return done / len(chats)
'''
_DONE_COUNT = SOLUTION_DONE.replace("    return done / count", "    return done")

START_PD = '''def passed_down(chats):
    messages = []
    for chat in chats:
        if chat["thumb"] == "down":
            messages.append(chat["message"])
    return messages
'''
SOLUTION_PD = START_PD.replace('''        if chat["thumb"] == "down":
            messages.append(chat["message"])''', '''        if chat["thumb"] == "down":
            if chat["passed"]:
                messages.append(chat["message"])''')
_PD_PASSED = START_PD.replace('''if chat["thumb"] == "down":''', '''if chat["passed"]:''')
_PD_UP = SOLUTION_PD.replace('''chat["thumb"] == "down"''', '''chat["thumb"] == "up"''')
_PD_CHAT = SOLUTION_PD.replace('''messages.append(chat["message"])''', '''messages.append(chat)''')

_CLEARED = {"en": "Your box is empty. Click *Start over* under the box to get the start code back.",
            "nl": "Je vak is leeg. Klik op *Opnieuw beginnen* onder het vak om de begincode terug te krijgen."}


def _chat(version: str, thumb, done: bool, passed: bool, message: str = "Where is my card?") -> dict:
    return {"message": message, "version": version, "reply": "…", "prompt_tokens": 310, "output_tokens": 40, "seconds": 7.0,
            "thumb": thumb, "done": done, "passed": passed}


_SMALL_A = [_chat("v1", "up", True, True), _chat("v2", None, False, True), _chat("v1", "down", False, False),
            _chat("v2", "up", True, False), _chat("v2", None, True, True)]
_SMALL_B = [_chat("v2", "down", False, True, "Is my PIN blocked?"), _chat("v1", None, True, True),
            _chat("v2", "down", False, False, "My card is lost."), _chat("v1", "up", True, True),
            _chat("v2", "down", False, True, "What rate do you use?"), _chat("v1", None, False, False)]
_SMALL_C = [_chat("v1", "up", True, True), _chat("v2", "up", True, True), _chat("v1", None, False, True),
            _chat("v2", None, False, False)]
WEEK_2 = Named(week(2), "WEEK_2")
WEEK_3 = Named(week(3), "WEEK_3")


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


EXERCISES = {
    "up_share": FunctionBox(
        name="up_share",
        picture=up_picture,
        function="up_share",
        start=START_UP,
        solution=SOLUTION_UP,
        visible=[(WEEK,)],
        hidden=[(WEEK_2,), (WEEK_3,), (_SMALL_A,), (_SMALL_B,), (_SMALL_C,)],
        mistakes=[
            Mistake("all", wrong=_UP_ALL,
                    en="`{call}` gives `{got}`: you divide by all chats, also the ones where the customer clicked nothing. Divide by the chats that got a thumb: `ups + downs`.",
                    nl="`{call}` geeft `{got}`: je deelt door alle chats, ook die waarin de klant nergens op klikte. Deel door de chats met een duim: `ups + downs`."),
            Mistake("downs", wrong=_UP_DOWNS,
                    en="`{call}` gives `{got}`: the share of thumbs down. The question is the share of thumbs up: `ups` on top.",
                    nl="`{call}` geeft `{got}`: het aandeel duimen omlaag. De vraag is het aandeel duimen omhoog: `ups` bovenaan."),
            Mistake("ratio", wrong=_UP_RATIO,
                    en="`{call}` gives `{got}`: thumbs up for each thumb down, which can be more than 1. A share divides by all thumbs: `ups + downs`.",
                    nl="`{call}` geeft `{got}`: duimen omhoog per duim omlaag, en dat kan meer dan 1 zijn. Een aandeel deelt door alle duimen: `ups + downs`."),
        ],
        texts=_texts("up_share",
                     empty={"en": "Your `up_share` divides by all chats, also those where the customer clicked nothing. Divide by the chats that got a thumb.",
                            "nl": "Je `up_share` deelt door alle chats, ook die waarin de klant nergens op klikte. Deel door de chats met een duim."},
                     wrong={"en": "`{call}` gives `{got}`, and it should give `{want}`: the thumbs up divided by all thumbs, up and down.",
                            "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de duimen omhoog gedeeld door alle duimen, omhoog en omlaag."},
                     correct={"en": "Your `up_share` is right in all {n} cases. Compare the two bars in each group of the picture: how far are the thumbs from the tasks done?",
                              "nl": "Je `up_share` klopt in alle {n} gevallen. Vergelijk de twee balken in elke groep van het plaatje: hoe ver liggen de duimen van de taken die gedaan zijn?"}),
    ),
    "done_share": FunctionBox(
        name="done_share",
        picture=done_picture,
        function="done_share",
        start=START_DONE,
        solution=SOLUTION_DONE,
        visible=[(WEEK, "v1")],
        hidden=[(WEEK, "v2"), (WEEK_2, "v1"), (WEEK_3, "v2"), (_SMALL_A, "v1"), (_SMALL_A, "v2"), (_SMALL_B, "v2"), (_SMALL_C, "v1")],
        mistakes=[
            Mistake("all", wrong=_DONE_ALL,
                    en="`{call}` gives `{got}`: the tasks done of all chats, of both versions. Count only the chats that got `version`.",
                    nl="`{call}` geeft `{got}`: de gedane taken van alle chats, van beide versies. Tel alleen de chats die `version` kregen."),
            Mistake("count-all", wrong=_DONE_COUNT_ALL,
                    en="`{call}` gives `{got}`: you count the tasks done of one version, but divide by the chats of both. Put `count = count + 1` inside the `if` on the version too.",
                    nl="`{call}` geeft `{got}`: je telt de gedane taken van één versie, maar deelt door de chats van beide. Zet `count = count + 1` ook binnen de `if` op de versie."),
            Mistake("count", wrong=_DONE_COUNT,
                    en="`{call}` gives `{got}`: the number of tasks done, not the share. Divide by the number of chats of the version: `done / count`.",
                    nl="`{call}` geeft `{got}`: het aantal gedane taken, niet het aandeel. Deel door het aantal chats van de versie: `done / count`."),
        ],
        texts=_texts("done_share",
                     empty={"en": "Your `done_share` counts the chats of both versions. Count only the chats of `version`.",
                            "nl": "Je `done_share` telt de chats van beide versies. Tel alleen de chats van `version`."},
                     wrong={"en": "`{call}` gives `{got}`, and it should give `{want}`: of the chats with that version, the share where the task got done.",
                            "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: van de chats met die versie het aandeel waarin de taak gedaan is."},
                     correct={"en": "Your `done_share` is right in all {n} cases. Look at the picture: does the week agree with the eval set about which version is better?",
                              "nl": "Je `done_share` klopt in alle {n} gevallen. Kijk naar het plaatje: is de week het met de eval set eens over welke versie beter is?"}),
    ),
    "passed_down": FunctionBox(
        name="passed_down",
        picture=passed_down_picture,
        function="passed_down",
        start=START_PD,
        solution=SOLUTION_PD,
        visible=[(WEEK,)],
        hidden=[(WEEK_2,), (WEEK_3,), (_SMALL_A,), (_SMALL_B,), (_SMALL_C,)],
        mistakes=[
            Mistake("passed", wrong=_PD_PASSED,
                    en="`{call}` gives `{got}`: every chat the eval set passed, also those with a thumb up or none. Keep a chat when it got a thumb down and the eval set passed it.",
                    nl="`{call}` geeft `{got}`: elke chat die de eval set doorliet, ook die met een duim omhoog of zonder duim. Houd een chat als hij een duim omlaag kreeg en de eval set hem doorliet."),
            Mistake("up", wrong=_PD_UP,
                    en="`{call}` gives `{got}`: chats with a thumb up. The ones to read are the chats with a thumb down: `chat[\"thumb\"] == \"down\"`.",
                    nl="`{call}` geeft `{got}`: chats met een duim omhoog. De chats om te lezen zijn die met een duim omlaag: `chat[\"thumb\"] == \"down\"`."),
            Mistake("chat", wrong=_PD_CHAT,
                    en="`{call}` gives whole chats. Add only the customer's message: `chat[\"message\"]`.",
                    nl="`{call}` geeft hele chats. Voeg alleen het bericht van de klant toe: `chat[\"message\"]`."),
        ],
        texts=_texts("passed_down",
                     empty={"en": "Your `passed_down` gives every chat with a thumb down. Keep only those the eval set passed.",
                            "nl": "Je `passed_down` geeft elke chat met een duim omlaag. Houd alleen die de eval set doorliet."},
                     wrong={"en": "`{call}` gives `{got}`, and it should give `{want}`: the messages of the chats with a thumb down that the eval set passed, in their order.",
                            "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de berichten van de chats met een duim omlaag die de eval set doorliet, in hun volgorde."},
                     correct={"en": "Your `passed_down` is right in all {n} cases. Read the cards: what did the eval set miss?",
                              "nl": "Je `passed_down` klopt in alle {n} gevallen. Lees de kaarten: wat miste de eval set?"}),
    ),
}

REQUIRED = ["up_share", "done_share", "passed_down", "worth"]

CASES = [
    ("up_share", START_UP, "empty", None),
    ("up_share", "", "empty", None),
    ("up_share", SOLUTION_UP, "correct", None),
    ("up_share", '''def up_share(chats):
    thumbs = [c["thumb"] for c in chats if c["thumb"] is not None]
    return thumbs.count("up") / len(thumbs)
''', "correct", None),
    ("up_share", _UP_ALL, "wrong", "all"),
    ("up_share", _UP_DOWNS, "wrong", "downs"),
    ("up_share", _UP_RATIO, "wrong", "ratio"),
    ("done_share", START_DONE, "empty", None),
    ("done_share", "", "empty", None),
    ("done_share", SOLUTION_DONE, "correct", None),
    ("done_share", '''def done_share(chats, version):
    mine = [c["done"] for c in chats if c["version"] == version]
    return sum(mine) / len(mine)
''', "correct", None),
    ("done_share", _DONE_ALL, "wrong", "all"),
    ("done_share", _DONE_COUNT_ALL, "wrong", "count-all"),
    ("done_share", _DONE_COUNT, "wrong", "count"),
    ("passed_down", START_PD, "empty", None),
    ("passed_down", "", "empty", None),
    ("passed_down", SOLUTION_PD, "correct", None),
    ("passed_down", '''def passed_down(chats):
    return [c["message"] for c in chats if c["thumb"] == "down" and c["passed"]]
''', "correct", None),
    ("passed_down", _PD_PASSED, "wrong", "passed"),
    ("passed_down", _PD_UP, "wrong", "up"),
    ("passed_down", _PD_CHAT, "wrong", "chat"),
]
