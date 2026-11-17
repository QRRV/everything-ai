# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-judges: who grades an answer. The 120 replies of version 1 of the bank's assistant
(build-error-analysis), graded by code (the checks of build-eval-set), by people (the course's notes) and by Qwen3 as a
judge (data/course-production-qwen.json): once asked plainly whether a reply is good, once with a rubric for one
criterion, whether the reply says anything about the bank that the facts do not say. How often each judge agrees with
the course's notes; three functions to write (read a verdict, the agreement, how many of people's fails the judge
catches), the pictures."""
import functools
import html
import importlib.util
import json
import re

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake


def _lesson_checks(lesson: str):
    spec = importlib.util.spec_from_file_location(lesson.replace("-", "_") + "_checks", paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _lesson_checks("build-error-analysis")
DATA = E.DATA
QUESTIONS, LABELS, NOTES, Named = E.QUESTIONS, E.LABELS, E.NOTES, E.Named
PLAIN = ("You grade a reply of an online bank's assistant to a customer. The assistant may use only these facts:\n"
         + E.FACTS + "\nIs the reply good? Answer with one word: pass or fail.")
RUBRIC = ("You grade a reply of an online bank's assistant to a customer. The assistant may use only these facts:\n"
          + E.FACTS + "\n"
          "Grade one thing: does the reply say anything about the bank, its cards or its app that the facts do not say?\n"
          "fail: it does, such as a time, a fee, a step or a feature that is not in the facts, or the opposite of a fact.\n"
          "pass: everything it says about the bank, its cards and its app is in the facts. Whether the reply fits the "
          "question does not matter here.\n"
          "Write one sentence on what you checked, then pass or fail on a line of its own.")
JUDGES = {"plain": PLAIN, "rubric": RUBRIC}


def judged() -> list:
    """What each judge reads: the customer's message and the reply of version 1."""
    try:
        answers = E.answers()
    except (OSError, KeyError):
        return []
    return [f"Customer: {q}\nReply: {json.loads(a['text'])['reply']}" for q, a in zip(QUESTIONS, answers)]


RECORD = {"plain": {"each": judged(), "system": PLAIN, "max_tokens": 10},
          "rubric": {"each": judged(), "system": RUBRIC, "max_tokens": 100}}


@functools.cache
def judgments(judge: str) -> tuple:
    """What a judge wrote about each reply of version 1."""
    return tuple(a["text"] for a in E.answers("build-judges", judge))


def words(text) -> list:
    """The words of a text in small letters, without punctuation: "Pass. All facts" gives ["pass", "all", "facts"]."""
    return re.findall(r"[a-z]+", text.lower())


def verdict(text):
    """A judge's verdict: True for pass, False for fail, from the first word that is one of the two; None when neither
    comes."""
    for word in words(text):
        if word == "pass":
            return True
        if word == "fail":
            return False
    return None


def agreement(judge, people):
    same = 0
    for i in range(len(people)):
        if judge[i] == people[i]:
            same = same + 1
    return same / len(people)


def caught(judge, people):
    """Of the replies people failed, the share the judge failed too."""
    failed = 0
    both = 0
    for i in range(len(people)):
        if people[i] == False:
            failed = failed + 1
            if judge[i] == False:
                both = both + 1
    return both / failed


def passed_too(judge, people):
    """Of the replies people passed, the share the judge passed too."""
    return caught([None if v is None else not v for v in judge], [not v for v in people])


def people(criterion: str) -> list:
    """The course's notes as verdicts: True for a pass. "good": no kind at all; otherwise a pass unless the reply has
    that kind."""
    if criterion == "good":
        return [not k for k in NOTES]
    return [criterion not in k for k in NOTES]


CRITERION = {"plain": "good", "rubric": "made-up"}   # what each judge grades, and so which notes it is measured against
PEOPLE_GOOD = Named(people("good"), "PEOPLE_GOOD")
PEOPLE_MADE_UP = Named(people("made-up"), "PEOPLE_MADE_UP")


def verdicts(judge: str) -> list:
    return [verdict(t) for t in judgments(judge)]


def _recorded_verdicts(judge: str) -> list:
    """The verdicts of a judge, empty while it is not recorded yet, so that the recorder can load this file first."""
    try:
        return verdicts(judge)
    except (OSError, KeyError):
        return []


JUDGE_PLAIN = Named(_recorded_verdicts("plain"), "JUDGE_PLAIN")
JUDGE_RUBRIC = Named(_recorded_verdicts("rubric"), "JUDGE_RUBRIC")


def findings() -> dict:
    out = {"n": len(QUESTIONS)}
    for judge, criterion in CRITERION.items():
        v, p = verdicts(judge), people(criterion)
        fails = [i for i in range(len(p)) if p[i] is False]
        passes = [i for i in range(len(p)) if p[i] is True]
        out[judge] = {"pass": sum(x is True for x in v), "fail": sum(x is False for x in v), "none": sum(x is None for x in v),
                      "agree": agreement(v, p), "same": sum(v[i] == p[i] for i in range(len(p))),
                      "caught": sum(v[i] is False for i in fails), "people_fail": len(fails),
                      "passed": sum(v[i] is True for i in passes), "people_pass": len(passes),
                      "not_last": sum(words(t)[-1:] not in (["pass"], ["fail"]) for t in judgments(judge)),
                      "texts": len(set(judgments(judge)))}
    return out


# --- pictures ---------------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui)"
_MONO = "font-family:var(--ea-font-code)"
_WORDS = {
    "en": {"plain": "plain judge", "rubric": "judge with a rubric", "pass": "pass", "fail": "fail", "none": "no verdict",
           "verdict_title": "Your verdict on what the judges wrote about the 120 replies",
           "verdict_alt": "For each judge a row of 120 small squares, one for each reply, with the verdict your function reads from what the judge wrote: an outlined square for pass, a filled square for fail, a gray square for no verdict. Under each row, the counts.",
           "count": "{p} pass, {f} fail, {n} no verdict",
           "agree_title": "Your agreement for each judge, against the course's notes",
           "agree_alt": "A bar for each judge: the share of the 120 replies where its verdict is the same as the course's, as your function computes it.",
           "against": "against {what}", "good": "a reply without any mistake", "made-up": "a reply that makes nothing up",
           "caught_title": "Your caught for each judge, beside the share of good replies it passed",
           "caught_alt": "For each judge two bars: the share of the replies the course failed that the judge failed too, as your function computes it, and the share of the replies the course passed that the judge passed too.",
           "caught": "caught: of the replies the course failed", "through": "passed too: of the replies the course passed",
           "customer": "customer", "reply": "the reply of version 1", "notes": "the course's notes", "said": "the judge wrote",
           "fine": "no mistake found", "read": "the verdict", "card": "Reply {i} of {n}"},
    "nl": {"plain": "eenvoudige judge", "rubric": "judge met een rubric", "pass": "pass", "fail": "fail", "none": "geen oordeel",
           "verdict_title": "Jouw verdict op wat de judges over de 120 antwoorden schreven",
           "verdict_alt": "Voor elke judge een rij van 120 kleine vierkantjes, één voor elk antwoord, met het oordeel dat je functie leest uit wat de judge schreef: een open vierkantje voor pass, een gevuld vierkantje voor fail, een grijs vierkantje voor geen oordeel. Onder elke rij de aantallen.",
           "count": "{p} pass, {f} fail, {n} geen oordeel",
           "agree_title": "Jouw agreement voor elke judge, naast de notities van de course",
           "agree_alt": "Een balk voor elke judge: het deel van de 120 antwoorden waarbij zijn oordeel hetzelfde is als dat van de course, zoals je functie het berekent.",
           "against": "naast {what}", "good": "een antwoord zonder enige fout", "made-up": "een antwoord dat niets verzint",
           "caught_title": "Jouw caught voor elke judge, naast het deel van de goede antwoorden dat hij goedkeurde",
           "caught_alt": "Voor elke judge twee balken: het deel van de antwoorden die de course afkeurde dat de judge ook afkeurde, zoals je functie het berekent, en het deel van de antwoorden die de course goedkeurde dat de judge ook goedkeurde.",
           "caught": "caught: van de antwoorden die de course afkeurde", "through": "ook goedgekeurd: van de antwoorden die de course goedkeurde",
           "customer": "klant", "reply": "het antwoord van versie 1", "notes": "de notities van de course", "said": "de judge schreef",
           "fine": "geen fout gevonden", "read": "het oordeel", "card": "Antwoord {i} van {n}"},
}


def _strip(marks: list) -> str:
    size, gap, per_row = 11, 3, 40
    rows = (len(marks) + per_row - 1) // per_row
    parts = []
    for i, v in enumerate(marks):
        x, y = (i % per_row) * (size + gap) + 2, (i // per_row) * (size + gap) + 2
        fill = "var(--ea-loss)" if v is False else ("var(--ea-ink-3)" if v is None else "var(--ea-page)")
        parts.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="2" style="fill:{fill};stroke:var(--ea-ink-2);stroke-width:1"/>')
    w, h = per_row * (size + gap) + 4, rows * (size + gap) + 4
    return f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" aria-hidden="true">{"".join(parts)}</svg>'


def _bar(label: str, share, color: str) -> str:
    known = isinstance(share, (int, float)) and not isinstance(share, bool)
    width = max(0.0, min(1.0, float(share))) * 100 if known else 0
    text = f"{share:.0%}" if known else html.escape(repr(share))
    return (f'<div style="display:flex;align-items:center;gap:0.6rem;margin:0.25rem 0">'
            f'<span style="{_UI};font-size:12px;color:var(--ea-ink-2);flex:0 0 16rem;max-width:45%">{html.escape(label)}</span>'
            f'<span style="flex:1;background:var(--ea-inset);border-radius:3px;height:14px;position:relative">'
            f'<span style="position:absolute;left:0;top:0;bottom:0;width:{width:.1f}%;background:{color};border-radius:3px"></span></span>'
            f'<span style="{_UI};font-size:13px;color:var(--ea-ink);flex:0 0 3.2rem;text-align:right">{text}</span></div>')


def _mine(seen, name: str):
    mine = seen.mine
    fn = mine.names.get(name) if mine is not None else None
    return fn if callable(fn) else None


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None


def verdict_picture(seen) -> str:
    """What the learner's verdict reads from each judgment of both judges."""
    fn = _mine(seen, "verdict")
    if fn is None:
        return ""
    w = _WORDS[seen.lang]
    body = ""
    for judge in CRITERION:
        marks = [_safe(fn, t) for t in judgments(judge)]
        marks = [m if m in (True, False) else None for m in marks]
        body += (f'<div style="{_UI};font-size:13px;color:var(--ea-ink);margin:0.4rem 0 0.15rem">{html.escape(w[judge])}</div>'
                 + _strip(marks)
                 + f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2)">'
                 + html.escape(w["count"].format(p=sum(m is True for m in marks), f=sum(m is False for m in marks), n=sum(m is None for m in marks)))
                 + "</div>")
    return yours.figure(w["verdict_title"], f'<div role="img" aria-label="{html.escape(w["verdict_alt"])}">{body}</div>')


def agreement_picture(seen) -> str:
    """The learner's agreement for each judge, against the notes for what it grades."""
    fn = _mine(seen, "agreement")
    if fn is None:
        return ""
    w = _WORDS[seen.lang]
    body = "".join(_bar(f'{w[judge]}, {w["against"].format(what=w[criterion])}', _safe(fn, verdicts(judge), people(criterion)),
                        "var(--ea-data)") for judge, criterion in CRITERION.items())
    return yours.figure(w["agree_title"], f'<div role="img" aria-label="{html.escape(w["agree_alt"])}">{body}</div>')


def caught_picture(seen) -> str:
    """The learner's caught for each judge, beside the share of good replies it passed."""
    fn = _mine(seen, "caught")
    if fn is None:
        return ""
    w = _WORDS[seen.lang]
    body = ""
    for judge, criterion in CRITERION.items():
        v, p = verdicts(judge), people(criterion)
        body += (f'<div style="{_UI};font-size:13px;color:var(--ea-ink);margin:0.5rem 0 0.1rem">{html.escape(w[judge])}</div>'
                 + _bar(w["caught"], _safe(fn, v, p), "var(--ea-loss)") + _bar(w["through"], passed_too(v, p), "var(--ea-data)"))
    return yours.figure(w["caught_title"], f'<div role="img" aria-label="{html.escape(w["caught_alt"])}">{body}</div>')


def judged_card(judge: str, i: int, lang: str) -> str:
    """Reply i with the course's notes, what a judge wrote about it, and the verdict read from that."""
    w = _WORDS[lang]
    e = html.escape
    reply = json.loads(E.answers()[i]["text"])["reply"]
    kinds = NOTES[i]
    chips = "".join(E._chip(k, lang) for k in kinds) if kinds else f'<span style="{_UI};font-size:12px;color:var(--ea-ink-2)">{e(w["fine"])}</span>'
    v = verdict(judgments(judge)[i])
    shown = w["pass"] if v is True else (w["fail"] if v is False else w["none"])
    rows = (f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2)">{e(w["customer"])}</div>'
            f'<div style="font-size:14px;color:var(--ea-ink);margin:0.15rem 0 0.45rem">{e(QUESTIONS[i])}</div>'
            f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2)">{e(w["reply"])}</div>'
            f'<div style="font-size:14px;color:var(--ea-ink);margin:0.15rem 0 0.45rem">{e(reply)}</div>'
            f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2)">{e(w["notes"])}</div>'
            f'<div style="margin:0.2rem 0 0.45rem">{chips}</div>'
            f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2)">{e(w["said"])} ({e(w[judge])})</div>'
            f'<pre style="{_MONO};font-size:12px;color:var(--ea-ink);background:var(--ea-inset);padding:0.4rem 0.6rem;'
            f'margin:0.15rem 0 0.3rem;white-space:pre-wrap;word-break:break-word">{e(judgments(judge)[i])}</pre>'
            f'<div style="{_UI};font-size:13px;color:var(--ea-ink)">{e(w["read"])}: <b>{e(shown)}</b></div>')
    return yours.figure(w["card"].format(i=i + 1, n=len(QUESTIONS)), rows)


_TABLES = {
    "en": {"graders_title": "Three ways to grade an answer",
           "graders_alt": "A table with a row for code, people and a model as a judge: what each sees, how fast it is, and what you check it against.",
           "head": ["", "sees", "speed", "check it against"],
           "rows": [["code", "the form, words, what you expect in advance", "fast, the same every time", "your notes, as in the last lesson"],
                    ["people", "what an answer means", "slow", "a rubric, so that two people grade alike"],
                    ["LLM as a judge", "what an answer means, if it is good enough", "fast", "what people graded"]],
           "measured_title": "Qwen3 as a judge on the 120 replies of version 1, against the course's notes",
           "measured_alt": "A table with a row for each judge: what it grades, how often it wrote pass and fail, its agreement with the course, the share of the replies the course failed that it failed too, and the share of those the course passed that it passed too.",
           "mhead": ["judge", "grades", "pass", "fail", "agreement", "caught", "passed too"],
           "grades": {"good": "is the reply good?", "made-up": "does it make anything up?"}},
    "nl": {"graders_title": "Drie manieren om een antwoord te beoordelen",
           "graders_alt": "Een tabel met een rij voor code, mensen en een model als judge: wat elk ziet, hoe snel het is, en waartegen je het controleert.",
           "head": ["", "ziet", "snelheid", "controleer het tegen"],
           "rows": [["code", "de vorm, woorden, wat je van tevoren verwacht", "snel, elke keer hetzelfde", "je notities, zoals in de vorige les"],
                    ["mensen", "wat een antwoord betekent", "langzaam", "een rubric, zodat twee mensen gelijk beoordelen"],
                    ["LLM as a judge", "wat een antwoord betekent, als hij goed genoeg is", "snel", "wat mensen beoordeelden"]],
           "measured_title": "Qwen3 als judge op de 120 antwoorden van versie 1, naast de notities van de course",
           "measured_alt": "Een tabel met een rij voor elke judge: wat hij beoordeelt, hoe vaak hij pass en fail schreef, zijn agreement met de course, het deel van de antwoorden die de course afkeurde dat hij ook afkeurde, en het deel van de antwoorden die de course goedkeurde dat hij ook goedkeurde.",
           "mhead": ["judge", "beoordeelt", "pass", "fail", "agreement", "caught", "ook goedgekeurd"],
           "grades": {"good": "is het antwoord goed?", "made-up": "verzint het iets?"}},
}


def _table(head: list, rows: list, right_from: int = 99) -> str:
    cell = f"{_UI};font-size:13px;color:var(--ea-ink);padding:0.25rem 0.6rem;vertical-align:top;border-top:1px solid var(--ea-line)"
    th = f"{_UI};font-size:12px;font-weight:400;color:var(--ea-ink-2);padding:0.2rem 0.6rem"
    out = "<tr>" + "".join(f'<th style="{th};text-align:{"right" if k >= right_from else "left"}">{html.escape(h)}</th>'
                           for k, h in enumerate(head)) + "</tr>"
    for row in rows:
        out += "<tr>" + "".join(f'<td style="{cell};text-align:{"right" if k >= right_from else "left"}">{v}</td>'
                                for k, v in enumerate(row)) + "</tr>"
    return f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;max-width:46rem">{out}</table></div>'


def graders_picture(lang: str) -> str:
    """Code, people and a judge: what each sees, how fast it is, what to check it against."""
    t = _TABLES[lang]
    rows = [[f"<b>{html.escape(r[0])}</b>"] + [html.escape(c) for c in r[1:]] for r in t["rows"]]
    return yours.figure(t["graders_title"], _table(t["head"], rows), alt=t["graders_alt"])


def measured_picture(lang: str) -> str:
    """Both judges against the course's notes: their verdicts, agreement, caught and passed too."""
    t, w, f = _TABLES[lang], _WORDS[lang], findings()
    rows = []
    for judge, criterion in CRITERION.items():
        m = f[judge]
        rows.append([html.escape(w[judge]), html.escape(t["grades"][criterion]), m["pass"], m["fail"], f"{m['agree']:.0%}",
                     f"{m['caught']} / {m['people_fail']}", f"{m['passed']} / {m['people_pass']}"])
    return yours.figure(t["measured_title"], _table(t["mhead"], rows, right_from=2), alt=t["measured_alt"])


# --- exercise 1: read a verdict --------------------------------------------------------------------------------------------

START_VERDICT = '''def verdict(text):
    if text == "pass":
        return True
    if text == "fail":
        return False
    return None
'''
SOLUTION_VERDICT = '''def verdict(text):
    for word in words(text):
        if word == "pass":
            return True
        if word == "fail":
            return False
    return None
'''
_VERDICT_EARLY = SOLUTION_VERDICT.replace('''            return False
    return None''', '''            return False
        return None''')
_VERDICT_LAST = '''def verdict(text):
    last = words(text)[-1]
    if last == "pass":
        return True
    if last == "fail":
        return False
    return None
'''
_RUBRIC_TEXT = "pass. All facts are included in the reply."
_CLEARED = E._CLEARED

# --- exercise 2: agreement -------------------------------------------------------------------------------------------------

START_AGREEMENT = '''def agreement(judge, people):
    same = 0
    for i in range(len(people)):
        same = same + 1
    return same / len(people)
'''
SOLUTION_AGREEMENT = START_AGREEMENT.replace('''        same = same + 1''', '''        if judge[i] == people[i]:
            same = same + 1''')
_AGREEMENT_COUNT = SOLUTION_AGREEMENT.replace("return same / len(people)", "return same")
_AGREEMENT_FAILS = SOLUTION_AGREEMENT.replace("if judge[i] == people[i]:", "if judge[i] == False and people[i] == False:")

# --- exercise 3: caught ----------------------------------------------------------------------------------------------------

START_CAUGHT = '''def caught(judge, people):
    failed = 0
    both = 0
    for i in range(len(people)):
        if people[i] == False:
            failed = failed + 1
    return both / failed
'''
SOLUTION_CAUGHT = START_CAUGHT.replace('''            failed = failed + 1''', '''            failed = failed + 1
            if judge[i] == False:
                both = both + 1''')
_CAUGHT_ALL = START_CAUGHT.replace('''            failed = failed + 1''', '''            failed = failed + 1
        if judge[i] == False:
            both = both + 1''')
_CAUGHT_NOT = SOLUTION_CAUGHT.replace("if judge[i] == False:", "if not judge[i]:")
_CAUGHT_COUNT = SOLUTION_CAUGHT.replace("return both / failed", "return both")


EXERCISES = {
    "verdict": FunctionBox(
        name="verdict",
        picture=verdict_picture,
        function="verdict",
        start=START_VERDICT,
        solution=SOLUTION_VERDICT,
        names={"words": words},
        visible=[(_RUBRIC_TEXT,)],
        hidden=[("pass",), ("fail",), ("Pass.",), ("The reply makes up a fee.\nfail",), ("fail: the facts give no 3 days.",),
                ("I cannot tell.",), ("",)],
        texts={
            "missing": {"en": "Run the cell with `verdict` first.", "nl": "Voer eerst de cel met `verdict` uit."},
            "cleared": _CLEARED,
            "unreached": {"en": "With `{call}`, your function gives `None`, as if the judge wrote no verdict, and it did. Does your function look at every word in turn, and does it give `None` only after the loop, when no word was pass or fail?",
                          "nl": "Met `{call}` geeft je functie `None`, alsof de judge geen oordeel schreef, en dat deed hij wel. Bekijkt je functie elk woord om de beurt, en geeft hij pas na de loop `None`, als geen enkel woord pass of fail was?"},
            "empty": {"en": "Your `verdict` reads only a text that is exactly `pass` or `fail`. A judge often writes more: go through `words(text)`, and give the verdict of the first word that is pass or fail.",
                      "nl": "Je `verdict` leest alleen een tekst die precies `pass` of `fail` is. Een judge schrijft vaak meer: loop door `words(text)`, en geef het oordeel van het eerste woord dat pass of fail is."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: `True` when the first of pass and fail in the text is pass, `False` when it is fail, `None` when neither comes.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: `True` als de eerste van pass en fail in de tekst pass is, `False` als het fail is, `None` als geen van beide voorkomt."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function read a verdict that comes after a sentence, and give `None` for a text without one?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Leest je functie een oordeel dat na een zin komt, en geeft hij `None` voor een tekst zonder oordeel?"},
            "no_function": {"en": "Your box does not make a function `verdict`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `verdict`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `verdict` is right in all {n} cases: it reads the verdict wherever the judge wrote it.",
                        "nl": "Je `verdict` klopt in alle {n} gevallen: hij leest het oordeel waar de judge het ook schreef."},
        },
    ),
    "agreement": FunctionBox(
        name="agreement",
        picture=agreement_picture,
        function="agreement",
        start=START_AGREEMENT,
        solution=SOLUTION_AGREEMENT,
        visible=[(JUDGE_PLAIN, PEOPLE_GOOD)],
        hidden=[(JUDGE_RUBRIC, PEOPLE_MADE_UP), ([True, False, None], [True, True, False]), ([False, False], [False, True]),
                ([True], [True])],
        mistakes=[
            Mistake("count", wrong=_AGREEMENT_COUNT,
                    en="`{call}` gives `{got}`: that is the number of replies where the two agree. Agreement is a share: divide by the number of replies, `len(people)`.",
                    nl="`{call}` geeft `{got}`: dat is het aantal antwoorden waarbij de twee het eens zijn. Agreement is een deel: deel door het aantal antwoorden, `len(people)`."),
            Mistake("fails", wrong=_AGREEMENT_FAILS,
                    en="`{call}` gives `{got}`: you count only the replies both failed. The two also agree where both passed: compare `judge[i] == people[i]`.",
                    nl="`{call}` geeft `{got}`: je telt alleen de antwoorden die allebei afkeurden. De twee zijn het ook eens waar allebei goedkeurden: vergelijk `judge[i] == people[i]`."),
        ],
        texts={
            "missing": {"en": "Run the cell with `agreement` first.", "nl": "Voer eerst de cel met `agreement` uit."},
            "cleared": _CLEARED,
            "empty": {"en": "Your `agreement` counts every reply, so it always gives 1.0. Count a reply only when the judge's verdict is the same as the course's.",
                      "nl": "Je `agreement` telt elk antwoord, dus geeft hij altijd 1.0. Tel een antwoord alleen als het oordeel van de judge hetzelfde is als dat van de course."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the share of the replies where `judge` and `people` have the same verdict.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: het deel van de antwoorden waarbij `judge` en `people` hetzelfde oordeel hebben."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`.",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
            "no_function": {"en": "Your box does not make a function `agreement`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `agreement`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `agreement` is right in all {n} cases. Look at the picture: which judge agrees more often with the course?",
                        "nl": "Je `agreement` klopt in alle {n} gevallen. Kijk naar het plaatje: welke judge is het vaker eens met de course?"},
        },
    ),
    "caught": FunctionBox(
        name="caught",
        picture=caught_picture,
        function="caught",
        start=START_CAUGHT,
        solution=SOLUTION_CAUGHT,
        visible=[(JUDGE_PLAIN, PEOPLE_GOOD)],
        hidden=[(JUDGE_RUBRIC, PEOPLE_MADE_UP), ([False, None, True, False], [False, False, False, True]), ([None], [False]),
                ([False, False], [False, True])],
        mistakes=[
            Mistake("all", wrong=_CAUGHT_ALL,
                    en="`{call}` gives `{got}`: you count every reply the judge failed, also the ones the course passed. Count a fail of the judge only inside the `if`, for the replies the course failed.",
                    nl="`{call}` geeft `{got}`: je telt elk antwoord dat de judge afkeurde, ook die de course goedkeurde. Tel een fail van de judge alleen binnen de `if`, voor de antwoorden die de course afkeurde."),
            Mistake("not", wrong=_CAUGHT_NOT, state="almost",
                    en="Almost. `{call}` gives `{got}`: `not judge[i]` is also `True` for `None`, a judgment without a verdict. Count only a real fail: `judge[i] == False`.",
                    nl="Bijna. `{call}` geeft `{got}`: `not judge[i]` is ook `True` voor `None`, een beoordeling zonder oordeel. Tel alleen een echte fail: `judge[i] == False`."),
            Mistake("count", wrong=_CAUGHT_COUNT,
                    en="`{call}` gives `{got}`: that is a number of replies. Give the share: divide `both` by `failed`.",
                    nl="`{call}` geeft `{got}`: dat is een aantal antwoorden. Geef het deel: deel `both` door `failed`."),
        ],
        texts={
            "missing": {"en": "Run the cell with `caught` first.", "nl": "Voer eerst de cel met `caught` uit."},
            "cleared": _CLEARED,
            "empty": {"en": "Your `caught` never counts `both`, so it always gives 0.0. Inside the `if`, count the reply in `both` when the judge failed it too.",
                      "nl": "Je `caught` telt `both` nooit, dus geeft hij altijd 0.0. Tel het antwoord binnen de `if` bij `both` als de judge het ook afkeurde."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: of the replies the course failed, the share the judge failed too.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: van de antwoorden die de course afkeurde, het deel dat de judge ook afkeurde."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function count a judgment without a verdict as a fail?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Telt je functie een beoordeling zonder oordeel als een fail?"},
            "no_function": {"en": "Your box does not make a function `caught`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `caught`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `caught` is right in all {n} cases. Read it beside the share of good replies each judge passed: one of the two alone does not tell you whether a judge is any good.",
                        "nl": "Je `caught` klopt in alle {n} gevallen. Lees hem naast het deel van de goede antwoorden dat elke judge goedkeurde: één van de twee alleen zegt niet of een judge iets waard is."},
        },
    ),
}

REQUIRED = ["verdict", "agreement", "caught", "what-agreement"]

CASES = [
    ("verdict", START_VERDICT, "empty", None),
    ("verdict", "", "empty", None),
    ("verdict", SOLUTION_VERDICT, "correct", None),
    ("verdict", '''def verdict(text):
    for word in words(text):
        if word == "pass" or word == "fail":
            return word == "pass"
    return None
''', "correct", None),
    ("verdict", _VERDICT_EARLY, "wrong", "unreached"),
    ("verdict", _VERDICT_LAST, "wrong", "unreached"),
    ("verdict", SOLUTION_VERDICT.replace("return True", "return False", 1), "wrong", None),
    ("agreement", START_AGREEMENT, "empty", None),
    ("agreement", "", "empty", None),
    ("agreement", SOLUTION_AGREEMENT, "correct", None),
    ("agreement", '''def agreement(judge, people):
    return sum(1 for j, p in zip(judge, people) if j == p) / len(people)
''', "correct", None),
    ("agreement", _AGREEMENT_COUNT, "wrong", "count"),
    ("agreement", _AGREEMENT_FAILS, "wrong", "fails"),
    ("caught", START_CAUGHT, "empty", None),
    ("caught", "", "empty", None),
    ("caught", SOLUTION_CAUGHT, "correct", None),
    ("caught", '''def caught(judge, people):
    failed = [i for i in range(len(people)) if people[i] == False]
    return sum(1 for i in failed if judge[i] == False) / len(failed)
''', "correct", None),
    ("caught", _CAUGHT_ALL, "wrong", "all"),
    ("caught", _CAUGHT_NOT, "almost", "not"),
    ("caught", _CAUGHT_COUNT, "wrong", "count"),
]
