# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-risk-decomposition: risk as chance times harm. A made-up AI assistant on a pharmacy's
website, with three hazards: a rude answer, a wrong dose and a missed emergency. The learner writes total_risk, the
classic risk equation summed over the hazards (Hendrycks, Introduction to AI Safety, Ethics, and Society, 2024, 4.1.2),
and disaster_risk, the risk of each hazardous event by the disaster risk equation with its four factors (4.1.5)."""
import contextlib
import copy
import html
import io
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the data --------------------------------------------------------------------------------------------------------

# Made-up: the assistant gives 2,000 answers a day; 500 of the questions are about a dose and 200 describe an
# emergency. Harm is in made-up harm points, where a rude answer counts 1.
HAZARDS = [
    {"name": "rude answer", "p": 0.05, "severity": 1},
    {"name": "wrong dose", "p": 0.001, "severity": 100},
    {"name": "missed emergency", "p": 0.0001, "severity": 5000},
]
EVENTS = [
    {"name": "rude answer", "p": 0.05, "severity": 1, "exposure": 2000, "vulnerability": 1.0},
    {"name": "wrong dose", "p": 0.004, "severity": 100, "exposure": 500, "vulnerability": 0.25},
    {"name": "missed emergency", "p": 0.001, "severity": 5000, "exposure": 200, "vulnerability": 0.5},
]
ANSWERS = 2000                     # answers a day
MISSED = [1, 2, 4, 6, 10, 20]      # the demo: emergencies missed per 10,000 emergency questions
NAMES = {"en": {"rude answer": "rude answer", "wrong dose": "wrong dose", "missed emergency": "missed emergency"},
         "nl": {"rude answer": "onbeleefd antwoord", "wrong dose": "verkeerde dosis", "missed emergency": "gemiste noodsituatie"}}


def total_risk(hazards):
    """What the learner writes in Exercise 1: the classic risk equation, summed over the hazards."""
    total = 0
    for h in hazards:
        total = total + h["p"] * h["severity"]
    return total


def disaster_risk(events):
    """What the learner writes in Exercise 2: the risk of each hazardous event, by its four factors."""
    risks = {}
    for e in events:
        risks[e["name"]] = e["p"] * e["severity"] * e["exposure"] * e["vulnerability"]
    return risks


def with_missed(n) -> list:
    """EVENTS with n emergencies missed per 10,000 emergency questions."""
    events = copy.deepcopy(EVENTS)
    events[2]["p"] = n / 10000
    return events


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:middle")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_CODE = "font-family:var(--ea-font-code);font-size:0.78rem"


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is (its cells, a style of the row), and each cell is (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{h}</th>' for h in head)
    body = "".join(f'<tr style="{style}">' + "".join(f'<td style="{_CELL}{";" + s if s else ""}">{cell}</td>'
                                                    for cell, s in cells_) + "</tr>" for cells_, style in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


def _code(text: str) -> str:
    return f'<span style="{_CODE}">{html.escape(text)}</span>'


def _shown(v, lang: str = "en") -> str:
    """A result as the pictures show it: a number with at most four decimals, else what Python shows."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        text = repr(v)
        return text if len(text) <= 40 else text[:37] + "..."
    try:
        f = float(v)
    except OverflowError:
        text = repr(v)
        return text if len(text) <= 40 else text[:37] + "..."
    if f != f:
        return "nan"
    text = f"{f:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", ",") if lang == "nl" else text


def _same(a, b) -> bool:
    try:
        return not isinstance(a, bool) and abs(float(a) - float(b)) <= 1e-9 + 1e-6 * abs(float(b))
    except (TypeError, ValueError, OverflowError):
        return False


def _bar(value, top, fill: str, width: int = 160) -> str:
    """A horizontal bar for a value from 0 to top."""
    try:
        w = 0 if top <= 0 else max(0.0, min(1.0, float(value) / top)) * width
    except (TypeError, ValueError, OverflowError):
        return ""
    w = 0.0 if w != w else w
    return (f'<span style="display:inline-block;width:{width}px;height:10px;background:var(--ea-inset);'
            f'vertical-align:middle"><span style="display:block;width:{w:.1f}px;height:10px;background:{fill}">'
            f'</span></span>')


_DEMO_WORDS = {
    "en": {"title": "Harm points a day, with {n} missed per 10,000 emergency questions",
           "title_1": "Harm points a day, with 1 missed per 10,000 emergency questions",
           "said": "With {n} missed per 10,000 emergency questions, the missed emergency carries {k} of the {total} harm "
                   "points a day: {part}.",
           "more": "more than half", "less": "less than half",
           "total": "total", "note": "Made-up harm points, where a rude answer counts 1. The bar of the missed "
                                     "emergency stands out; the other two stay the same."},
    "nl": {"title": "Schadepunten per dag, met {n} gemist per 10.000 vragen over een noodsituatie",
           "title_1": "Schadepunten per dag, met 1 gemist per 10.000 vragen over een noodsituatie",
           "said": "Met {n} gemist per 10.000 vragen over een noodsituatie draagt de gemiste noodsituatie {k} van de "
                   "{total} schadepunten per dag: {part}.",
           "more": "meer dan de helft", "less": "minder dan de helft",
           "total": "totaal", "note": "Verzonnen schadepunten, waarbij een onbeleefd antwoord 1 telt. De balk van de "
                                      "gemiste noodsituatie valt op; de andere twee blijven gelijk."},
}


def _whole(v, lang: str) -> str:
    """A whole number as the text writes it: 1,150 in English, 1150 in Dutch, which uses a dot only from 10.000."""
    n = int(round(float(v)))
    if lang == "en":
        return f"{n:,}"
    return f"{n:,}".replace(",", ".") if abs(n) >= 10000 else str(n)


def demo_frame(n, lang: str) -> tuple[str, str]:
    """The demo: the risk of each event a day, with n emergencies missed per 10,000 emergency questions."""
    n = min(MISSED, key=lambda m: abs(m - float(n)))
    w = _DEMO_WORDS[lang]
    risks = disaster_risk(with_missed(n))
    total = sum(risks.values())
    top = disaster_risk(with_missed(max(MISSED)))["missed emergency"]
    rows = []
    for name, r in risks.items():
        fill = "var(--ea-ink)" if name == "missed emergency" else "var(--ea-ink-3)"
        rows.append(([(html.escape(NAMES[lang][name]), ""), (_bar(r, top, fill, 220), ""),
                      (_code(_whole(r, lang)), "text-align:right")], ""))
    rows.append(([(html.escape(w["total"]), "font-weight:600"), ("", ""),
                  (_code(_whole(total, lang)), "text-align:right;font-weight:600")], ""))
    k = risks["missed emergency"]
    title = w["title_1"] if n == 1 else w["title"].format(n=n)
    said = w["said"].format(n=n, k=_whole(k, lang), total=_whole(total, lang),
                            part=w["more"] if 2 * k > total else w["less"])
    body = _table(rows, ["", "", ""]) + _note(w["note"])
    return yours.figure(title, body, alt=title + ". " + said), said


# --- the pictures under the checks -----------------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _all gives for a function that stopped with an error


def _all(fn, calls: list):
    """The learner's function on copies of each list of arguments in `calls`, all within one time limit of a check;
    _STOPPED when it stops with an error, and the Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(lambda: [fn(*(copy.deepcopy(a) for a in args)) for args in calls])
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


_TOTAL_WORDS = {
    "en": {"title": "Your total_risk beside the course's, for the three hazards",
           "head": ["hazard", "chance per answer", "severity", "chance × severity", ""],
           "course": "total, the course's", "yours": "total, yours", "stopped": "error",
           "note": "The bars show each hazard's part of the total. A dashed row is one where your total_risk gives "
                   "something other than the course's."},
    "nl": {"title": "Jouw total_risk naast die van de course, voor de drie gevaren",
           "head": ["gevaar", "kans per antwoord", "ernst", "kans × ernst", ""],
           "course": "totaal van de course", "yours": "jouw totaal", "stopped": "fout",
           "note": "De balken tonen het deel van elk gevaar in het totaal. Een gestippelde rij is er een waar jouw "
                   "total_risk iets anders geeft dan die van de course."},
}


def total_picture(seen) -> str:
    """The three hazards with their chance, severity and risk, and the course's total beside the learner's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _TOTAL_WORDS[seen.lang]
    lang = seen.lang
    got = _all(fn, [(HAZARDS,)])
    want = total_risk(HAZARDS)
    rows = []
    for h in HAZARDS:
        r = h["p"] * h["severity"]
        rows.append(([(html.escape(NAMES[lang][h["name"]]), ""), (_code(_shown(h["p"], lang)), ""),
                      (_code(_shown(h["severity"], lang)), ""), (_code(_shown(r, lang)), ""),
                      (_bar(r, want, "var(--ea-ink-3)"), "")], ""))
    rows.append(([(html.escape(w["course"]), "font-weight:600"), ("", ""), ("", ""),
                  (_code(_shown(want, lang)), "font-weight:600"), (_bar(want, want, "var(--ea-ink-3)"), "")], ""))
    if got is _STOPPED:
        shown, off, bar = html.escape(w["stopped"]), True, ""
    else:
        shown, off = _code(_shown(got[0], lang)), not _same(got[0], want)
        bar = _bar(got[0], want, "var(--ea-ink)") if not off or isinstance(got[0], (int, float)) else ""
    rows.append(([(html.escape(w["yours"]), "font-weight:600"), ("", ""), ("", ""), (shown, "font-weight:600"),
                  (bar, "")], _OFF if off else ""))
    body = _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


_DISASTER_WORDS = {
    "en": {"title": "Your disaster_risk beside the course's: harm points a day",
           "head": ["hazardous event", "p", "severity", "exposure", "vulnerability", "the course's", "yours", ""],
           "stopped": "error", "missing": "missing", "not_dict": "not a dict: {got}",
           "note": "Upper bar the course's, lower bar yours. A dashed row is one where your disaster_risk gives "
                   "something other than the course's."},
    "nl": {"title": "Jouw disaster_risk naast die van de course: schadepunten per dag",
           "head": ["gevaarlijke gebeurtenis", "p", "ernst", "blootstelling", "kwetsbaarheid", "van de course",
                    "de jouwe", ""],
           "stopped": "fout", "missing": "ontbreekt", "not_dict": "geen dict: {got}",
           "note": "Bovenste balk die van de course, onderste balk de jouwe. Een gestippelde rij is er een waar jouw "
                   "disaster_risk iets anders geeft dan die van de course."},
}


def disaster_picture(seen) -> str:
    """For the three events: the four factors, the course's risk and the learner's, as numbers and bars."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _DISASTER_WORDS[seen.lang]
    lang = seen.lang
    got = _all(fn, [(EVENTS,)])
    want = disaster_risk(EVENTS)
    top = max(want.values())
    rows = []
    for e in EVENTS:
        r = want[e["name"]]
        if got is _STOPPED:
            shown, off, mine = html.escape(w["stopped"]), True, None
        elif not isinstance(got[0], dict):
            shown, off, mine = html.escape(w["not_dict"].format(got=_shown(got[0], lang))), True, None
        elif e["name"] not in got[0]:
            shown, off, mine = html.escape(w["missing"]), True, None
        else:
            mine = got[0][e["name"]]
            shown, off = _code(_shown(mine, lang)), not _same(mine, r)
        bars = _bar(r, top, "var(--ea-ink-3)") + "<br>" + (
            _bar(mine, top, "var(--ea-ink)") if isinstance(mine, (int, float)) and not isinstance(mine, bool) else "")
        rows.append(([(html.escape(NAMES[lang][e["name"]]), ""), (_code(_shown(e["p"], lang)), ""),
                      (_code(_shown(e["severity"], lang)), ""), (_code(_shown(e["exposure"], lang)), ""),
                      (_code(_shown(e["vulnerability"], lang)), ""), (_code(_shown(r, lang)), ""),
                      (shown, "font-weight:600"), (bars, "")], _OFF if off else ""))
    body = _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["total_risk", "disaster_risk", "demonstration", "zero"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _names_error(e) -> bool:
    return isinstance(e, NameError) and re.search(r"'(HAZARDS|EVENTS|C)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `HAZARDS` or `C`, and the check does not give it that name: use the names between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `HAZARDS` of `C`, en de check geeft die naam niet mee: gebruik de namen tussen de haakjes van de `def`-regel.")


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


# exercise 1: total_risk; the first case is the three hazards of the pharmacy's assistant
TOTAL_CASES = [
    (Named("HAZARDS", HAZARDS),),
    ([{"name": "a", "p": 0.5, "severity": 4}],),
    ([],),
    ([{"name": "a", "p": 0.2, "severity": 10}, {"name": "b", "p": 0.1, "severity": 0}],),
    ([{"name": "a", "p": 1.0, "severity": 3}, {"name": "b", "p": 0.5, "severity": 2}],),
]
_T_START = "def total_risk(hazards):\n    ...\n"
_T = ("def total_risk(hazards):\n"
      "    total = 0\n"
      "    for h in hazards:\n"
      "        total = total + h[\"p\"] * h[\"severity\"]\n"
      "    return total\n")
_T_P = _T.replace('total + h["p"] * h["severity"]', 'total + h["p"]')
_T_SEVERITY = _T.replace('total + h["p"] * h["severity"]', 'total + h["severity"]')
_T_PLUS = _T.replace('h["p"] * h["severity"]', 'h["p"] + h["severity"]')
_T_LARGEST = _T.replace('total = total + h["p"] * h["severity"]', 'total = max(total, h["p"] * h["severity"])')
_T_MEAN = _T.replace("    return total\n", "    return total / len(hazards)\n")
_T_EARLY = _T.replace("    return total\n", "        return total\n")

# exercise 2: disaster_risk; the first case is the three events of the pharmacy's assistant, a day
DISASTER_CASES = [
    (Named("EVENTS", EVENTS),),
    ([{"name": "a", "p": 0.5, "severity": 2, "exposure": 10, "vulnerability": 0.5}],),
    ([],),
    ([{"name": "a", "p": 0.1, "severity": 10, "exposure": 4, "vulnerability": 0.0},
      {"name": "b", "p": 0.2, "severity": 5, "exposure": 3, "vulnerability": 1.0}],),
    ([{"name": "x", "p": 1.0, "severity": 1, "exposure": 1, "vulnerability": 1.0},
      {"name": "y", "p": 0.5, "severity": 4, "exposure": 2, "vulnerability": 0.5}],),
]
_D_START = "def disaster_risk(events):\n    ...\n"
_D = ("def disaster_risk(events):\n"
      "    risks = {}\n"
      "    for e in events:\n"
      "        risks[e[\"name\"]] = e[\"p\"] * e[\"severity\"] * e[\"exposure\"] * e[\"vulnerability\"]\n"
      "    return risks\n")
_D_NO_VULNERABILITY = _D.replace(' * e["vulnerability"]', "")
_D_NO_EXPOSURE = _D.replace(' * e["exposure"]', "")
_D_PLUS = _D.replace('e["p"] * e["severity"] * e["exposure"] * e["vulnerability"]',
                     'e["p"] + e["severity"] + e["exposure"] + e["vulnerability"]')
_D_EARLY = _D.replace("    return risks\n", "        return risks\n")
_D_KEY = _D.replace('risks[e["name"]]', 'risks["name"]')
_D_TOTAL = ("def disaster_risk(events):\n"
            "    total = 0\n"
            "    for e in events:\n"
            "        total = total + e[\"p\"] * e[\"severity\"] * e[\"exposure\"] * e[\"vulnerability\"]\n"
            "    return total\n")


EXERCISES = {
    "total_risk": FunctionBox(
        name="total_risk",
        picture=total_picture,
        function="total_risk",
        start=_T_START,
        solution=_T,
        visible=[TOTAL_CASES[0]],
        hidden=TOTAL_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("p-only", wrong=_T_P,
                    en="`{call}` gives {got}, and it should give {want}: that is the sum of the chances alone. Risk is the chance times the severity, for each hazard.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is alleen de som van de kansen. Risico is de kans maal de ernst, voor elk gevaar."),
            Mistake("severity-only", wrong=_T_SEVERITY,
                    en="`{call}` gives {got}, and it should give {want}: that is the sum of the severities alone, as if every hazard happened for sure. Multiply each severity by its chance.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is alleen de som van de ernst, alsof elk gevaar zeker gebeurt. Vermenigvuldig elke ernst met de kans erop."),
            Mistake("plus", wrong=_T_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: you add the chance and the severity. Multiply them.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt de kans en de ernst op. Vermenigvuldig ze."),
            Mistake("largest", wrong=_T_LARGEST,
                    en="`{call}` gives {got}, and it should give {want}: that is the risk of the largest hazard. The total adds the risks of all hazards.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is het risico van het grootste gevaar. Het totaal telt de risico's van alle gevaren op."),
            Mistake("mean", wrong=_T_MEAN,
                    en="`{call}` gives {got}, and it should give {want}: that is the average risk of a hazard. Return the sum itself.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is het gemiddelde risico van een gevaar. Geef de som zelf terug."),
            Mistake("early", wrong=_T_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your `return` is inside the loop, so the function stops after the first hazard. Move `return` back, level with `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return` staat in de loop, dus de functie stopt na het eerste gevaar. Zet `return` terug, op gelijke hoogte met `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: for each hazard, its chance times its severity; add them up and return the total.",
                      "nl": "Vervang de `...`: voor elk gevaar de kans maal de ernst; tel ze op en geef het totaal terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Start `total` at 0, add each hazard's `\"p\"` times its `\"severity\"`, and return `total`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Begin `total` bij 0, tel bij elk gevaar de `\"p\"` maal de `\"severity\"` op, en geef `total` terug."},
            "almost": {"en": "Right for the three hazards. But `{call}` gives {got}, and it should give {want}. Does your function use the `hazards` it gets?",
                       "nl": "Goed voor de drie gevaren. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `hazards` die ze krijgt?"},
            "correct": {"en": "Your `total_risk` is right in all {n} cases. Under the check, your total stands under the course's, with each hazard's part.",
                        "nl": "Je `total_risk` klopt in alle {n} gevallen. Onder de check staat jouw totaal onder dat van de course, met het deel van elk gevaar."},
        },
    ),
    "disaster_risk": FunctionBox(
        name="disaster_risk",
        picture=disaster_picture,
        function="disaster_risk",
        start=_D_START,
        solution=_D,
        visible=[DISASTER_CASES[0]],
        hidden=DISASTER_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("no-vulnerability", wrong=_D_NO_VULNERABILITY,
                    en="`{call}` gives {got}, and it should give {want}: the vulnerability is missing from the product. Multiply by `e[\"vulnerability\"]` as well.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de kwetsbaarheid ontbreekt in het product. Vermenigvuldig ook met `e[\"vulnerability\"]`."),
            Mistake("no-exposure", wrong=_D_NO_EXPOSURE,
                    en="`{call}` gives {got}, and it should give {want}: the exposure is missing from the product, so you get the risk of one question, not of a day. Multiply by `e[\"exposure\"]` as well.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de blootstelling ontbreekt in het product, dus je krijgt het risico van één vraag, niet van een dag. Vermenigvuldig ook met `e[\"exposure\"]`."),
            Mistake("plus", wrong=_D_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: you add the four factors. Multiply them.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt de vier factoren op. Vermenigvuldig ze."),
            Mistake("early", wrong=_D_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your `return` is inside the loop, so the dict holds only the first event. Move `return` back, level with `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return` staat in de loop, dus de dict bevat alleen de eerste gebeurtenis. Zet `return` terug, op gelijke hoogte met `for`."),
            Mistake("key", wrong=_D_KEY,
                    en="`{call}` gives {got}, and it should give {want}: every risk goes under the key `\"name\"` itself, so each one replaces the one before. Use the event's name, `e[\"name\"]`, as the key.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk risico komt onder de key `\"name\"` zelf, dus elk vervangt het vorige. Gebruik de naam van de gebeurtenis, `e[\"name\"]`, als key."),
            Mistake("total", wrong=_D_TOTAL,
                    en="`{call}` gives {got}, and it should give {want}: that is the total over the events. Return a dict with the risk of each event under its name.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is het totaal over de gebeurtenissen. Geef een dict terug met het risico van elke gebeurtenis onder de naam ervan."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: make an empty dict, put each event's four factors multiplied under its name, and return the dict.",
                      "nl": "Vervang de `...`: maak een lege dict, zet bij elke gebeurtenis de vier factoren vermenigvuldigd onder de naam ervan, en geef de dict terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. For each event, multiply `\"p\"`, `\"severity\"`, `\"exposure\"` and `\"vulnerability\"`, and put the result in the dict under the event's `\"name\"`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Vermenigvuldig bij elke gebeurtenis `\"p\"`, `\"severity\"`, `\"exposure\"` en `\"vulnerability\"`, en zet de uitkomst in de dict onder de `\"name\"` ervan."},
            "almost": {"en": "Right for the three events. But `{call}` gives {got}, and it should give {want}. Does your function use the `events` it gets?",
                       "nl": "Goed voor de drie gebeurtenissen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `events` die ze krijgt?"},
            "correct": {"en": "Your `disaster_risk` is right in all {n} cases. Under the check, your risks stand beside the course's, with a bar for each.",
                        "nl": "Je `disaster_risk` klopt in alle {n} gevallen. Onder de check staan jouw risico's naast die van de course, met een balk voor elk."},
        },
    ),
}

CASES = [
    ("total_risk", _T_START, "empty", None),
    ("total_risk", _T, "correct", None),
    ("total_risk", "def total_risk(hazards):\n    risks = [h[\"severity\"] * h[\"p\"] for h in hazards]\n    return sum(risks)\n", "correct", None),
    ("total_risk", _T_P, "wrong", "p-only"),
    ("total_risk", _T_SEVERITY, "wrong", "severity-only"),
    ("total_risk", _T_PLUS, "wrong", "plus"),
    ("total_risk", _T_LARGEST, "wrong", "largest"),
    ("total_risk", _T_MEAN, "wrong", "mean"),
    ("total_risk", _T_EARLY, "wrong", "early"),
    ("total_risk", "def total_risk(hazards):\n    return 0.65\n", "almost", None),
    ("total_risk", "def total_risk(hazards):\n    return len(HAZARDS)\n", "error", "lesson-names"),
    ("disaster_risk", _D_START, "empty", None),
    ("disaster_risk", _D, "correct", None),
    ("disaster_risk", "def disaster_risk(events):\n    out = {}\n    for e in events:\n        hazard = e[\"p\"] * e[\"severity\"]\n        out[e[\"name\"]] = hazard * e[\"exposure\"] * e[\"vulnerability\"]\n    return out\n", "correct", None),
    ("disaster_risk", _D_NO_VULNERABILITY, "wrong", "no-vulnerability"),
    ("disaster_risk", _D_NO_EXPOSURE, "wrong", "no-exposure"),
    ("disaster_risk", _D_PLUS, "wrong", "plus"),
    ("disaster_risk", _D_EARLY, "wrong", "early"),
    ("disaster_risk", _D_KEY, "wrong", "key"),
    ("disaster_risk", _D_TOTAL, "wrong", "total"),
    ("disaster_risk", "def disaster_risk(events):\n    return {\"rude answer\": 100.0, \"wrong dose\": 50.0, \"missed emergency\": 500.0}\n", "almost", None),
    ("disaster_risk", "def disaster_risk(events):\n    return len(EVENTS)\n", "error", "lesson-names"),
]
