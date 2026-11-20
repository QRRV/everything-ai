# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-frontier-frameworks: frontier safety frameworks, capability thresholds and if-then
commitments. The learner writes reached, the domains of a made-up lab's framework whose capability threshold a new
model reaches, where a domain that was not measured counts as reached, and statuses, which versions of a model are
below, in the alert band under the threshold, or at the threshold. A demo moves the alert line."""
import contextlib
import copy
import html
import io
import math
import re

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the data --------------------------------------------------------------------------------------------------------

# A made-up lab's framework: for each domain the capability threshold, as a score from 0 to 1 on the lab's own
# evaluations of that domain. All numbers are made up for the lesson.
THRESHOLDS = {"bio": 0.6, "cyber": 0.7, "autonomy": 0.5}
NAMES = {"en": {"bio": "biology", "cyber": "cyber", "autonomy": "autonomy"},
         "nl": {"bio": "biologie", "cyber": "cyber", "autonomy": "autonomie"}}
# The demo: ten versions of a made-up model on one evaluation, its capability threshold, and the margins of the slider.
VERSIONS = [0.21, 0.30, 0.38, 0.45, 0.53, 0.58, 0.64, 0.69, 0.73, 0.79]
THRESHOLD = 0.7
MARGINS = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def reached(scores, thresholds):
    """What the learner writes in Exercise 1: the domains whose threshold is reached; one without a score counts too."""
    hits = []
    for domain, threshold in thresholds.items():
        if domain not in scores or scores[domain] >= threshold:
            hits.append(domain)
    return hits


def statuses(scores, threshold, margin):
    """What the learner writes in Exercise 2: for each score "reached", "alert" (at most margin below the threshold) or
    "below"."""
    result = []
    for score in scores:
        if score >= threshold:
            result.append("reached")
        elif score >= threshold - margin:
            result.append("alert")
        else:
            result.append("below")
    return result


def warning(margin):
    """The demo: the first version (counted from 1) with an alert or more, the first that reaches the threshold, and the
    number of versions of warning in between."""
    seen = statuses(VERSIONS, THRESHOLD, margin)
    first = next(i for i, s in enumerate(seen) if s != "below") + 1
    hit = seen.index("reached") + 1
    return first, hit, hit - first


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


def _number(v):
    """v as a float, or None for something that is not a number, such as True or a text."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError, OverflowError):
        return None


def _value(v) -> str:
    """A result as the check shows it: a number with at most four decimals, a text in quotes, a list or dict of them,
    else what Python shows, cut short when it is long."""
    if isinstance(v, list):
        text = "[" + ", ".join(_value(x) for x in v) + "]"
    elif isinstance(v, dict):
        text = "{" + ", ".join(f"{_value(k)}: {_value(x)}" for k, x in v.items()) + "}"
    elif isinstance(v, str):
        text = '"' + v + '"'
    else:
        f = _number(v)
        if f is None or f != f or f in (float("inf"), float("-inf")):
            text = repr(v) if f is None else str(f)
        else:
            text = f"{f:.4f}".rstrip("0").rstrip(".")
            text = "0" if text == "-0" else text
    return text if len(text) <= 70 else text[:67] + "..."


def _prose(x: float, lang: str, digits: int = 2) -> str:
    """A number in a picture's words: at most `digits` decimals, with a decimal comma in Dutch."""
    text = f"{x:.{digits}f}"
    text = text.rstrip("0").rstrip(".") if digits > 0 else text
    return text.replace(".", ",") if lang == "nl" else text


def _same(got, want) -> bool:
    """Whether the learner's result is the course's: the same list of texts, in the same order."""
    return isinstance(got, list) and got == want


_DEMO_WORDS = {
    "en": {"title": "Ten versions of a model, with an alert line {m} below the threshold",
           "title0": "Ten versions of a model, with no alert line",
           "note0": "Each point is one version's score on the made-up lab's evaluation. The dashed line is the "
                    "capability threshold; with a margin of 0 there is no alert line. A filled purple point has "
                    "reached the threshold, a grey point is below it. All numbers are made up.",
           "x": "model version", "y": "score on the evaluation", "threshold": "threshold {t}", "alert": "alert {a}",
           "said": "With a margin of {m}, the alert line is at {a}. The first alert comes at version {first}, and "
                   "version {hit} reaches the threshold: {k} {versions} of warning to get the promised safeguards "
                   "ready.",
           "said0": "With a margin of 0 there is no alert band: the first signal is the threshold itself, at version "
                    "{hit}, with no version of warning before it.",
           "one": "version", "many": "versions",
           "note": "Each point is one version's score on the made-up lab's evaluation. The upper dashed line is the "
                   "capability threshold, the lower one the alert line. A filled purple point has reached the "
                   "threshold, an open purple point is in the alert band, a grey point is below it. All numbers are "
                   "made up."},
    "nl": {"title": "Tien versies van een model, met een alertlijn {m} onder de threshold",
           "title0": "Tien versies van een model, zonder alertlijn",
           "note0": "Elk punt is de score van één versie op de evaluatie van het verzonnen lab. De gestippelde lijn "
                    "is de capability threshold; met een marge van 0 is er geen alertlijn. Een gevuld paars punt "
                    "heeft de threshold bereikt, een grijs punt zit eronder. Alle getallen zijn verzonnen.",
           "x": "versie van het model", "y": "score op de evaluatie", "threshold": "threshold {t}",
           "alert": "alert {a}",
           "said": "Met een marge van {m} ligt de alertlijn op {a}. Het eerste alert komt bij versie {first}, en "
                   "versie {hit} bereikt de threshold: {k} {versions} waarschuwing om de beloofde safeguards klaar "
                   "te hebben.",
           "said0": "Met een marge van 0 is er geen alertband: het eerste signaal is de threshold zelf, bij versie "
                    "{hit}, zonder één versie waarschuwing ervoor.",
           "one": "versie", "many": "versies",
           "note": "Elk punt is de score van één versie op de evaluatie van het verzonnen lab. De bovenste gestippelde "
                   "lijn is de capability threshold, de onderste de alertlijn. Een gevuld paars punt heeft de "
                   "threshold bereikt, een open paars punt zit in de alertband, een grijs punt zit eronder. Alle "
                   "getallen zijn verzonnen."},
}


def demo_said(margin, lang: str) -> str:
    w = _DEMO_WORDS[lang]
    first, hit, k = warning(margin)
    if margin == 0:
        return w["said0"].format(hit=hit)
    return w["said"].format(m=_prose(margin, lang), a=_prose(THRESHOLD - margin, lang), first=first, hit=hit, k=k,
                            versions=w["one"] if k == 1 else w["many"])


def _status_marks(panel, xs, ys, seen):
    """The course's statuses as points: filled purple at the threshold, open purple in the alert band, grey below."""
    for x, y, s in zip(xs, ys, seen):
        if s == "reached":
            panel.points([x], [y], role="loss", r=5)
        elif s == "alert":
            panel.points([x], [y], role="loss", r=5, hollow=True)
        else:
            panel.points([x], [y], role="ink-3", r=5)


def demo_frame(v, lang: str) -> tuple[str, str]:
    """The demo: ten versions of a model, the threshold, and the alert line at the chosen margin below it."""
    margin = min(MARGINS, key=lambda m: abs(m - float(v)))
    w = _DEMO_WORDS[lang]
    xs = list(range(1, len(VERSIONS) + 1))
    panel = chart.Panel(x=(0.4, 10.6), y=(0, 1), xlabel=w["x"], ylabel=w["y"], xticks=xs,
                        yticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    panel.line([0.4, 10.6], [THRESHOLD, THRESHOLD], role="ink", dashed=True)
    panel.text(0.6, THRESHOLD + 0.05, w["threshold"].format(t=_prose(THRESHOLD, lang)), role="ink", anchor="start",
               halo=True)
    if margin > 0:
        panel.line([0.4, 10.6], [THRESHOLD - margin, THRESHOLD - margin], role="ink-3", dashed=True)
        panel.text(0.6, THRESHOLD - margin - 0.06, w["alert"].format(a=_prose(THRESHOLD - margin, lang)),
                   role="ink-2", anchor="start", halo=True)
    _status_marks(panel, xs, VERSIONS, statuses(VERSIONS, THRESHOLD, margin))
    title = w["title"].format(m=_prose(margin, lang)) if margin > 0 else w["title0"]
    said = demo_said(margin, lang)
    svg = chart.figure([panel], alt=title + ". " + said, width=380, height=240)
    if lang == "nl":
        svg = chart.with_decimal_comma(svg)
    note = w["note"] if margin > 0 else w["note0"]
    return yours.figure(title, svg + _note(note), alt=title + ". " + said), said


# --- the pictures under the checks -----------------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _all gives for a call that stopped with an error


def _all(fn, calls: list) -> list:
    """The learner's function on copies of each list of arguments in `calls`, all within one time limit of a check:
    a list of the results, with _STOPPED for each call that stops with an error; the Timeout passes on."""
    def one(args):
        try:
            return fn(*(copy.deepcopy(a) for a in args))
        except Timeout:
            raise
        except Exception:   # noqa: BLE001   the check above says what went wrong
            return _STOPPED

    with contextlib.redirect_stdout(io.StringIO()):
        return with_time_limit(lambda: [one(args) for args in calls])


_REACHED_WORDS = {
    "en": {"title": "Your reached on the made-up lab's new model",
           "x": "score on the lab's evaluation", "column": "reached", "missing": "not measured",
           "head": ["scores", "thresholds", "the course's", "yours"], "stopped": "error",
           "note": "Each row is one domain of the made-up lab's framework, for the first case of the check. The bar is "
                   "the new model's score and the short line across it the capability threshold; a purple bar has reached it. "
                   "Autonomy was not measured. In the column on the right, a dot is the course's answer and a ring "
                   "is yours. The table is every case of the check; a dashed row is one where your reached gives "
                   "something other than the course's."},
    "nl": {"title": "Jouw reached op het nieuwe model van het verzonnen lab",
           "x": "score op de evaluatie van het lab", "column": "bereikt", "missing": "niet gemeten",
           "head": ["scores", "thresholds", "van de course", "de jouwe"], "stopped": "fout",
           "note": "Elke rij is één domein uit het framework van het verzonnen lab, voor het eerste geval van de check. "
                   "De staaf is de score van het nieuwe model en het korte streepje erdoor de capability threshold; een "
                   "paarse staaf heeft die bereikt. Autonomie is niet gemeten. In de kolom rechts is een stip het "
                   "antwoord van de course en een ring het jouwe. De tabel bevat elk geval van de check; een "
                   "gestippelde rij is er een waar jouw reached iets anders geeft dan die van de course."},
}


def reached_picture(seen) -> str:
    """The first case of the check as bars per domain with the threshold, the course's and the learner's answer beside
    each, and every case of the check in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _REACHED_WORDS[seen.lang]
    got = _all(fn, REACHED_CASES)
    rows = []
    for i, (scores, thresholds) in enumerate(REACHED_CASES):
        want = reached(scores, thresholds)
        if got[i] is _STOPPED:
            shown, off = html.escape(w["stopped"]), True
        else:
            shown, off = _code(_value(got[i])), not _same(got[i], want)
        rows.append(([(_code(_value(scores)), ""), (_code(_value(thresholds)), ""), (_code(_value(want)), ""),
                      (shown, "font-weight:600")], _OFF if off else ""))
    scores, thresholds = REACHED_CASES[0]
    want = reached(scores, thresholds)
    mine = got[0] if isinstance(got[0], list) else []
    panel = chart.Panel(x=(0, 1.3), y=(0.4, 3.9), xlabel=w["x"], xticks=[0, 0.25, 0.5, 0.75, 1], yticks=[])
    panel.text(1.17, 3.62, w["column"], role="ink-2", anchor="middle")
    for (domain, threshold), y in zip(thresholds.items(), (3, 2, 1)):
        panel.text(0, y + 0.36, NAMES[seen.lang][domain], role="ink-2", anchor="start", halo=True)
        if domain in scores:
            panel.rect(0, y - 0.13, scores[domain], y + 0.13, role="loss" if domain in want else "ink-3", solid=True)
        else:
            panel.text(0.02, y, w["missing"], role="ink-3", anchor="start")
        panel.line([threshold, threshold], [y - 0.24, y + 0.24], role="ink")
        if domain in mine:
            panel.points([1.17], [y], role="ink", r=8, hollow=True)
        if domain in want:
            panel.points([1.17], [y], role="loss", r=4)
    svg = chart.figure([panel], alt=w["title"], width=380, height=230)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


_STATUS_WORDS = {
    "en": {"title": "Your statuses beside the course's",
           "x": "version", "y": "score", "threshold": "threshold", "alert": "alert line",
           "head": ["scores", "threshold", "margin", "the course's", "yours"], "stopped": "error",
           "note": "The chart is the first case of the check: each point is a version's score, the dashed lines are "
                   "the threshold and the alert line below it. A filled purple point has reached the threshold, an "
                   "open purple point is in the alert band, a grey point is below. The word at the top, above each "
                   "point, is what your statuses gives, purple where it differs from the course's. The table is "
                   "every case; a dashed row is one where your statuses gives something other than the course's."},
    "nl": {"title": "Jouw statuses naast die van de course",
           "x": "versie", "y": "score", "threshold": "threshold", "alert": "alertlijn",
           "head": ["scores", "threshold", "margin", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De grafiek is het eerste geval van de check: elk punt is de score van een versie, de gestippelde "
                   "lijnen zijn de threshold en de alertlijn eronder. Een gevuld paars punt heeft de threshold "
                   "bereikt, een open paars punt zit in de alertband, een grijs punt zit eronder. Het woord bovenaan, "
                   "boven elk punt, is wat jouw statuses geeft, paars waar het verschilt van die van de course. De "
                   "tabel bevat elk geval; een gestippelde rij is er een waar jouw statuses iets anders geeft dan die "
                   "van de course."},
}


def statuses_picture(seen) -> str:
    """The first case of the check as points with the threshold and the alert line, the learner's status above each
    point, and every case of the check in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _STATUS_WORDS[seen.lang]
    got = _all(fn, STATUS_CASES)
    rows = []
    for i, (scores, threshold, margin) in enumerate(STATUS_CASES):
        want = statuses(scores, threshold, margin)
        if got[i] is _STOPPED:
            shown, off = html.escape(w["stopped"]), True
        else:
            shown, off = _code(_value(got[i])), not _same(got[i], want)
        rows.append(([(_code(_value(scores)), ""), (_code(_value(threshold)), "text-align:right"),
                      (_code(_value(margin)), "text-align:right"), (_code(_value(want)), ""),
                      (shown, "font-weight:600")], _OFF if off else ""))
    scores, threshold, margin = STATUS_CASES[0]
    want = statuses(scores, threshold, margin)
    xs = list(range(1, len(scores) + 1))
    panel = chart.Panel(x=(0.5, len(scores) + 0.5), y=(0.3, 0.9), xlabel=w["x"], ylabel=w["y"], xticks=xs,
                        yticks=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    panel.line([0.5, len(scores) + 0.5], [threshold, threshold], role="ink", dashed=True)
    panel.text(0.6, threshold + 0.025, w["threshold"], role="ink", anchor="start", halo=True)
    panel.line([0.5, len(scores) + 0.5], [threshold - margin, threshold - margin], role="ink-3", dashed=True)
    panel.text(0.6, threshold - margin + 0.025, w["alert"], role="ink-2", anchor="start", halo=True)
    _status_marks(panel, xs, scores, want)
    mine = got[0]
    if isinstance(mine, list):
        for x, y, m, s in zip(xs, scores, mine, want):
            label = m if isinstance(m, str) else _value(m)
            panel.text(x, 0.84, label[:12], role="ink-2" if m == s else "loss", anchor="middle", halo=True)
    svg = chart.figure([panel], alt=w["title"], width=380, height=240)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["reached", "statuses", "lagging", "margin"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# exercise 1: reached; the first case is the lab's new model, with autonomy not measured, then a score exactly at its
# threshold, nothing reached, nothing measured, another framework, and a framework with no domains
REACHED_CASES = [({"bio": 0.62, "cyber": 0.55}, THRESHOLDS),
                 ({"bio": 0.4, "cyber": 0.7, "autonomy": 0.2}, THRESHOLDS),
                 ({"bio": 0.1, "cyber": 0.2, "autonomy": 0.3}, THRESHOLDS),
                 ({}, THRESHOLDS),
                 ({"a": 0.9, "b": 0.95}, {"a": 0.9, "b": 0.8}),
                 ({"x": 0.5}, {"x": 0.6, "y": 0.1}),
                 ({"x": 0.3}, {})]
_H_START = "def reached(scores, thresholds):\n    ...\n"
_H = ("def reached(scores, thresholds):\n"
      "    hits = []\n"
      "    for domain, threshold in thresholds.items():\n"
      "        if domain not in scores or scores[domain] >= threshold:\n"
      "            hits.append(domain)\n"
      "    return hits\n")
_H_STRICT = _H.replace("scores[domain] >= threshold", "scores[domain] > threshold")
_H_SKIP = _H.replace("if domain not in scores or scores[domain] >= threshold",
                     "if domain in scores and scores[domain] >= threshold")
_H_LOOKUP = _H.replace("if domain not in scores or scores[domain] >= threshold", "if scores[domain] >= threshold")
_H_BELOW = _H.replace("scores[domain] >= threshold", "scores[domain] < threshold")
_H_VALUE = _H.replace("            hits.append(domain)\n", "            hits.append(threshold)\n")
_H_EARLY = _H.replace("    return hits\n", "        return hits\n")

# exercise 2: statuses; the first case is four versions of a model, then a score exactly on the alert line, no alert
# band at all, one version at the threshold, no versions, and an alert band that holds two versions
STATUS_CASES = [([0.45, 0.58, 0.66, 0.71], 0.7, 0.1),
                ([0.5, 0.75, 0.8], 0.75, 0.25),
                ([0.1, 0.2], 0.5, 0),
                ([0.9], 0.5, 0.2),
                ([], 0.7, 0.1),
                ([0.3, 0.45, 0.6, 0.2], 0.5, 0.25)]
_S_START = "def statuses(scores, threshold, margin):\n    ...\n"
_S = ("def statuses(scores, threshold, margin):\n"
      "    result = []\n"
      "    for score in scores:\n"
      "        if score >= threshold:\n"
      "            result.append(\"reached\")\n"
      "        elif score >= threshold - margin:\n"
      "            result.append(\"alert\")\n"
      "        else:\n"
      "            result.append(\"below\")\n"
      "    return result\n")
_S_ORDER = ("def statuses(scores, threshold, margin):\n"
            "    result = []\n"
            "    for score in scores:\n"
            "        if score >= threshold - margin:\n"
            "            result.append(\"alert\")\n"
            "        elif score >= threshold:\n"
            "            result.append(\"reached\")\n"
            "        else:\n"
            "            result.append(\"below\")\n"
            "    return result\n")
_S_PLUS = _S.replace("elif score >= threshold - margin:", "elif score >= threshold + margin:")
_S_MARGIN = _S.replace("elif score >= threshold - margin:", "elif score >= margin:")
_S_STRICT = _S.replace("        if score >= threshold:\n", "        if score > threshold:\n")
_S_EARLY = _S.replace("    return result\n", "        return result\n")


EXERCISES = {
    "reached": FunctionBox(
        name="reached",
        picture=reached_picture,
        function="reached",
        start=_H_START,
        solution=_H,
        visible=[REACHED_CASES[0]],
        hidden=REACHED_CASES[1:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            Mistake("lookup", error=lambda e: isinstance(e, KeyError),
                    en="`{call}` stops with a `KeyError`: a domain of the framework has no score, because it was not measured. Check first whether the domain is in `scores`; a domain that was not measured counts as reached, following the rule in Anthropic's policy of 2025.",
                    nl="`{call}` stopt met een `KeyError`: een domein uit het framework heeft geen score, omdat het niet gemeten is. Kijk eerst of het domein in `scores` staat; een domein dat niet gemeten is, telt als bereikt, volgens de regel in het beleid van Anthropic uit 2025."),
            Mistake("strict", wrong=_H_STRICT,
                    en="`{call}` gives {got}, and it should give {want}: you use `>`, so a score exactly at the threshold does not count. A threshold is reached when the score is at it or above it: use `>=`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je gebruikt `>`, dus een score die precies op de threshold ligt, telt niet. Een threshold is bereikt als de score erop of erboven ligt: gebruik `>=`."),
            Mistake("skip", wrong=_H_SKIP,
                    en="`{call}` gives {got}, and it should give {want}: a domain that was not measured counts as not reached in your function. Then a lab that skips a test never has to act. A domain without a score counts as reached.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in jouw functie telt een domein dat niet gemeten is als niet bereikt. Dan hoeft een lab dat een test overslaat nooit iets te doen. Een domein zonder score telt als bereikt."),
            Mistake("below", wrong=_H_BELOW,
                    en="`{call}` gives {got}, and it should give {want}: you keep the domains whose score is under the threshold. Reached means at the threshold or above it: compare with `>=`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je houdt de domeinen over waarvan de score onder de threshold ligt. Bereikt betekent op de threshold of erboven: vergelijk met `>=`."),
            Mistake("value", wrong=_H_VALUE,
                    en="`{call}` gives {got}, and it should give {want}: you put the thresholds in the list. The list holds the names of the domains whose threshold is reached: append `domain`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je zet de thresholds in de lijst. De lijst bevat de namen van de domeinen waarvan de threshold bereikt is: voeg `domain` toe."),
            Mistake("early", wrong=_H_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your `return` stands inside the loop, so the function stops after the first domain. Put `return hits` after the loop, level with `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return` staat in de loop, dus de functie stopt na het eerste domein. Zet `return hits` na de loop, op gelijke hoogte met `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go through the domains of `thresholds`, and keep each domain that has no score, or whose score is at its threshold or above it.",
                      "nl": "Vervang de `...`: loop langs de domeinen van `thresholds`, en houd elk domein over dat geen score heeft, of waarvan de score op de threshold of erboven ligt."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Begin `hits` as an empty list. For each `domain` and `threshold` in `thresholds.items()`, append `domain` when it is not in `scores`, or when its score is at least `threshold`; after the loop, return `hits`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Begin `hits` als een lege lijst. Voeg voor elk `domain` en `threshold` in `thresholds.items()` het `domain` toe als het niet in `scores` staat, of als de score minstens `threshold` is; geef na de loop `hits` terug."},
            "almost": {"en": "Right for the lab's new model. But `{call}` gives {got}, and it should give {want}. Does your function use the `scores` and `thresholds` it gets?",
                       "nl": "Goed voor het nieuwe model van het lab. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `scores` en `thresholds` die ze krijgt?"},
            "correct": {"en": "Your `reached` is right in all {n} cases. Under the check, your rings stand beside the domains whose threshold you found reached.",
                        "nl": "Je `reached` klopt in alle {n} gevallen. Onder de check staan jouw ringen bij de domeinen waarvan je de threshold bereikt vond."},
        },
    ),
    "statuses": FunctionBox(
        name="statuses",
        picture=statuses_picture,
        function="statuses",
        start=_S_START,
        solution=_S,
        visible=[STATUS_CASES[0]],
        hidden=STATUS_CASES[1:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            Mistake("order", wrong=_S_ORDER,
                    en="`{call}` gives {got}, and it should give {want}: you test the alert line first. A score at the threshold is also above the alert line, so it gets \"alert\" and never \"reached\". Test the threshold first, then the alert line.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je test eerst de alertlijn. Een score op de threshold ligt ook boven de alertlijn, dus die krijgt \"alert\" en nooit \"reached\". Test eerst de threshold, dan de alertlijn."),
            Mistake("plus", wrong=_S_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: your alert line lies above the threshold, because you add `margin` to it. An alert warns before the threshold is reached, so the alert line lies `margin` below it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je alertlijn ligt boven de threshold, omdat je `margin` erbij optelt. Een alert waarschuwt voordat de threshold bereikt is, dus de alertlijn ligt `margin` eronder."),
            Mistake("margin", wrong=_S_MARGIN,
                    en="`{call}` gives {got}, and it should give {want}: you compare the score with `margin` itself. The margin is a distance, not a score: the alert line lies `margin` below `threshold`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je vergelijkt de score met `margin` zelf. De marge is een afstand, geen score: de alertlijn ligt `margin` onder `threshold`."),
            Mistake("strict", wrong=_S_STRICT,
                    en="`{call}` gives {got}, and it should give {want}: you use `>` for the threshold, so a score exactly at it gets only \"alert\". At the threshold counts as reached: use `>=`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je gebruikt `>` voor de threshold, dus een score die er precies op ligt, krijgt maar \"alert\". Op de threshold telt als bereikt: gebruik `>=`."),
            Mistake("early", wrong=_S_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your `return` stands inside the loop, so the function stops after the first version. Put `return result` after the loop, level with `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return` staat in de loop, dus de functie stopt na de eerste versie. Zet `return result` na de loop, op gelijke hoogte met `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: for each score, add \"reached\" at the threshold or above it, \"alert\" at most `margin` below it, and \"below\" otherwise.",
                      "nl": "Vervang de `...`: zet voor elke score \"reached\" in de lijst op de threshold of erboven, \"alert\" hooguit `margin` eronder, en anders \"below\"."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Begin `result` as an empty list. For each `score`, append \"reached\" when it is at least `threshold`, else \"alert\" when it is at most `margin` below `threshold`, else \"below\"; after the loop, return `result`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Begin `result` als een lege lijst. Voeg voor elke `score` \"reached\" toe als die minstens `threshold` is, anders \"alert\" als die hooguit `margin` onder `threshold` ligt, en anders \"below\"; geef na de loop `result` terug."},
            "almost": {"en": "Right for the first versions. But `{call}` gives {got}, and it should give {want}. Does your function use the `scores`, `threshold` and `margin` it gets?",
                       "nl": "Goed voor de eerste versies. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `scores`, `threshold` en `margin` die ze krijgt?"},
            "correct": {"en": "Your `statuses` is right in all {n} cases. Under the check, your status stands above each version.",
                        "nl": "Je `statuses` klopt in alle {n} gevallen. Onder de check staat jouw status boven elke versie."},
        },
    ),
}

CASES = [
    ("reached", _H_START, "empty", None),
    ("reached", _H, "correct", None),
    ("reached", "def reached(scores, thresholds):\n    hits = []\n    for domain in thresholds:\n        if domain in scores:\n            if scores[domain] >= thresholds[domain]:\n                hits.append(domain)\n        else:\n            hits.append(domain)\n    return hits\n", "correct", None),
    ("reached", "def reached(scores, thresholds):\n    return [d for d, t in thresholds.items() if scores.get(d, 1) >= t]\n", "correct", None),
    ("reached", _H_LOOKUP, "error", "lookup"),
    ("reached", _H_STRICT, "wrong", "strict"),
    ("reached", _H_SKIP, "wrong", "skip"),
    ("reached", _H_BELOW, "wrong", "below"),
    ("reached", _H_VALUE, "wrong", "value"),
    ("reached", _H_EARLY, "wrong", "early"),
    ("reached", "def reached(scores, thresholds):\n    return [\"bio\", \"autonomy\"]\n", "almost", None),
    ("statuses", _S_START, "empty", None),
    ("statuses", _S, "correct", None),
    ("statuses", "def statuses(scores, threshold, margin):\n    result = []\n    for score in scores:\n        if score < threshold - margin:\n            result.append(\"below\")\n        elif score < threshold:\n            result.append(\"alert\")\n        else:\n            result.append(\"reached\")\n    return result\n", "correct", None),
    ("statuses", "def statuses(scores, threshold, margin):\n    return [\"reached\" if s >= threshold else \"alert\" if s >= threshold - margin else \"below\" for s in scores]\n", "correct", None),
    ("statuses", _S_ORDER, "wrong", "order"),
    ("statuses", _S_PLUS, "wrong", "plus"),
    ("statuses", _S_MARGIN, "wrong", "margin"),
    ("statuses", _S_STRICT, "wrong", "strict"),
    ("statuses", _S_EARLY, "wrong", "early"),
    ("statuses", "def statuses(scores, threshold, margin):\n    return [\"below\", \"below\", \"alert\", \"reached\"]\n", "almost", None),
]
