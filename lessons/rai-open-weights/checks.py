# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-open-weights: open weights, for and against, and marginal risk. The learner writes
marginal_risk, the risk a release adds over a made-up lab's threats (chance times harm, with the release minus
without it), and risk_over_time, the risk after each of a row of releases that each add a little. A demo adds one
release after another."""
import contextlib
import copy
import html
import io
import math
import re

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the data --------------------------------------------------------------------------------------------------------

# A made-up lab weighs releasing its model's weights. For each threat: the chance in a year that it happens without the
# release and with it, and the harm when it does, in harm points. All numbers are made up for the lesson.
THREATS = [{"name": "spam", "without": 0.9, "with": 0.92, "harm": 1},
           {"name": "phishing", "without": 0.3, "with": 0.34, "harm": 10},
           {"name": "fake reviews", "without": 0.5, "with": 0.55, "harm": 2},
           {"name": "malware", "without": 0.05, "with": 0.06, "harm": 50}]
NAMES = {"en": {"spam": "spam", "phishing": "phishing", "fake reviews": "fake reviews", "malware": "malware"},
         "nl": {"spam": "spam", "phishing": "phishing", "fake reviews": "nepreviews", "malware": "malware"}}
STEP = 0.25                          # the demo: what each later release adds, compared with the world just before it
RELEASES = list(range(1, 11))        # the demo's slider: how many releases follow the lab's


def risk(threats, key):
    """The risk of the threats in one world, "without" or "with" the release: chance times harm, summed."""
    total = 0
    for threat in threats:
        total = total + threat[key] * threat["harm"]
    return total


def marginal_risk(threats):
    """What the learner writes in Exercise 1: the risk with the release minus the risk without it."""
    without = 0
    with_release = 0
    for threat in threats:
        without = without + threat["without"] * threat["harm"]
        with_release = with_release + threat["with"] * threat["harm"]
    return with_release - without


def risk_over_time(start, extras):
    """What the learner writes in Exercise 2: the risk after each release, when each adds its extra to the risk just
    before it."""
    risk_ = start
    risks = []
    for extra in extras:
        risk_ = risk_ + extra
        risks.append(risk_)
    return risks


def demo_risks(n: int) -> list:
    """The demo: the risk before the lab's release, after it, and after n later releases that each add STEP."""
    start = risk(THREATS, "without")
    return [start] + risk_over_time(start, [marginal_risk(THREATS)] + [STEP] * n)


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
    """A result as the check shows it: a number with at most four decimals, a list of them, else what Python shows, cut
    short when it is long."""
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
    """Whether the learner's result is the course's number, or list of numbers, up to rounding."""
    if isinstance(want, list):
        return isinstance(got, list) and len(got) == len(want) and all(_same(g, w) for g, w in zip(got, want))
    return _number(got) is not None and math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-9)


_DEMO_WORDS = {
    "en": {"title": "The lab's release and {n} {releases} after it",
           "one": "release", "many": "releases", "add": ("adds", "add"), "each": ("", " each"),
           "x": "releases", "y": "risk in harm points",
           "said": "Before the lab's release the risk is {start} harm points. The lab's release adds {lab}; {n} later "
                   "{releases} {add} {step}{each}, compared with the world just before. Together that is {now}: {up}% "
                   "above the start, while the last release added {last}% to the risk just before it.",
           "start": "start",
           "note": "The first bar is the risk before the lab's release, chance times harm summed over its four "
                   "threats; release 1 is the lab's, and each later release adds 0.25, as in exercise 2. The grey part and the dashed line are the risk before the lab's release; "
                   "the purple part is what all releases up to that bar added. All numbers are made up."},
    "nl": {"title": "De release van het lab en {n} {releases} daarna",
           "one": "release", "many": "releases", "add": ("voegt", "voegen"), "each": ("", "elk "),
           "x": "releases", "y": "risico in schadepunten",
           "said": "Voor de release van het lab is het risico {start} schadepunten. De release van het lab voegt {lab} "
                   "toe; {n} latere {releases} {add} {each}{step} toe, vergeleken met de wereld vlak ervoor. Samen is dat "
                   "{now}: {up}% boven het begin, terwijl de laatste release {last}% toevoegde aan het risico vlak "
                   "ervoor.",
           "start": "begin",
           "note": "De eerste staaf is het risico voor de release van het lab, kans maal schade opgeteld over de vier "
                   "dreigingen; release 1 is die van het lab, en elke latere release voegt 0,25 toe, zoals in oefening 2. "
                   "Het grijze deel en de gestippelde lijn zijn het risico voor de "
                   "release van het lab; het paarse deel is wat alle releases tot en met die staaf toevoegden. Alle "
                   "getallen zijn verzonnen."},
}


def demo_said(n: int, lang: str) -> str:
    w = _DEMO_WORDS[lang]
    risks = demo_risks(n)
    last = (risks[-1] - risks[-2]) / risks[-2] * 100
    return w["said"].format(start=_prose(risks[0], lang), lab=_prose(marginal_risk(THREATS), lang), n=n,
                            releases=w["one"] if n == 1 else w["many"], add=w["add"][n > 1], each=w["each"][n > 1],
                            step=_prose(STEP, lang),
                            now=_prose(risks[-1], lang), up=_prose((risks[-1] / risks[0] - 1) * 100, lang, 0),
                            last=f"{last:.1f}".replace(".", "," if lang == "nl" else "."))


def demo_frame(v, lang: str) -> tuple[str, str]:
    """The demo: the risk as a row of bars, before the lab's release, after it and after n later releases."""
    n = min(RELEASES, key=lambda m: abs(m - float(v)))
    w = _DEMO_WORDS[lang]
    risks = demo_risks(n)
    k = len(risks)
    panel = chart.Panel(x=(-0.6, 11.6), y=(0, 12), xlabel=w["x"], ylabel=w["y"], xticks=list(range(k)),
                        xtick_labels=[w["start"]] + [str(i) for i in range(1, k)], yticks=[0, 2, 4, 6, 8, 10, 12])
    for i in range(k):
        panel.rect(i - 0.35, 0, i + 0.35, risks[0], role="ink-3", solid=True)
        for j in range(1, i + 1):   # each release's own step, on top of the steps before it
            panel.rect(i - 0.35, risks[j - 1], i + 0.35, risks[j], role="loss", solid=True)
    panel.line([-0.5, 11.5], [risks[0], risks[0]], role="ink-2", dashed=True)
    word = w["one"] if n == 1 else w["many"]
    title = w["title"].format(n=n, releases=word)
    said = demo_said(n, lang)
    svg = chart.figure([panel], alt=title + ". " + said, width=380, height=240)
    if lang == "nl":
        svg = chart.with_decimal_comma(svg)
    return yours.figure(title, svg + _note(w["note"]), alt=title + ". " + said), said


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


_MARGINAL_WORDS = {
    "en": {"title": "Your marginal_risk on the lab's four threats",
           "x": "risk in harm points: chance times harm", "without": "without the release", "extra": "what it adds",
           "yours": "yours",
           "head": ["threats", "the course's", "yours"], "stopped": "error",
           "total": "Over all four threats, your marginal_risk gives {got}; the course's gives {want}.",
           "note": "Each row is one threat of the made-up lab, given to your marginal_risk on its own. The grey bar is "
                   "its risk without the release, the purple bar what the release adds; your ring stands at the risk "
                   "without the release plus what your marginal_risk gives. The table is every case of the check; a "
                   "dashed row is one where your marginal_risk gives something other than the course's."},
    "nl": {"title": "Jouw marginal_risk op de vier dreigingen van het lab",
           "x": "risico in schadepunten: kans maal schade", "without": "zonder de release",
           "extra": "wat die toevoegt", "yours": "de jouwe",
           "head": ["dreigingen", "van de course", "de jouwe"], "stopped": "fout",
           "total": "Over alle vier de dreigingen geeft jouw marginal_risk {got}; die van de course geeft {want}.",
           "note": "Elke rij is één dreiging van het verzonnen lab, los aan jouw marginal_risk gegeven. De grijze staaf "
                   "is het risico zonder de release, de paarse staaf wat de release toevoegt; jouw ring staat bij het "
                   "risico zonder de release plus wat jouw marginal_risk geeft. De tabel bevat elk geval van de check; "
                   "een gestippelde rij is er een waar jouw marginal_risk iets anders geeft dan die van de course."},
}


def marginal_picture(seen) -> str:
    """The learner's marginal_risk on each of the lab's threats beside the course's bars, the total over all four, and
    every case of the check in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _MARGINAL_WORDS[seen.lang]
    got = _all(fn, MARGINAL_CASES)
    rows = []
    for i, (threats,) in enumerate(MARGINAL_CASES):
        want = marginal_risk(threats)
        if got[i] is _STOPPED:
            shown, off = html.escape(w["stopped"]), True
        else:
            shown, off = _code(_value(got[i])), not _same(got[i], want)
        rows.append(([(_code(_value(threats)), ""), (_code(_value(want)), "text-align:right"),
                      (shown, "text-align:right;font-weight:600")], _OFF if off else ""))
    mine = _all(fn, [([t],) for t in THREATS] + [(THREATS,)])
    panel = chart.Panel(x=(0, 4), y=(0.4, 5.7), xlabel=w["x"], xticks=[0, 1, 2, 3, 4], yticks=[])
    for t, y in zip(THREATS, (4, 3, 2, 1)):
        before, after = t["without"] * t["harm"], t["with"] * t["harm"]
        panel.text(0, y + 0.42, NAMES[seen.lang][t["name"]], role="ink-2", anchor="start", halo=True)
        panel.rect(0, y - 0.16, before, y + 0.16, role="ink-3", solid=True)
        panel.rect(before, y - 0.16, after, y + 0.16, role="loss", solid=True)
    for k, (t, y) in enumerate(zip(THREATS, (4, 3, 2, 1))):
        m = None if mine[k] is _STOPPED else _number(mine[k])
        if m is not None:
            panel.points([min(max(t["without"] * t["harm"] + m, 0), 4)], [y], role="ink", r=6, hollow=True)
    panel.rect(1.8, 5.32, 1.96, 5.48, role="ink-3", solid=True)   # a key above the rows, where no ring can be
    panel.text(2.06, 5.4, w["without"], role="ink-2", anchor="start")
    panel.rect(1.8, 4.97, 1.96, 5.13, role="loss", solid=True)
    panel.text(2.06, 5.05, w["extra"], role="loss", anchor="start")
    panel.points([1.88], [4.7], role="ink", r=6, hollow=True)
    panel.text(2.06, 4.7, w["yours"], role="ink", anchor="start")
    svg = chart.figure([panel], alt=w["title"], width=380, height=270)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    total = mine[-1]
    shown = (w["stopped"] if total is _STOPPED else
             _prose(total, seen.lang, 4) if _number(total) is not None else _value(total))
    line = w["total"].format(got=shown, want=_prose(marginal_risk(THREATS), seen.lang, 4))
    body = svg + _note(line) + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


_TIME_WORDS = {
    "en": {"title": "Your risk_over_time beside the course's",
           "x": "after release", "y": "risk", "course": "the course's", "yours": "yours", "start": "start",
           "head": ["start", "extras", "the course's", "yours"], "stopped": "error",
           "note": "The chart is the first case of the check: the line is the course's risk after each release, the "
                   "rings yours, and the dashed line the risk at the start. The table is every case; a dashed row is "
                   "one where your risk_over_time gives something other than the course's."},
    "nl": {"title": "Jouw risk_over_time naast die van de course",
           "x": "na release", "y": "risico", "course": "van de course", "yours": "de jouwe", "start": "begin",
           "head": ["start", "extras", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De grafiek is het eerste geval van de check: de lijn is het risico van de course na elke release, de "
                   "ringen zijn de jouwe, en de gestippelde lijn is het risico bij het begin. De tabel bevat elk geval; "
                   "een gestippelde rij is er een waar jouw risk_over_time iets anders geeft dan die van de course."},
}


def time_picture(seen) -> str:
    """The first case of the check as a line with the learner's risks as rings, and every case in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _TIME_WORDS[seen.lang]
    got = _all(fn, TIME_CASES)
    rows = []
    for i, (start, extras) in enumerate(TIME_CASES):
        want = risk_over_time(start, extras)
        if got[i] is _STOPPED:
            shown, off = html.escape(w["stopped"]), True
        else:
            shown, off = _code(_value(got[i])), not _same(got[i], want)
        rows.append(([(_code(_value(start)), "text-align:right"), (_code(_value(extras)), ""),
                      (_code(_value(want)), ""), (shown, "font-weight:600")], _OFF if off else ""))
    start, extras = TIME_CASES[0]
    want = risk_over_time(start, extras)
    xs = list(range(1, len(extras) + 1))
    top = math.ceil(max(want) + 1)
    panel = chart.Panel(x=(0, len(extras) + 0.5), y=(start - 1, top), xlabel=w["x"], ylabel=w["y"],
                        xticks=[0] + xs, yticks=list(range(math.ceil(start - 1), top + 1)))
    panel.line([0, len(extras) + 0.5], [start, start], role="ink-3", dashed=True)
    panel.line([0] + xs, [start] + want, role="loss")
    panel.points([0], [start], role="loss")
    mine = got[0]
    if isinstance(mine, list):
        for x, m in zip(xs, mine):
            if _number(m) is not None:
                panel.points([x], [min(max(float(m), start - 1), top)], role="ink", r=6, hollow=True)
    panel.line([0.3, 0.75], [top - 0.35, top - 0.35], role="loss")   # a key at the top left
    panel.text(0.9, top - 0.35, w["course"], role="loss", anchor="start")
    panel.points([0.52], [top - 0.85], role="ink", r=6, hollow=True)
    panel.text(0.9, top - 0.85, w["yours"], role="ink", anchor="start")
    svg = chart.figure([panel], alt=w["title"], width=380, height=230)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["marginal_risk", "risk_over_time", "recall", "duplicate"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _t(name, without, with_, harm):
    return {"name": name, "without": without, "with": with_, "harm": harm}


# exercise 1: marginal_risk; the first case is two of the lab's threats, then threats whose harm is not 1, a threat the
# release makes less likely, and no threats at all
MARGINAL_CASES = [([_t("spam", 0.9, 0.92, 1), _t("phishing", 0.3, 0.34, 10)],),
                  (THREATS,),
                  ([_t("malware", 0.05, 0.06, 50)],),
                  ([_t("spam", 0.9, 0.92, 1)],),
                  ([_t("a", 0.2, 0.2, 5), _t("b", 0.1, 0.3, 4)],),
                  ([_t("old", 0.5, 0.4, 3), _t("new", 0.0, 0.1, 20)],),
                  ([],)]
_M_START = "def marginal_risk(threats):\n    ...\n"
_M = ("def marginal_risk(threats):\n"
      "    without = 0\n"
      "    with_release = 0\n"
      "    for threat in threats:\n"
      "        without = without + threat[\"without\"] * threat[\"harm\"]\n"
      "        with_release = with_release + threat[\"with\"] * threat[\"harm\"]\n"
      "    return with_release - without\n")
_M_TOTAL = _M.replace("    return with_release - without\n", "    return with_release\n")
_M_REVERSE = _M.replace("    return with_release - without\n", "    return without - with_release\n")
_M_CHANCE = (_M.replace("threat[\"without\"] * threat[\"harm\"]", "threat[\"without\"]")
             .replace("threat[\"with\"] * threat[\"harm\"]", "threat[\"with\"]"))
_M_RATIO = _M.replace("    return with_release - without\n", "    return with_release / without\n")
_M_EARLY = _M.replace("    return with_release - without\n", "        return with_release - without\n")

# exercise 2: risk_over_time; the first case is the lab's release and two after it
TIME_CASES = [(7.4, [1.02, 0.25, 0.25]),
              (0, [1, 1, 1]),
              (5, [0.5]),
              (2, []),
              (10, [2, -1, 0.5, 0.25]),
              (1, [0.1, 0.2, 0.3, 0.4, 0.5])]
_R_START = "def risk_over_time(start, extras):\n    ...\n"
_R = ("def risk_over_time(start, extras):\n"
      "    risk = start\n"
      "    risks = []\n"
      "    for extra in extras:\n"
      "        risk = risk + extra\n"
      "        risks.append(risk)\n"
      "    return risks\n")
_R_ZERO = _R.replace("    risk = start\n", "    risk = 0\n")
_R_EACH = _R.replace("        risk = risk + extra\n        risks.append(risk)\n",
                     "        risks.append(start + extra)\n")
_R_BEFORE = _R.replace("        risk = risk + extra\n        risks.append(risk)\n",
                       "        risks.append(risk)\n        risk = risk + extra\n")
_R_EXTRA = _R.replace("        risks.append(risk)\n", "        risks.append(extra)\n")
_R_EARLY = _R.replace("    return risks\n", "        return risks\n")


EXERCISES = {
    "marginal_risk": FunctionBox(
        name="marginal_risk",
        picture=marginal_picture,
        function="marginal_risk",
        start=_M_START,
        solution=_M,
        visible=[MARGINAL_CASES[0]],
        hidden=MARGINAL_CASES[1:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            Mistake("total", wrong=_M_TOTAL,
                    en="`{call}` gives {got}, and it should give {want}: you return the whole risk with the release. Most of that risk was already there without it: the marginal risk is only what the release adds.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft het hele risico met de release terug. Het meeste daarvan was er zonder de release ook al: het marginal risk is alleen wat de release toevoegt."),
            Mistake("reverse", wrong=_M_REVERSE,
                    en="`{call}` gives {got}, and it should give {want}: the right size, but the wrong sign. A release that makes harm more likely adds risk, so its marginal risk is positive: take the world without the release away from the world with it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de goede grootte, maar het verkeerde teken. Een release die schade waarschijnlijker maakt, voegt risico toe, dus het marginal risk is positief: haal de wereld zonder de release af van de wereld met de release."),
            Mistake("chance", wrong=_M_CHANCE,
                    en="`{call}` gives {got}, and it should give {want}: you add up only the chances. Risk is chance times harm: a small rise in the chance of a threat with a large harm can add more than a big rise in the chance of a small one. Multiply each chance by `threat[\"harm\"]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt alleen de kansen op. Risico is kans maal schade: een kleine stijging van de kans op een dreiging met veel schade kan meer toevoegen dan een grote stijging bij een kleine. Vermenigvuldig elke kans met `threat[\"harm\"]`."),
            Mistake("ratio", wrong=_M_RATIO,
                    en="`{call}` gives {got}, and it should give {want}: you divide, so you get how many times as large the risk becomes. The marginal risk is how much it grows, in harm points.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je deelt, dus je krijgt hoeveel keer zo groot het risico wordt. Het marginal risk is hoeveel het groeit, in schadepunten."),
            Mistake("early", wrong=_M_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your `return` stands inside the loop, so the function stops after the first threat. Put the `return` after the loop, level with `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return` staat in de loop, dus de functie stopt na de eerste dreiging. Zet de `return` na de loop, op gelijke hoogte met `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: add up chance times harm for each threat, once for the world without the release and once for the world with it, and return what the release adds.",
                      "nl": "Vervang de `...`: tel voor elke dreiging kans maal schade op, één keer voor de wereld zonder de release en één keer voor de wereld met de release, en geef terug wat de release toevoegt."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Begin `without` and `with_release` at 0. For each `threat`, add `threat[\"without\"]` times `threat[\"harm\"]` to `without`, and the same with `threat[\"with\"]` to `with_release`; after the loop, return what the release adds.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Begin `without` en `with_release` bij 0. Tel voor elke `threat` `threat[\"without\"]` maal `threat[\"harm\"]` op bij `without`, en hetzelfde met `threat[\"with\"]` bij `with_release`; geef na de loop terug wat de release toevoegt."},
            "almost": {"en": "Right for the first threats. But `{call}` gives {got}, and it should give {want}. Does your function use the threats it gets?",
                       "nl": "Goed voor de eerste dreigingen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de dreigingen die ze krijgt?"},
            "correct": {"en": "Your `marginal_risk` is right in all {n} cases. Under the check, your ring stands where the risk ends for each of the lab's threats.",
                        "nl": "Je `marginal_risk` klopt in alle {n} gevallen. Onder de check staat jouw ring waar het risico eindigt, bij elk van de dreigingen van het lab."},
        },
    ),
    "risk_over_time": FunctionBox(
        name="risk_over_time",
        picture=time_picture,
        function="risk_over_time",
        start=_R_START,
        solution=_R,
        visible=[TIME_CASES[0]],
        hidden=TIME_CASES[1:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            Mistake("zero", wrong=_R_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: you start at 0, so you add up only the extras. The releases add to a risk that was already there: begin `risk` at `start`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je begint bij 0, dus je telt alleen de extra's op. De releases komen bovenop een risico dat er al was: begin `risk` bij `start`."),
            Mistake("each", wrong=_R_EACH,
                    en="`{call}` gives {got}, and it should give {want}: you add each extra to the start, as if every release came alone. Each release adds to the world just before it, with the releases before it included: keep a running `risk`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt elke extra op bij het begin, alsof elke release de enige was. Elke release komt bovenop de wereld vlak ervoor, met de releases daarvoor erbij: houd een lopend `risk` bij."),
            Mistake("before", wrong=_R_BEFORE,
                    en="`{call}` gives {got}, and it should give {want}: you add the risk to the list before the release adds its extra, so each risk is one release late. First add the extra, then append.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je zet het risico in de lijst voordat de release zijn extra toevoegt, dus elk risico loopt één release achter. Tel eerst de extra op, en voeg dan toe."),
            Mistake("extra", wrong=_R_EXTRA,
                    en="`{call}` gives {got}, and it should give {want}: you put each extra in the list. The list holds the risk after each release: append `risk`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je zet elke extra in de lijst. De lijst bevat het risico na elke release: voeg `risk` toe."),
            Mistake("early", wrong=_R_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your `return` stands inside the loop, so the function stops after the first release. Put `return risks` after the loop, level with `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return` staat in de loop, dus de functie stopt na de eerste release. Zet `return risks` na de loop, op gelijke hoogte met `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: keep a running `risk` that begins at `start`, add each extra to it, and put the risk after each release in a list.",
                      "nl": "Vervang de `...`: houd een lopend `risk` bij dat bij `start` begint, tel er elke extra bij op, en zet het risico na elke release in een lijst."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Begin `risk` at `start` and `risks` as an empty list. For each `extra`, add it to `risk`, then append `risk` to `risks`; after the loop, return `risks`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Begin `risk` bij `start` en `risks` als een lege lijst. Tel elke `extra` op bij `risk` en voeg daarna `risk` toe aan `risks`; geef na de loop `risks` terug."},
            "almost": {"en": "Right for the first releases. But `{call}` gives {got}, and it should give {want}. Does your function use the `start` and the `extras` it gets?",
                       "nl": "Goed voor de eerste releases. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `start` en de `extras` die ze krijgt?"},
            "correct": {"en": "Your `risk_over_time` is right in all {n} cases. Under the check, your risks are the rings on the course's line.",
                        "nl": "Je `risk_over_time` klopt in alle {n} gevallen. Onder de check zijn jouw risico's de ringen op de lijn van de course."},
        },
    ),
}

CASES = [
    ("marginal_risk", _M_START, "empty", None),
    ("marginal_risk", _M, "correct", None),
    ("marginal_risk", "def marginal_risk(threats):\n    extra = 0\n    for threat in threats:\n        extra += (threat[\"with\"] - threat[\"without\"]) * threat[\"harm\"]\n    return extra\n", "correct", None),
    ("marginal_risk", "def marginal_risk(threats):\n    return sum((t[\"with\"] - t[\"without\"]) * t[\"harm\"] for t in threats)\n", "correct", None),
    ("marginal_risk", _M_TOTAL, "wrong", "total"),
    ("marginal_risk", _M_REVERSE, "wrong", "reverse"),
    ("marginal_risk", _M_CHANCE, "wrong", "chance"),
    ("marginal_risk", _M_RATIO, "wrong", "ratio"),
    ("marginal_risk", _M_EARLY, "wrong", "early"),
    ("marginal_risk", "def marginal_risk(threats):\n    return 0.42\n", "almost", None),
    ("risk_over_time", _R_START, "empty", None),
    ("risk_over_time", _R, "correct", None),
    ("risk_over_time", "def risk_over_time(start, extras):\n    risks = []\n    for extra in extras:\n        start += extra\n        risks.append(start)\n    return risks\n", "correct", None),
    ("risk_over_time", "def risk_over_time(start, extras):\n    risks = []\n    for i in range(len(extras)):\n        risks.append(start + sum(extras[:i + 1]))\n    return risks\n", "correct", None),
    ("risk_over_time", _R_ZERO, "wrong", "zero"),
    ("risk_over_time", _R_EACH, "wrong", "each"),
    ("risk_over_time", _R_BEFORE, "wrong", "before"),
    ("risk_over_time", _R_EXTRA, "wrong", "extra"),
    ("risk_over_time", _R_EARLY, "wrong", "early"),
    ("risk_over_time", "def risk_over_time(start, extras):\n    return [8.42, 8.67, 8.92]\n", "almost", None),
]
