# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-boxing-and-tool-ai: boxes, oracles and tools. The learner writes checks_needed, how many
answers of an oracle to check at random to catch a false one with a given chance (after the test questions of
Armstrong, Sandberg and Bostrom, 2012), and best, which picks a control protocol from Table 7 of Greenblatt et al.
(2024): the highest safety for a floor on usefulness. A demo lets the red team choose how often to add a backdoor
against upfront auditing, in the course's own reading of the paper's model, and shows what is left of safety."""
import contextlib
import copy
import html
import io
import re

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the control protocols: Table 7 of Greenblatt, Shlegeris, Sachan and Roger (version of 17 July 2024) ----------

RESULTS = {   # name: (safety, usefulness), in percent, in the order of the paper's Table 7
    "Trusted model": (100, 2),
    "Upfront auditing": (15, 100),
    "Trusted monitoring (defer 0%)": (62, 97),
    "Trusted monitoring (defer 5%)": (74, 95),
    "Trusted monitoring (defer 10%)": (83, 93),
    "Trusted monitoring (defer 20%)": (92, 83),
    "Trusted monitoring (defer 50%)": (100, 51),
    "Untrusted monitoring (defer 0%)": (79, 96),
    "Untrusted monitoring (defer 5%)": (82, 95),
    "Untrusted monitoring (defer 10%)": (92, 92),
    "Untrusted monitoring (defer 20%)": (98, 85),
    "Trusted editing (5%)": (73, 97),
    "Trusted editing (10%)": (85, 96),
    "Trusted editing (20%)": (92, 94),
    "Trusted editing (50%)": (96, 89),
}


def best(protocols, floor):
    """The name of the protocol with the highest safety among those with a usefulness of at least floor; of two that
    are equally safe, the more useful one; None when none is useful enough."""
    choice, top_s, top_u = None, -1, -1
    for name, scores in protocols.items():
        safety, usefulness = scores
        if usefulness >= floor:
            if safety > top_s or (safety == top_s and usefulness > top_u):
                choice, top_s, top_u = name, safety, usefulness
    return choice


# --- random checks on an oracle, after Armstrong, Sandberg and Bostrom (2012) -------------------------------------

def caught(f: float, k: int) -> float:
    """The chance that k answers checked at random hold at least one false answer, when a share f is false."""
    return 1 - (1 - f) ** k


def checks_needed(f, target):
    """The smallest number of random checks that catch at least one false answer with a chance of at least target."""
    k, missed = 0, 1.0
    while 1 - missed < target:
        missed = missed * (1 - f)
        k = k + 1
    return k


# --- upfront auditing against a rare attack: the course's own reading of the paper's model (p. 8) ------------------

PROBLEMS, AUDITED, CORRECT = 5000, 100, 0.53   # 5,000 problems, the first 2% audited, GPT-4 right on 53% of them


def red_wins(p: float, problems: int = PROBLEMS, audited: int = AUDITED, correct: float = CORRECT) -> float:
    """The chance that the red team wins when its attack adds a backdoor to each problem with probability p: no
    attempt among the audited first problems, and at least one attempt that works later, which needs a correct
    solution. The course's own reading of the paper's model; its lowest safety matches the paper's 15%."""
    return (1 - p) ** audited * (1 - (1 - correct * p) ** (problems - audited))


def safety(p: float, correct: float = CORRECT) -> float:
    """The blue team's chance: no backdoor gets through."""
    return 1 - red_wins(p, correct=correct)


P_VALUES = [round(0.01 * k, 2) for k in range(1, 61)]   # the demo's stops: p from 0.01% to 0.60%, in percent
P_START = 0.05                                           # the demo starts at 0.05%


def lowest(values: list = P_VALUES) -> tuple:
    """The stop of the demo where safety is lowest, in percent, and that safety."""
    return min(((v, safety(v / 100)) for v in values), key=lambda t: t[1])


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


def number(v: float, lang: str, digits: int = 2) -> str:
    text = f"{v:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def percent(v: float, lang: str, digits: int = 1) -> str:
    """A chance as a percentage: 15.2% in English, 15,2% in Dutch."""
    return number(100 * v, lang, digits) + "%"


def show_value(v) -> str:
    """A result as the check shows it: as Python shows it, cut short when it is long."""
    text = repr(v)
    return text if len(text) <= 200 else text[:197] + "..."


def _whole(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def same_count(got, want) -> bool:
    """Whether the learner's number of checks is the course's: the same whole number."""
    return _whole(got) and got == want


def same_name(got, want) -> bool:
    """Whether the learner's protocol is the course's: the same name, or both None."""
    return type(got) is type(want) and got == want


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


def _rows(fn, cases: list, function, box, compare, words: dict) -> tuple:
    """The rows of the table under a check: the call, the course's result and the learner's, a dashed row where they
    differ; and the learner's results."""
    got = _all(fn, cases)
    rows = []
    for i, case in enumerate(cases):
        want = function(*copy.deepcopy(case))
        if got[i] is _STOPPED:
            shown_, off = html.escape(words["stopped"]), True
        else:
            shown_, off = _code(show_value(got[i])), not compare(got[i], want)
        rows.append(([(_code(box._call_text(case)), ""), (_code(show_value(want)), ""), (shown_, "font-weight:600")],
                     _OFF if off else ""))
    return rows, got


# the picture of *The idea*: trusted monitoring as a flow of boxes

_FLOW_WORDS = {
    "en": {"title": "Trusted monitoring", "u": ["U: GPT-4", "writes the code"], "t": ["T: GPT-3.5", "rates how", "suspicious it is"],
           "h": ["most suspicious 2%:", "H audits; a backdoor", "shuts U down"], "rest": ["the other 98%:", "submitted", "unchecked"],
           "alt": "A problem goes to U, GPT-4, which writes the code. T, GPT-3.5, rates how suspicious the code is. The most "
                  "suspicious 2% go to H for an audit, and a backdoor found there shuts U down; the other 98% are submitted unchecked.",
           "note": "U is the untrusted model, T the trusted model, H the small budget of careful audits. "
                   "After Greenblatt et al. (2024), Table 2."},
    "nl": {"title": "Trusted monitoring", "u": ["U: GPT-4", "schrijft de code"], "t": ["T: GPT-3.5", "geeft aan hoe", "verdacht die is"],
           "h": ["verdachtste 2%:", "H controleert; een", "backdoor zet U uit"], "rest": ["de andere 98%:", "zonder controle", "ingediend"],
           "alt": "Een probleem gaat naar U, GPT-4, dat de code schrijft. T, GPT-3.5, geeft aan hoe verdacht de code is. De "
                  "verdachtste 2% gaan naar H voor controle, en een backdoor die daar gevonden wordt, zet U uit; de andere 98% worden zonder controle ingediend.",
           "note": "U is het untrusted model, T het trusted model, H het kleine budget voor zorgvuldige controles. "
                   "Naar Greenblatt et al. (2024), tabel 2."},
}


def _box(x: float, y: float, w: float, h: float, lines: list, strong: bool = False) -> str:
    ink, line = "var(--ea-ink)", ("var(--ea-ink)" if strong else "var(--ea-ink-2)")
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" style="fill:var(--ea-page, transparent);stroke:{line};stroke-width:{1.8 if strong else 1.4}"/>']
    top = y + h / 2 - (len(lines) - 1) * 9 + 5
    for i, text in enumerate(lines):
        weight = ";font-weight:600" if i == 0 else ""
        out.append(f'<text x="{x + w / 2}" y="{top + 18 * i}" text-anchor="middle" style="fill:{ink};font-size:14px{weight}">{html.escape(text)}</text>')
    return "".join(out)


def _arrow(x0: float, y0: float, x1: float, y1: float) -> str:
    n = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 or 1.0
    ux, uy = (x1 - x0) / n, (y1 - y0) / n
    px, py = -uy, ux
    pts = [(x1, y1), (x1 - 9 * ux + 4.5 * px, y1 - 9 * uy + 4.5 * py), (x1 - 9 * ux - 4.5 * px, y1 - 9 * uy - 4.5 * py)]
    c = "var(--ea-ink-2)"
    return (f'<line x1="{x0}" y1="{y0}" x2="{x1 - 8 * ux:.1f}" y2="{y1 - 8 * uy:.1f}" style="stroke:{c};stroke-width:1.6"/>'
            f'<polygon points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" style="fill:{c}"/>')


def flow_svg(lang: str) -> str:
    """Trusted monitoring as boxes: U writes, T rates, the most suspicious 2% go to H, the rest is submitted. Words are
    14 px in a drawing 470 wide, so they stay 11 px or more down to 370 px."""
    w = _FLOW_WORDS[lang]
    parts = [_box(4, 62, 120, 60, w["u"], strong=True), _arrow(124, 92, 150, 92),
             _box(150, 54, 124, 76, w["t"]), _arrow(274, 80, 302, 40), _arrow(274, 104, 302, 146),
             _box(302, 4, 164, 72, w["h"]), _box(302, 106, 164, 70, w["rest"])]
    return (f'<svg viewBox="0 0 470 180" role="img" aria-label="{html.escape(w["alt"])}" style="width:100%;'
            f'max-width:min(100%,470px);height:auto;font-family:var(--ea-font-ui)">{"".join(parts)}</svg>')


def flow_picture(lang: str) -> str:
    w = _FLOW_WORDS[lang]
    return yours.figure(w["title"], flow_svg(lang) + _note(w["note"]), alt=w["alt"])


# exercise 1: checks_needed, drawn as the chance of catching against the number of checks

_CHECKS_WORDS = {
    "en": {"title": "Your random checks", "x": "answers checked at random, k", "y": "chance to catch a false one",
           "target": "target 0.95", "head": ["the call", "the course's", "yours"], "stopped": "error",
           "note": "The curve is the first case: 1% of the answers false, so k checks catch at least one false answer with "
                   "a chance of 1 − 0.99 to the power k. The dashed line is the target of 0.95. The dot is the course's "
                   "number of checks, the ring yours. The table is every case of the check; a dashed row is one where your "
                   "function gives something other than the course's."},
    "nl": {"title": "Jouw willekeurige controles", "x": "willekeurig gecontroleerde antwoorden, k", "y": "kans om een fout te vinden",
           "target": "doel 0,95", "head": ["de aanroep", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De curve is het eerste geval: 1% van de antwoorden is fout, dus k controles vinden minstens één fout "
                   "antwoord met een kans van 1 − 0,99 tot de macht k. De gestippelde lijn is het doel van 0,95. De stip is "
                   "het aantal controles van de course, de ring het jouwe. De tabel bevat elk geval van de check; een "
                   "gestippelde rij is er een waar jouw functie iets anders geeft dan die van de course."},
}


def checks_picture(seen) -> str:
    """Under the check of exercise 1: for the first case, the chance of catching a false answer against the number
    of checks, with the target, the course's answer as a dot and the learner's as a ring; every case in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CHECKS_WORDS[seen.lang]
    rows, got = _rows(fn, CHECKS_CASES, checks_needed, EXERCISES["checks"], same_count, w)
    f, target = CHECKS_CASES[0]
    panel = chart.Panel(x=(0, 400), y=(0, 1.0), xlabel=w["x"], ylabel=w["y"], xticks=[0, 100, 200, 300, 400],
                        yticks=[0, 0.25, 0.5, 0.75, 1.0])
    ks = list(range(0, 401, 4))
    panel.line(ks, [caught(f, k) for k in ks], role="ink-2", width=2.0)
    panel.line([0, 400], [target, target], role="hyper", width=1.6, dashed=True)
    panel.text(60, target + 0.05, w["target"], role="ink-2")
    want = checks_needed(f, target)
    mine = got[0]
    if _whole(mine) and 0 <= mine <= 400:
        panel.points([mine], [caught(f, mine)], role="ink", r=8, hollow=True)
    panel.points([want], [caught(f, want)], role="ink", r=4)
    panel.text(want, caught(f, want) - 0.09, str(want), role="ink")
    svg = chart.figure([panel], alt=w["title"], width=420, height=250)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# exercise 2: best, drawn as the protocols of Table 7 with the floor

_BEST_WORDS = {
    "en": {"title": "Your choice of protocol", "x": "usefulness (%)", "y": "safety (%)", "floor": "floor 90",
           "head": ["the call", "the course's", "yours"], "stopped": "error",
           "note": "Each dot is a row of Table 7 of Greenblatt et al. (2024). The dashed line is the floor of the first case, "
                   "90: the rows left of it are not useful enough, drawn light. The filled dot is the course's choice, the "
                   "ring yours. The table is every case of the check; a dashed row is one where your function gives something "
                   "other than the course's."},
    "nl": {"title": "Jouw keuze van protocol", "x": "usefulness (%)", "y": "safety (%)", "floor": "ondergrens 90",
           "head": ["de aanroep", "van de course", "de jouwe"], "stopped": "fout",
           "note": "Elke stip is een rij uit tabel 7 van Greenblatt et al. (2024). De gestippelde lijn is de ondergrens van het "
                   "eerste geval, 90: de rijen links ervan zijn niet nuttig genoeg, licht getekend. De volle stip is de "
                   "keuze van de course, de ring de jouwe. De tabel bevat elk geval van de check; een gestippelde rij is er een "
                   "waar jouw functie iets anders geeft dan die van de course."},
}


def best_picture(seen) -> str:
    """Under the check of exercise 2: the rows of Table 7 as dots of safety against usefulness, with the floor of the
    first case, the course's choice filled and the learner's as a ring; every case in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _BEST_WORDS[seen.lang]
    rows, got = _rows(fn, BEST_CASES, best, EXERCISES["best"], same_name, w)
    protocols, floor = BEST_CASES[0]
    panel = chart.Panel(x=(0, 105), y=(0, 108), xlabel=w["x"], ylabel=w["y"], xticks=[0, 25, 50, 75, 100],
                        yticks=[0, 25, 50, 75, 100])
    panel.line([floor, floor], [0, 108], role="hyper", width=1.6, dashed=True)
    panel.text(floor - 2, 6, w["floor"], role="ink-2", anchor="end")   # low on the line, where no row of the table is
    out = [(u, s) for s, u in protocols.values() if u < floor]
    ok = [(u, s) for s, u in protocols.values() if u >= floor]
    panel.points([u for u, _ in out], [s for _, s in out], role="ink-3", r=3.5)
    panel.points([u for u, _ in ok], [s for _, s in ok], role="ink-2", r=3.5)
    mine = got[0]
    if isinstance(mine, str) and mine in protocols:
        s, u = protocols[mine]
        panel.points([u], [s], role="ink", r=8, hollow=True)
    s, u = protocols[best(protocols, floor)]
    panel.points([u], [s], role="ink", r=5)
    svg = chart.figure([panel], alt=w["title"], width=420, height=260)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the demo: how often should the red team attack? ---------------------------------------------------------------

DEMO_WORDS = {
    "en": {"title": "Upfront auditing against a rare attack", "x": "p: chance of a backdoor per problem (%)",
           "y": "safety (%)", "paper": "15% in the paper", "here": "p = {p}%"},
    "nl": {"title": "Upfront auditing tegen een zeldzame aanval", "x": "p: kans op een backdoor per probleem (%)",
           "y": "safety (%)", "paper": "15% in de paper", "here": "p = {p}%"},
}


def demo_frame(v: float, lang: str) -> tuple:
    """The demo at p = v percent: safety against p, with a dot at v and the paper's 15% dashed; and the sentence under
    it, with the chances that make up the red team's win."""
    w = DEMO_WORDS[lang]
    p = v / 100
    panel = chart.Panel(x=(0, 0.62), y=(0, 100), xlabel=w["x"], ylabel=w["y"], xticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                        yticks=[0, 25, 50, 75, 100])
    xs = [0.005 * k for k in range(1, 125)]
    panel.line([0.0, 0.62], [15, 15], role="ink-3", width=1.4, dashed=True)
    panel.text(0.5, 20, w["paper"], role="ink-3")
    panel.line(xs, [100 * safety(x / 100) for x in xs], role="ink-2", width=2.2)
    panel.points([v], [100 * safety(p)], role="ink", r=5.5)
    # the words go where the curve is not: to the right while it falls, above it near the bottom, to the left once it rises
    here, y = w["here"].format(p=number(v, lang)), min(100 * safety(p) + 7, 97)
    if v < 0.08:
        panel.text(v + 0.015, y, here, role="ink", anchor="start", halo=True)
    elif v < 0.25:
        panel.text(max(v + 0.03, 0.14), y + 9, here, role="ink", halo=True)
    else:
        panel.text(v - 0.015, y, here, role="ink", anchor="end", halo=True)
    svg = chart.figure([panel], alt=demo_alt(v, lang), width=440, height=250)
    return svg, demo_said(v, lang)


def demo_alt(v: float, lang: str) -> str:
    if lang == "nl":
        return (f"Safety tegen p, de kans op een backdoor per probleem: hoog bij een heel kleine p, het laagst rond 0,13%, "
                f"daarna weer hoger; een stip bij p = {number(v, lang)}%, met safety {percent(safety(v / 100), lang)}.")
    return (f"Safety against p, the chance of a backdoor per problem: high for a very small p, lowest near 0.13%, then "
            f"higher again; a dot at p = {number(v, lang)}%, with safety {percent(safety(v / 100), lang)}.")


def demo_said(v: float, lang: str) -> str:
    """What the sentence under the demo says at p = v percent."""
    p = v / 100
    attempts = number(PROBLEMS * p, lang, 1)
    clean = (1 - p) ** AUDITED
    later = 1 - (1 - CORRECT * p) ** (PROBLEMS - AUDITED)
    if lang == "nl":
        return (f"Met p = {number(v, lang)}% probeert GPT-4 gemiddeld {attempts} keer een backdoor in 5000 problemen. De kans "
                f"dat geen poging in de eerste 100 valt, die gecontroleerd worden, is {percent(clean, lang)}; de kans dat daarna "
                f"minstens één poging werkt, is {percent(later, lang)}. Een backdoor komt erdoor met een kans van {percent(clean * later, lang)}, "
                f"dus safety is **{percent(safety(p), lang)}**.")
    return (f"With p = {number(v, lang)}%, GPT-4 tries a backdoor {attempts} times in 5,000 problems on average. The chance "
            f"that no attempt falls in the first 100, which are audited, is {percent(clean, lang)}; the chance that at least "
            f"one attempt works after them is {percent(later, lang)}. A backdoor gets through with a chance of {percent(clean * later, lang)}, "
            f"so safety is **{percent(safety(p), lang)}**.")


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["checks", "best", "oracle", "agents"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# exercise 1: checks_needed; the first case is 1% false answers and a chance of 95%
CHECKS_CASES = [
    (0.01, 0.95),
    (0.05, 0.95),
    (0.5, 0.9),
    (0.1, 0.5),
    (0.5, 0.5),
    (0.2, 0.99),
    (0.003, 0.95),
]
_C_DEF = "def checks_needed(f, target):\n"
_C_START = _C_DEF + "    ...\n"
_C = (_C_DEF +
      "    k, missed = 0, 1.0\n"
      "    while 1 - missed < target:\n"
      "        missed = missed * (1 - f)\n"
      "        k = k + 1\n"
      "    return k\n")
_C_LE = _C.replace("while 1 - missed < target:", "while 1 - missed <= target:")
_C_F = _C.replace("missed = missed * (1 - f)", "missed = missed * f")
_C_PLUS = (_C_DEF +
           "    k, missed = 0, 1.0\n"
           "    while 1 - missed < target:\n"
           "        missed = (1 - f) ** k\n"
           "        k = k + 1\n"
           "    return k\n")
_C_MISSED = _C.replace("while 1 - missed < target:", "while missed < target:")
_C_LINEAR = _C.replace("missed = missed * (1 - f)", "missed = missed - f")

# exercise 2: best; the first case is Table 7 with a floor of 90, where two rows have the highest safety, 92
TIES = {"A": (90, 80), "B": (90, 70), "C": (60, 99)}   # the course's own: two equally safe protocols, the first more useful
BEST_CASES = [
    (RESULTS, 90),
    (RESULTS, 97),
    (RESULTS, 100),
    (RESULTS, 85),
    (RESULTS, 0),
    (RESULTS, 101),
    (TIES, 50),
]
_B_DEF = "def best(protocols, floor):\n"
_B_START = _B_DEF + "    ...\n"
_B = (_B_DEF +
      "    choice, top_s, top_u = None, -1, -1\n"
      "    for name, scores in protocols.items():\n"
      "        safety, usefulness = scores\n"
      "        if usefulness >= floor:\n"
      "            if safety > top_s or (safety == top_s and usefulness > top_u):\n"
      "                choice, top_s, top_u = name, safety, usefulness\n"
      "    return choice\n")
_B_NOFLOOR = (_B_DEF +
              "    choice, top_s, top_u = None, -1, -1\n"
              "    for name, scores in protocols.items():\n"
              "        safety, usefulness = scores\n"
              "        if safety > top_s or (safety == top_s and usefulness > top_u):\n"
              "            choice, top_s, top_u = name, safety, usefulness\n"
              "    return choice\n")
_B_STRICT = _B.replace("if usefulness >= floor:", "if usefulness > floor:")
_B_NOTIE = _B.replace("if safety > top_s or (safety == top_s and usefulness > top_u):", "if safety > top_s:")
_B_LAST = _B.replace("if safety > top_s or (safety == top_s and usefulness > top_u):", "if safety >= top_s:")
_B_USEFUL = _B.replace("if safety > top_s or (safety == top_s and usefulness > top_u):",
                       "if usefulness > top_u or (usefulness == top_u and safety > top_s):")
_B_SCORES = _B.replace("    return choice\n", "    return top_s, top_u\n")
_B_FIRST = _B.replace("    return choice\n", "        return choice\n")

_NAMED = [(RESULTS, "results"), (TIES, "TIES")]   # the data the texts call by name


def _lit(value) -> str:
    """A value the way the lesson writes it in code: results and TIES by name, texts in double quotes."""
    for data, name in _NAMED:
        if type(value) is type(data) and value == data:
            return name
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, tuple):
        return "(" + ", ".join(_lit(v) for v in value) + ("," if len(value) == 1 else "") + ")"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_lit(k)}: {_lit(v)}" for k, v in value.items()) + "}"
    return repr(value)


class NamedBox(FunctionBox):
    """A FunctionBox whose messages write a call the way the lesson does: `best(results, 90)`."""

    def _call_text(self, case) -> str:
        return f"{self.function}({', '.join(_lit(a) for a in case)})"


EXERCISES = {
    "checks": NamedBox(
        name="checks",
        picture=checks_picture,
        function="checks_needed",
        start=_C_START,
        solution=_C,
        visible=[CHECKS_CASES[0]],
        hidden=CHECKS_CASES[1:],
        first=[_DOTS],
        show=show_value,
        compare=same_count,
        mistakes=[
            Mistake("le", wrong=_C_LE,
                    en="`{call}` gives {got}, and it should give {want}: your loop goes on when the chance of catching is exactly `target`. The task asks for a chance of at least `target`, so stop as soon as it gets there.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je loop gaat door als de kans om een fout te vinden precies `target` is. De opdracht vraagt een kans van minstens `target`, dus stop zodra die bereikt is."),
            Mistake("f", wrong=_C_F,
                    en="`{call}` gives {got}, and it should give {want}: your `missed` uses `f`, the chance that a checked answer is false. But `missed` is the chance that every checked answer is true.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `missed` gebruikt `f`, de kans dat een gecontroleerd antwoord fout is. Maar `missed` is de kans dat elk gecontroleerd antwoord goed is."),
            Mistake("plus", wrong=_C_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: one check too many. When the loop stops, `missed` must belong to the `k` you give back, so work out `missed` for the new number of checks.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: één controle te veel. Als de loop stopt, moet `missed` horen bij de `k` die je teruggeeft, dus reken `missed` uit voor het nieuwe aantal controles."),
            Mistake("missed", wrong=_C_MISSED,
                    en="`{call}` gives {got}, and it should give {want}: your loop compares `missed`, the chance that every check finds a true answer, with `target`. Compare the chance of catching a false one: `1 - missed`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je loop vergelijkt `missed`, de kans dat elke controle een goed antwoord vindt, met `target`. Vergelijk de kans om een fout antwoord te vinden: `1 - missed`."),
            Mistake("linear", wrong=_C_LINEAR,
                    en="`{call}` gives {got}, and it should give {want}: your `missed` goes down by `f` with every check, as if k checks caught a false answer with a chance of k times f. But two checks can both find a false answer, and adding the chances counts that case twice.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `missed` gaat met elke controle `f` omlaag, alsof k controles een fout antwoord vinden met een kans van k keer f. Maar twee controles kunnen allebei een fout antwoord vinden, en optellen telt dat geval dubbel."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start `k` at `0` and `missed` at `1.0`, loop while `1 - missed` is below `target`, and give back `k`.",
                      "nl": "Vervang de `...`: begin `k` bij `0` en `missed` bij `1.0`, ga door zolang `1 - missed` onder `target` ligt, en geef `k` terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. `missed` is the chance that all `k` checked answers are true; in each round of the loop, it becomes that chance for one more check, and `k` goes up by 1. The loop stops at the first `k` where `1 - missed` reaches `target`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. `missed` is de kans dat alle `k` gecontroleerde antwoorden goed zijn; in elke ronde van de loop wordt die de kans voor één controle meer, en gaat `k` 1 omhoog. De loop stopt bij de eerste `k` waarbij `1 - missed` `target` haalt."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `f` and `target` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie wel de `f` en `target` uit de aanroep?"},
            "correct": {"en": "Your `checks_needed` is right in all {n} cases. With 1% false answers, 299 random checks catch at least one with a chance of 95%.",
                        "nl": "Je `checks_needed` klopt in alle {n} gevallen. Met 1% foute antwoorden vinden 299 willekeurige controles er met een kans van 95% minstens één."},
        },
    ),
    "best": NamedBox(
        name="best",
        picture=best_picture,
        function="best",
        start=_B_START,
        solution=_B,
        visible=[BEST_CASES[0]],
        hidden=BEST_CASES[1:],
        first=[_DOTS],
        show=show_value,
        compare=same_name,
        mistakes=[
            Mistake("nofloor", wrong=_B_NOFLOOR,
                    en="`{call}` gives {got}, and it should give {want}: your function picks the safest protocol of all, including the ones below the floor. Skip a protocol whose `usefulness` is below `floor`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie kiest het veiligste protocol van allemaal, ook de protocollen onder de ondergrens. Sla een protocol over als de `usefulness` onder `floor` ligt."),
            Mistake("strict", wrong=_B_STRICT,
                    en="`{call}` gives {got}, and it should give {want}: a protocol whose usefulness is exactly `floor` drops out in your function. A usefulness of at least `floor` is enough.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een protocol met een usefulness van precies `floor` valt bij jou af. Een usefulness van minstens `floor` is genoeg."),
            Mistake("notie", wrong=_B_NOTIE,
                    en="`{call}` gives {got}, and it should give {want}: of two protocols with the same safety, your function keeps the first it meets. Then the more useful one should win.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: van twee protocollen met dezelfde safety houdt je functie het eerste protocol dat ze tegenkomt. Dan hoort het nuttigste te winnen."),
            Mistake("last", wrong=_B_LAST,
                    en="`{call}` gives {got}, and it should give {want}: of two protocols with the same safety, your function keeps the last it meets, also when that one is less useful. Then the more useful one should win.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: van twee protocollen met dezelfde safety houdt je functie het laatste protocol dat ze tegenkomt, ook als dat minder nuttig is. Dan hoort het nuttigste te winnen."),
            Mistake("useful", wrong=_B_USEFUL,
                    en="`{call}` gives {got}, and it should give {want}: your function picks the most useful protocol. Usefulness only has to pass the floor; among those that pass, the highest safety wins.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie kiest het nuttigste protocol. De usefulness hoeft alleen de ondergrens te halen; van de protocollen die dat doen, wint de hoogste safety."),
            Mistake("scores", wrong=_B_SCORES,
                    en="`{call}` gives {got}, and it should give {want}: you give back the safety and usefulness of the best protocol. Give back its name, `choice`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft de safety en usefulness van het beste protocol terug. Geef de naam terug, `choice`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go through `protocols.items()`, keep the best name so far in `choice`, and give back `choice`.",
                      "nl": "Vervang de `...`: loop door `protocols.items()`, houd de beste naam tot nu toe bij in `choice`, en geef `choice` terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Of the protocols with a usefulness of at least `floor`, the safest wins; of two that are equally safe, the more useful one; and with none useful enough, `None`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Van de protocollen met een usefulness van minstens `floor` wint het veiligste; van twee die even veilig zijn, het nuttigste; en als geen enkel nuttig genoeg is, `None`."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `protocols` and the `floor` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie wel de `protocols` en de `floor` uit de aanroep?"},
            "correct": {"en": "Your `best` is right in all {n} cases. At a floor of 90, trusted editing (20%) gives 92% safety, and wins from untrusted monitoring (defer 10%) on usefulness.",
                        "nl": "Je `best` klopt in alle {n} gevallen. Bij een ondergrens van 90 geeft trusted editing (20%) 92% safety, en wint het van untrusted monitoring (defer 10%) op usefulness."},
        },
    ),
}

CASES = [
    ("checks", _C_START, "empty", None),
    ("checks", _C, "correct", None),
    ("checks", _C_DEF + "    k = 0\n    while 1 - (1 - f) ** k < target:\n        k = k + 1\n    return k\n", "correct", None),
    ("checks", _C_LE, "wrong", "le"),
    ("checks", _C_F, "wrong", "f"),
    ("checks", _C_PLUS, "wrong", "plus"),
    ("checks", _C_MISSED, "wrong", "missed"),
    ("checks", _C_LINEAR, "wrong", "linear"),
    ("checks", _C_DEF + "    return 299\n", "almost", None),
    ("best", _B_START, "empty", None),
    ("best", _B, "correct", None),
    ("best", _B_DEF + "    allowed = [(s, u, name) for name, (s, u) in protocols.items() if u >= floor]\n"
                      "    return max(allowed)[2] if allowed else None\n", "correct", None),
    ("best", _B_NOFLOOR, "wrong", "nofloor"),
    ("best", _B_STRICT, "wrong", "strict"),
    ("best", _B_NOTIE, "wrong", "notie"),
    ("best", _B_LAST, "wrong", "last"),
    ("best", _B_USEFUL, "wrong", "useful"),
    ("best", _B_SCORES, "wrong", "scores"),
    ("best", _B_FIRST, "wrong", "unreached"),   # the check's own message: a return in the loop gives None here
    ("best", _B_DEF + '    return "Trusted editing (20%)"\n', "almost", None),
]
