# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-dangerous-capabilities: testing for dangerous capabilities. The learner writes pass_at_k,
pass@k from n runs of which c succeeded (the unbiased estimator that RepliBench uses, after Chen et al., 2021), and
ruled_out, the smallest rate of success per attempt that a test with no success at all rules out at 95%. A demo
shows, for a number of attempts without a success, which rates such a test rules out and which it cannot."""
import contextlib
import copy
import html
import io
import re

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- pass@k from runs already done -----------------------------------------------------------------------------------

def pass_at_k(n, c, k):
    """The chance that k of the n runs, picked at random without picking one twice, hold at least one of the c runs
    that succeeded: the unbiased estimator of pass@k (Chen et al., 2021), as RepliBench uses it with n = 25, k = 10."""
    missed = 1.0
    for i in range(k):
        missed = missed * (n - c - i) / (n - i)
    return 1 - missed


def any_pass(p: float, k: int) -> float:
    """The chance that at least one of k independent attempts passes, when each passes with chance p."""
    return 1 - (1 - p) ** k


# --- what a test without a single success rules out ------------------------------------------------------------------

RATES = [k / 100 for k in range(1, 100)]   # the rates of Exercise 2: 0.01, 0.02, ..., 0.99


def ruled_out(attempts, rates):
    """The smallest rate in rates for which zero successes in this many attempts would come with a chance below 0.05;
    None when no rate in the list is ruled out."""
    for rate in rates:
        if (1 - rate) ** attempts < 0.05:
            return rate
    return None


def bound(n: int) -> float:
    """The rate above which n attempts without a success rule a rate out at 95%: 1 - 0.05 to the power 1/n."""
    return 1 - 0.05 ** (1 / n)


def edge_text(n: int, lang: str) -> str:
    """The bound after n attempts as the demo writes it, in percent: two decimals below 10%, so that 58 attempts
    (5.03%) and 59 (4.95%) read differently."""
    edge = 100 * bound(n)
    return number(edge, lang, 2 if edge < 10 else 1)


N_VALUES = list(range(1, 101))   # the demo's stops: 1 to 100 attempts
N_START = 10


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


def number(v: float, lang: str, digits: int = 1) -> str:
    text = f"{v:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def percent(v: float, lang: str, digits: int = 1) -> str:
    """A chance as a percentage: 7.7% in English, 7,7% in Dutch."""
    return number(100 * v, lang, digits) + "%"


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def show_value(v) -> str:
    """A result as the check shows it: a number to at most six decimals, anything else as Python shows it, cut short
    when it is long."""
    text = repr(round(v, 6)) if isinstance(v, float) else repr(v)
    return text if len(text) <= 200 else text[:197] + "..."


def same_chance(got, want) -> bool:
    """Whether the learner's pass@k is the course's: a number within a billionth."""
    return _num(got) and abs(got - want) < 1e-9


def same_rate(got, want) -> bool:
    """Whether the learner's rate is the course's: the same number, or both None."""
    if want is None:
        return got is None
    return _num(got) and abs(got - want) < 1e-9


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


# exercise 1: pass_at_k, drawn as pass@k against k for 25 runs

_PASS_WORDS = {
    "en": {"title": "Your pass@k", "x": "runs picked, k", "y": "pass@k", "head": ["the call", "the course's", "yours"],
           "stopped": "error",
           "note": "The lines are pass@k for 25 runs with 1, 2, 3 and 5 successes, from bottom to top; the dashed line is "
                   "k = 10, as in RepliBench. The dots are the course's pass@10 for the cases with 25 runs, the rings "
                   "yours. The table shows every case of the check; a dashed row is one where your function gives something "
                   "other than the course's."},
    "nl": {"title": "Jouw pass@k", "x": "gekozen runs, k", "y": "pass@k", "head": ["de aanroep", "van de course", "de jouwe"],
           "stopped": "fout",
           "note": "De lijnen zijn pass@k voor 25 runs met 1, 2, 3 en 5 successen, van onder naar boven; de gestippelde lijn "
                   "is k = 10, zoals bij RepliBench. De stippen zijn de pass@10 van de course voor de gevallen met 25 runs, "
                   "de ringen de jouwe. De tabel bevat elk geval van de check; een gestippelde rij is er een waar jouw "
                   "functie iets anders geeft dan die van de course."},
}


def pass_picture(seen) -> str:
    """Under the check of exercise 1: pass@k against k for 25 runs with a few numbers of successes, the course's
    pass@10 as dots and the learner's as rings; every case in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _PASS_WORDS[seen.lang]
    rows, got = _rows(fn, PASS_CASES, pass_at_k, EXERCISES["pass"], same_chance, w)
    panel = chart.Panel(x=(0, 25), y=(0, 1.0), xlabel=w["x"], ylabel=w["y"], xticks=[0, 5, 10, 15, 20, 25],
                        yticks=[0, 0.25, 0.5, 0.75, 1.0])
    panel.line([10, 10], [0, 1.0], role="hyper", width=1.4, dashed=True)
    ks = list(range(0, 26))
    for c in (1, 2, 3, 5):
        panel.line(ks, [pass_at_k(25, c, k) for k in ks], role="ink-3", width=1.6)
    for i, (n, c, k) in enumerate(PASS_CASES):
        if n == 25 and k == 10:
            if _num(got[i]) and 0 <= got[i] <= 1:
                panel.points([10], [got[i]], role="ink", r=8, hollow=True)
            panel.points([10], [pass_at_k(n, c, k)], role="ink", r=4)
    svg = chart.figure([panel], alt=w["title"], width=420, height=250)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# exercise 2: ruled_out, drawn as the smallest rate ruled out against the number of attempts

_RULED_WORDS = {
    "en": {"title": "What no success rules out", "x": "attempts without a success", "y": "smallest rate ruled out",
           "head": ["the call", "the course's", "yours"], "stopped": "error",
           "note": "The line is the course's answer for the rates 0.01 to 0.99, for 1 to 100 attempts without a success. "
                   "The dots are the course's answers for the cases with these rates and at most 100 attempts, the rings "
                   "yours. The table shows every case of the check; a dashed row is one where your function gives something other than the course's."},
    "nl": {"title": "Wat geen enkel succes uitsluit", "x": "pogingen zonder succes", "y": "kleinste uitgesloten kans",
           "head": ["de aanroep", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De lijn is het antwoord van de course voor de kansen 0,01 tot en met 0,99, bij 1 tot en met 100 pogingen "
                   "zonder succes. De stippen zijn de antwoorden van de course voor de gevallen met deze kansen en hoogstens "
                   "100 pogingen, de ringen de jouwe. De tabel bevat elk geval van de check; een gestippelde rij is er een waar jouw functie iets "
                   "anders geeft dan die van de course."},
}


def ruled_picture(seen) -> str:
    """Under the check of exercise 2: the smallest rate ruled out against the number of attempts, the course's
    answers as dots and the learner's as rings; every case in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _RULED_WORDS[seen.lang]
    rows, got = _rows(fn, RULED_CASES, ruled_out, EXERCISES["ruled"], same_rate, w)
    panel = chart.Panel(x=(0, 100), y=(0, 1.0), xlabel=w["x"], ylabel=w["y"], xticks=[0, 25, 50, 75, 100],
                        yticks=[0, 0.25, 0.5, 0.75, 1.0])
    ns = list(range(1, 101))
    panel.line(ns, [ruled_out(n, RATES) for n in ns], role="ink-3", width=1.8)
    for i, (n, rates) in enumerate(RULED_CASES):
        if rates == RATES and 1 <= n <= 100:
            if _num(got[i]) and 0 <= got[i] <= 1:
                panel.points([n], [got[i]], role="ink", r=8, hollow=True)
            panel.points([n], [ruled_out(n, rates)], role="ink", r=4)
    svg = chart.figure([panel], alt=w["title"], width=420, height=250)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the demo: what zero successes in n attempts rule out ---------------------------------------------------------------

DEMO_WORDS = {
    "en": {"x": "success rate per attempt (%)", "y": "chance of zero successes (%)", "line": "5%", "out": "ruled out",
           "here": "{r}%", "beyond": "ruled out: above {r}%"},
    "nl": {"x": "kans op succes per poging (%)", "y": "kans op nul successen (%)", "line": "5%", "out": "uitgesloten",
           "here": "{r}%", "beyond": "uitgesloten: boven {r}%"},
}


def demo_frame(n: int, lang: str) -> tuple:
    """The demo at n attempts without a success: the chance of zero successes against the rate per attempt, with the
    5% line and the rates it rules out; and the sentence under it."""
    w = DEMO_WORDS[lang]
    panel = chart.Panel(x=(0, 50), y=(0, 100), xlabel=w["x"], ylabel=w["y"], xticks=[0, 10, 20, 30, 40, 50],
                        yticks=[0, 25, 50, 75, 100])
    edge = 100 * bound(n)
    if edge < 50:
        panel.rect(edge, 0, 50, 100, role="ink-3")
        if edge < 38:
            panel.text((edge + 50) / 2, 88, w["out"], role="ink-2")
    rs = [0.5 * j for j in range(0, 101)]
    panel.line([0, 50], [5, 5], role="hyper", width=1.4, dashed=True)
    if edge >= 12:   # the label of the 5% line where the curve is high, away from the dot
        panel.text(2.5, 11, w["line"], role="ink-2", anchor="start")
    else:
        panel.text(49, 11, w["line"], role="ink-2", anchor="end", halo=True)
    panel.line(rs, [100 * (1 - r / 100) ** n for r in rs], role="ink-2", width=2.2)
    if edge < 50:
        panel.points([edge], [5], role="ink", r=5.5)
        if edge <= 38:
            panel.text(edge + 1, 17, w["here"].format(r=edge_text(n, lang)), role="ink", anchor="start", halo=True)
        else:   # near the right edge, the label goes to the left of the dot
            panel.text(edge - 1.5, 17, w["here"].format(r=edge_text(n, lang)), role="ink", anchor="end", halo=True)
    else:   # with 1 to 4 attempts, only rates above 50% are ruled out: beyond the picture
        panel.text(49, 88, w["beyond"].format(r=edge_text(n, lang)), role="ink-2", anchor="end", halo=True)
    svg = chart.figure([panel], alt=demo_alt(n, lang), width=440, height=250)
    if lang == "nl":
        svg = chart.with_decimal_comma(svg)
    return svg, demo_said(n, lang)


def demo_alt(n: int, lang: str) -> str:
    if lang == "nl":
        return (f"De kans op nul successen in {n} pogingen tegen de kans op succes per poging: 100% bij een kans van 0, "
                f"daarna dalend; kansen boven {edge_text(n, lang)}% geven nul successen met een kans onder 5% en "
                f"zijn uitgesloten.")
    return (f"The chance of zero successes in {n} attempts against the success rate per attempt: 100% at a rate of 0, then "
            f"falling; rates above {edge_text(n, lang)}% give zero successes with a chance below 5% and are ruled out.")


def demo_said(n: int, lang: str) -> str:
    """What the sentence under the demo says at n attempts without a success."""
    edge, five, one = edge_text(n, lang), percent(0.95 ** n, lang), percent(0.99 ** n, lang)
    if lang == "nl":
        poging = "poging" if n == 1 else "pogingen"
        return (f"Na nul successen in {n} {poging} zijn kansen boven **{edge}%** uitgesloten: een model met zo'n kans "
                f"per poging zou in meer dan 95% van zulke tests minstens één keer slagen. Een model met een kans van 5% "
                f"haalt nog in {five} van de tests nul successen, en een model met 1% in {one}.")
    attempt = "attempt" if n == 1 else "attempts"
    return (f"After zero successes in {n} {attempt}, rates above **{edge}%** are ruled out: a model with such a rate per "
            f"attempt would succeed at least once in more than 95% of such tests. A model with a rate of 5% still ends "
            f"with zero successes in {five} of the tests, and one with 1% in {one}.")


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["pass", "ruled", "claim", "story"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# exercise 1: pass_at_k; the first case is RepliBench's setting, 25 runs and pass@10, with one success
PASS_CASES = [
    (25, 1, 10),
    (25, 0, 10),
    (25, 2, 10),
    (25, 3, 10),
    (25, 5, 10),
    (25, 16, 10),
    (10, 3, 1),
    (4, 2, 2),
]
_P_DEF = "def pass_at_k(n, c, k):\n"
_P_START = _P_DEF + "    ...\n"
_P = (_P_DEF +
      "    missed = 1.0\n"
      "    for i in range(k):\n"
      "        missed = missed * (n - c - i) / (n - i)\n"
      "    return 1 - missed\n")
_P_REPLACE = _P.replace("missed * (n - c - i) / (n - i)", "missed * (n - c) / n")
_P_TOP = _P.replace("missed * (n - c - i) / (n - i)", "missed * (n - c - i) / n")
_P_SUCCESS = _P.replace("missed * (n - c - i) / (n - i)", "missed * (c - i) / (n - i)")
_P_START1 = _P.replace("for i in range(k):", "for i in range(1, k + 1):")
_P_PLUS = _P.replace("for i in range(k):", "for i in range(k + 1):")
_P_MISSED = _P.replace("    return 1 - missed\n", "    return missed\n")
_P_OVER = _P.replace("missed = missed * (n - c - i) / (n - i)", "missed = (n - c - i) / (n - i)")

# exercise 2: ruled_out; the first case is 10 attempts and the rates 0.01 to 0.99
RULED_CASES = [
    (10, RATES),
    (20, RATES),
    (50, RATES),
    (100, RATES),
    (2, RATES),
    (300, RATES),
    (0, RATES),
    (10, [0.1, 0.2, 0.3]),
    (10, [0.1, 0.2]),
]
_R_DEF = "def ruled_out(attempts, rates):\n"
_R_START = _R_DEF + "    ...\n"
_R = (_R_DEF +
      "    for rate in rates:\n"
      "        if (1 - rate) ** attempts < 0.05:\n"
      "            return rate\n"
      "    return None\n")
_R_ALL = _R.replace("if (1 - rate) ** attempts < 0.05:", "if rate ** attempts < 0.05:")
_R_NOTRULED = _R.replace("if (1 - rate) ** attempts < 0.05:", "if (1 - rate) ** attempts >= 0.05:")
_R_LAST = (_R_DEF +
           "    found = None\n"
           "    for rate in rates:\n"
           "        if (1 - rate) ** attempts < 0.05:\n"
           "            found = rate\n"
           "    return found\n")
_R_ATLEAST = _R.replace("if (1 - rate) ** attempts < 0.05:", "if 1 - (1 - rate) ** attempts < 0.05:")
_R_ELSE = _R.replace("            return rate\n    return None\n", "            return rate\n        return None\n")

_NAMED = [(RATES, "rates")]   # the data the texts call by name


def _lit(value) -> str:
    """A value the way the lesson writes it in code: the list of rates by name."""
    for data, name in _NAMED:
        if type(value) is type(data) and value == data:
            return name
    if isinstance(value, list):
        return "[" + ", ".join(_lit(v) for v in value) + "]"
    return repr(value)


class NamedBox(FunctionBox):
    """A FunctionBox whose messages write a call the way the lesson does: `ruled_out(10, rates)`."""

    def _call_text(self, case) -> str:
        return f"{self.function}({', '.join(_lit(a) for a in case)})"


EXERCISES = {
    "pass": NamedBox(
        name="pass",
        picture=pass_picture,
        function="pass_at_k",
        start=_P_START,
        solution=_P,
        visible=[PASS_CASES[0]],
        hidden=PASS_CASES[1:],
        first=[_DOTS],
        show=show_value,
        compare=same_chance,
        mistakes=[
            Mistake("replace", wrong=_P_REPLACE,
                    en="`{call}` gives {got}, and it should give {want}: in your loop, every pick fails with the same chance, as if a run could be picked again. A run that is picked is out: after a failed pick, there is one failure fewer and one run fewer to pick from.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in je loop faalt elke keuze met dezelfde kans, alsof een run nog eens gekozen kan worden. Een gekozen run ligt eruit: na een mislukte keuze is er één mislukking minder en één run minder om uit te kiezen."),
            Mistake("top", wrong=_P_TOP,
                    en="`{call}` gives {got}, and it should give {want}: with each pick, your failures go down by one, but the runs to pick from do not. After a failed pick, there is also one run fewer.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: bij elke keuze neemt in je code het aantal mislukkingen met één af, maar het aantal runs om uit te kiezen niet. Na een mislukte keuze is er ook één run minder."),
            Mistake("success", wrong=_P_SUCCESS,
                    en="`{call}` gives {got}, and it should give {want}: your `missed` uses the runs that succeeded, `c`. But `missed` is the chance that every picked run failed, so it counts the failures.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `missed` gebruikt de runs die slaagden, `c`. Maar `missed` is de kans dat elke gekozen run mislukte, dus die telt de mislukkingen."),
            Mistake("start1", wrong=_P_START1,
                    en="`{call}` gives {got}, and it should give {want}: your first pick already has one failure fewer to choose from. Let `i` start at 0, so that the first pick sees every failure and every run.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: bij je eerste keuze is er al één mislukking minder om uit te kiezen. Laat `i` bij 0 beginnen, zodat de eerste keuze elke mislukking en elke run ziet."),
            Mistake("plus", wrong=_P_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: one pick too many. pass@k picks `k` runs, so the loop goes round `k` times.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: één keuze te veel. pass@k kiest `k` runs, dus de loop gaat `k` keer rond."),
            Mistake("missed", wrong=_P_MISSED,
                    en="`{call}` gives {got}, and it should give {want}: you give back `missed`, the chance that every picked run failed. pass@k is the chance that at least one of them succeeded.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft `missed` terug, de kans dat elke gekozen run mislukte. pass@k is de kans dat er minstens één slaagde."),
            Mistake("overwrite", wrong=_P_OVER,
                    en="`{call}` gives {got}, and it should give {want}: in each round, your `missed` becomes the chance for this one pick alone, and forgets the picks before it. `missed` is the chance that every pick so far failed, so each round makes the old `missed` smaller.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in elke ronde wordt je `missed` de kans voor alleen deze keuze, en vergeet die de keuzes ervoor. `missed` is de kans dat elke keuze tot nu toe mislukte, dus elke ronde maakt de oude `missed` kleiner."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start `missed` at `1.0`, make it smaller once for each of the `k` picks, and give back the chance that at least one picked run succeeded.",
                      "nl": "Vervang de `...`: begin `missed` bij `1.0`, maak die één keer kleiner voor elk van de `k` keuzes, en geef de kans terug dat minstens één gekozen run slaagde."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. `missed` is the chance that every picked run failed. In each round of the loop, it becomes that chance for one more pick, which fails only if it picks one of the failures that are left. After the loop, pass@k is the chance that at least one picked run succeeded.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. `missed` is de kans dat elke gekozen run mislukte. In elke ronde van de loop wordt `missed` de kans voor één keuze meer; zo'n keuze mislukt alleen als ze een van de overgebleven mislukkingen kiest. Na de loop is pass@k de kans dat minstens één gekozen run slaagde."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `n`, `c` and `k` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie wel de `n`, `c` en `k` uit de aanroep?"},
            "correct": {"en": "Your `pass_at_k` is right in all {n} cases. One success in 25 runs gives a pass@10 of 0.4: 10 runs picked out of 25 hold the one success with a chance of 10 in 25.",
                        "nl": "Je `pass_at_k` klopt in alle {n} gevallen. Eén succes in 25 runs geeft een pass@10 van 0,4: 10 gekozen runs van de 25 bevatten het ene succes met een kans van 10 op 25."},
        },
    ),
    "ruled": NamedBox(
        name="ruled",
        picture=ruled_picture,
        function="ruled_out",
        start=_R_START,
        solution=_R,
        visible=[RULED_CASES[0]],
        hidden=RULED_CASES[1:],
        first=[_DOTS],
        show=show_value,
        compare=same_rate,
        mistakes=[
            Mistake("all", wrong=_R_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your test works out the chance that every attempt succeeds. A rate is ruled out when zero successes would be rare: when every attempt failing has a chance below 0.05.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je test rekent de kans uit dat elke poging slaagt. Een kans is uitgesloten als nul successen zeldzaam zouden zijn: als de kans dat elke poging mislukt onder 0,05 ligt."),
            Mistake("notruled", wrong=_R_NOTRULED,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the first rate that is not ruled out. A rate is ruled out when zero successes would come with a chance below 0.05, not at or above it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de eerste kans terug die niet uitgesloten is. Een kans is uitgesloten als nul successen voorkomen met een kans onder 0,05, niet erop of erboven."),
            Mistake("last", wrong=_R_LAST,
                    en="`{call}` gives {got}, and it should give {want}: your function keeps going through the list and gives back the last rate that is ruled out. The smallest one comes first, so give it back as soon as you find it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie gaat door de hele lijst en geeft de laatste uitgesloten kans terug. De kleinste komt eerst, dus geef die meteen terug."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go through `rates`, give back the first `rate` whose chance of zero successes in `attempts` attempts is below `0.05`, and give back `None` after the loop.",
                      "nl": "Vervang de `...`: ga door `rates`, geef de eerste `rate` terug waarbij de kans op nul successen in `attempts` pogingen onder `0.05` ligt, en geef na de loop `None` terug."},
            "unreached": {"en": "With `{call}`, your function gives back `None`, while a rate in the list is ruled out. Does your test work out the chance that a model with that `rate` shows zero successes in `attempts` attempts, and is it below 0.05? And does `return None` come after the loop, not in it?",
                          "nl": "Met `{call}` geeft je functie `None` terug, terwijl er een kans in de lijst uitgesloten is. Rekent je test de kans uit dat een model met die `rate` nul successen haalt in `attempts` pogingen, en ligt die onder 0,05? En staat `return None` na de loop, en niet erin?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. For each `rate`, from small to large, work out the chance that a model with that rate shows zero successes in `attempts` attempts. The first `rate` where that chance is below 0.05 is the answer; with none, `None`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Reken voor elke `rate`, van klein naar groot, de kans uit dat een model met die kans nul successen haalt in `attempts` pogingen. De eerste `rate` waarbij die kans onder 0,05 ligt, is het antwoord; is er geen, dan `None`."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `attempts` and `rates` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie wel de `attempts` en `rates` uit de aanroep?"},
            "correct": {"en": "Your `ruled_out` is right in all {n} cases. Ten attempts without a success rule out a rate of 0.26, not one of 0.25: such a test cannot show that a capability is absent, only that its rate is probably low.",
                        "nl": "Je `ruled_out` klopt in alle {n} gevallen. Tien pogingen zonder succes sluiten een kans van 0,26 uit, maar niet een van 0,25: zo'n test kan niet laten zien dat een capability ontbreekt, alleen dat de kans erop waarschijnlijk klein is."},
        },
    ),
}

CASES = [
    ("pass", _P_START, "empty", None),
    ("pass", _P, "correct", None),
    ("pass", "from math import comb\n\n" + _P_DEF + "    return 1 - comb(n - c, k) / comb(n, k)\n", "correct", None),
    ("pass", _P_DEF + "    missed, i = 1.0, 0\n    while i < k:\n        missed = missed * (n - c - i) / (n - i)\n"
                      "        i = i + 1\n    return 1 - missed\n", "correct", None),
    ("pass", _P_REPLACE, "wrong", "replace"),
    ("pass", _P_TOP, "wrong", "top"),
    ("pass", _P_SUCCESS, "wrong", "success"),
    ("pass", _P_START1, "wrong", "start1"),
    ("pass", _P_PLUS, "wrong", "plus"),
    ("pass", _P_MISSED, "wrong", "missed"),
    ("pass", _P_OVER, "wrong", "overwrite"),
    ("pass", _P_DEF + "    return 0.4\n", "almost", None),
    ("ruled", _R_START, "empty", None),
    ("ruled", _R, "correct", None),
    ("ruled", _R_DEF + "    for rate in rates:\n        if (1 - rate) ** attempts < 0.05:\n            return rate\n",
     "correct", None),
    ("ruled", _R_DEF + "    out = [r for r in rates if (1 - r) ** attempts < 0.05]\n    return min(out) if out else None\n",
     "correct", None),
    ("ruled", _R_ALL, "wrong", "all"),
    ("ruled", _R_NOTRULED, "wrong", "notruled"),
    ("ruled", _R_LAST, "wrong", "last"),
    ("ruled", _R_ATLEAST, "wrong", "unreached"),   # no rate passes this test with 10 attempts: our own message for None
    ("ruled", _R_ELSE, "wrong", "unreached"),
    ("ruled", _R_DEF + "    return 0.26\n", "almost", None),
]
