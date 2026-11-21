# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-ai-for-defense: AI used for defense, and the offense-defense balance. The learner writes
zero_days, the expected number of zero days an attacker acquires in the idealized model of Garfinkel and Dafoe (2019),
from the number of vulnerabilities and the efforts of attacker and defender, and scaled, the same when better tools
multiply both efforts, which shows offensive-then-defensive scaling. A demo changes the number of vulnerabilities."""
import contextlib
import copy
import html
import io
import re

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, same_value, with_time_limit

# --- the data --------------------------------------------------------------------------------------------------------

N = 10                          # made-up: the vulnerabilities in the software of exercise 1 and the worked example
ATTACK = 4                      # the attacker's effort in exercise 1's first case and its picture
VULNS = [5, 10, 20, 40]         # the stops of the demo's slider: the number of vulnerabilities
FACTORS = list(range(1, 11))    # exercise 2's first case: better tools multiply both efforts by 1 to 10


def zero_days(n, attack, defend):
    """What the learner writes in Exercise 1: the expected number of zero days the attacker acquires, after Garfinkel
    and Dafoe: what the attacker finds, times the share of all vulnerabilities the defender has not found."""
    found = min(n, attack)
    unknown = (n - min(n, defend)) / n
    return found * unknown


def scaled(n, attack, defend, factors):
    """What the learner writes in Exercise 2: the expected zero days when better tools multiply both efforts by each
    factor."""
    result = []
    for factor in factors:
        result.append(zero_days(n, attack * factor, defend * factor))
    return result


def peak(n):
    """Where the attacker's expected zero days are highest when both sides make the same effort: at half of n, with a
    quarter of n zero days; from an effort of n on, there are none."""
    return n / 2, n / 4


def first_late(efforts_until: float = 10, vulns=VULNS):
    """The smallest number of vulnerabilities of the slider at which the attacker still gains from more effort up to an
    effort of `efforts_until`, when both sides make the same effort."""
    return next(n for n in vulns if peak(n)[0] >= efforts_until)


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


def _value(v, digits: int = 4, limit: int = 70) -> str:
    """A result as the check shows it: a number with at most `digits` decimals, a text in quotes, a list or a dict of
    them, else what Python shows, cut short when it is longer than `limit`."""
    if isinstance(v, (list, tuple)):
        text = "[" + ", ".join(_value(x, digits, 10 ** 6) for x in v) + "]"
    elif isinstance(v, dict):
        text = "{" + ", ".join(f"{_value(k, digits)}: {_value(x, digits)}" for k, x in v.items()) + "}"
    elif isinstance(v, str):
        text = '"' + v + '"'
    else:
        f = _number(v)
        if f is None or f != f or f in (float("inf"), float("-inf")):
            text = repr(v) if f is None else str(f)
        else:
            text = f"{f:.{digits}f}".rstrip("0").rstrip(".")
            text = "0" if text == "-0" else text
    return text if len(text) <= limit else text[:limit - 3] + "..."


def show_zero_days(v) -> str:
    return _value(v, 4, 120)


def show_scaled(v) -> str:
    return _value(v, 3, 200)


def _prose(x: float, lang: str, digits: int = 2) -> str:
    """A number in a picture's words: at most `digits` decimals, with a decimal comma in Dutch."""
    text = f"{x:.{digits}f}"
    text = text.rstrip("0").rstrip(".") if digits > 0 else text
    return text.replace(".", ",") if lang == "nl" else text


def _close(got, want) -> bool:
    """Whether a number of the learner's is the course's, as the check compares them."""
    return _number(got) is not None and same_value(got, want)


def _same(got, want) -> bool:
    """Whether the learner's list of zero days is the course's, as the check compares them."""
    return isinstance(got, list) and same_value(got, want)


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


# the demo: the attacker's expected zero days when both sides make the same effort, for a number of vulnerabilities
_DEMO_WORDS = {
    "en": {"title": "Zero days with {n} vulnerabilities, when both sides make the same effort",
           "x": "effort of each side", "y": "expected zero days", "peak": "most at {e}",
           "said": "With {n} vulnerabilities, the attacker's expected zero days rise until an effort of {e}, where "
                   "they reach {top}, and fall to 0 at an effort of {n}, where the defender finds them all.",
           "note": "Both sides make the same effort, in units of one expected discovery, after Garfinkel and Dafoe's "
                   "idealized model.{gray} The numbers are made up.",
           "gray": " The dashed gray line is 10 vulnerabilities, for comparison."},
    "nl": {"title": "Zero days bij {n} kwetsbaarheden, als beide kanten even veel moeite doen",
           "x": "inspanning van elke kant", "y": "verwachte zero days", "peak": "meeste bij {e}",
           "said": "Bij {n} kwetsbaarheden stijgt het verwachte aantal zero days van de aanvaller tot een inspanning "
                   "van {e}, waar het {top} bereikt, en zakt het naar 0 bij een inspanning van {n}, waar de "
                   "verdediger ze allemaal vindt.",
           "note": "Beide kanten doen even veel moeite, in eenheden van één verwachte ontdekking, naar het "
                   "geïdealiseerde model van Garfinkel en Dafoe.{gray} De getallen zijn verzonnen.",
           "gray": " De gestreepte grijze lijn is 10 kwetsbaarheden, ter vergelijking."},
}


def demo_said(n, lang: str) -> str:
    e, top = peak(n)
    return _DEMO_WORDS[lang]["said"].format(n=n, e=_prose(e, lang), top=_prose(top, lang))


def _even_curve(n, top_effort: float = 42) -> tuple[list, list]:
    """The attacker's expected zero days at each effort from 0 to top_effort, in steps of a half, when both sides make
    the same effort."""
    xs = [k / 2 for k in range(int(top_effort * 2) + 1)]
    return xs, [zero_days(n, x, x) for x in xs]


def demo_frame(v, lang: str) -> tuple[str, str]:
    """The demo: the attacker's expected zero days against the effort of each side, at the chosen number of
    vulnerabilities, with its highest point marked."""
    n = min(VULNS, key=lambda k: abs(k - float(v)))
    w = _DEMO_WORDS[lang]
    panel = chart.Panel(x=(0, 42), y=(0, 11.5), xlabel=w["x"], ylabel=w["y"], xticks=[0, 10, 20, 30, 40],
                        yticks=[0, 2, 4, 6, 8, 10])
    if n != N:
        panel.line(*_even_curve(N), role="ink-3", width=1.6, dashed=True)
    panel.line(*_even_curve(n), role="data")
    e, top = peak(n)
    panel.points([e], [top], role="data", r=4)
    words = w["peak"].format(e=_prose(e, lang))
    if n < N:   # the dashed line of 10 rises above this peak: the label goes right of it, level with the dot
        panel.text(N + 0.8, top - 0.25, words, role="ink-2", anchor="start", halo=True)
    else:       # above the peak, inside the panel
        panel.text(e + 0.8, top + 0.6, words, role="ink-2", anchor="start", halo=True)
    title = w["title"].format(n=n)
    said = demo_said(n, lang)
    svg = chart.figure([panel], alt=title + ". " + said, width=380, height=240)
    if lang == "nl":
        svg = chart.with_decimal_comma(svg)
    note = w["note"].format(gray=w["gray"] if n != N else "")   # the dashed line of 10 only when it is another line
    return yours.figure(title, svg + _note(note), alt=title + ". " + said), said


# --- the pictures under the checks -----------------------------------------------------------------------------------

_ZERO_WORDS = {
    "en": {"title": "Your zero days for 10 vulnerabilities and an attacker's effort of 4",
           "x": "defender's effort", "y": "expected zero days",
           "head": ["n", "attack", "defend", "the course's", "yours"], "stopped": "error",
           "note": "The chart keeps the first case's 10 vulnerabilities and attacker's effort of 4, and changes the "
                   "defender's effort from 0 to 12. A dot is the course's number of zero days and a ring is yours. "
                   "The table is every case; a dashed row is one where your zero_days gives something other than the "
                   "course's. The numbers are made up."},
    "nl": {"title": "Jouw zero days bij 10 kwetsbaarheden en een inspanning van de aanvaller van 4",
           "x": "inspanning van de verdediger", "y": "verwachte zero days",
           "head": ["n", "attack", "defend", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De grafiek houdt de 10 kwetsbaarheden en de inspanning van de aanvaller van 4 uit het eerste geval "
                   "vast, en verandert de inspanning van de verdediger van 0 tot 12. Een stip is het aantal zero days "
                   "van de course en een ring het jouwe. De tabel bevat elk geval; een gestippelde rij is er een waar "
                   "jouw zero_days iets anders geeft dan die van de course. De getallen zijn verzonnen."},
}


def zero_days_picture(seen) -> str:
    """The first case's attacker against every defender's effort from 0 to 12, the course's zero days as dots and the
    learner's as rings, and every case in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _ZERO_WORDS[seen.lang]
    defends = list(range(13))
    grid = [(N, ATTACK, d) for d in defends]
    got = _all(fn, ZERO_CASES + grid)
    rows = []
    for i, case in enumerate(ZERO_CASES):
        want = zero_days(*case)
        if got[i] is _STOPPED:
            shown_, off = html.escape(w["stopped"]), True
        else:
            shown_, off = _code(show_zero_days(got[i])), not _close(got[i], want)
        rows.append(([(_code(_value(x)), "text-align:right") for x in case]
                     + [(_code(show_zero_days(want)), "text-align:right"), (shown_, "font-weight:600;text-align:right")],
                     _OFF if off else ""))
    panel = chart.Panel(x=(-0.4, 12.4), y=(0, 4.6), xlabel=w["x"], ylabel=w["y"], xticks=[0, 2, 4, 6, 8, 10, 12],
                        yticks=[0, 1, 2, 3, 4])
    want = [zero_days(N, ATTACK, d) for d in defends]
    panel.line(defends, want, role="ink-3")
    rx, ry = [], []
    for d, m in zip(defends, got[len(ZERO_CASES):]):
        f = None if m is _STOPPED else _number(m)
        if f is not None and -0.2 <= f <= 4.6:
            rx.append(d)
            ry.append(f)
    panel.points(rx, ry, role="ink", r=7, hollow=True)
    panel.points(defends, want, role="ink", r=3)
    svg = chart.figure([panel], alt=w["title"], width=380, height=240)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


_SCALED_WORDS = {
    "en": {"title": "Your zero days as tools get better, for an attacker who makes twice the effort",
           "x": "factor of better tools", "y": "expected zero days",
           "head": ["n", "attack", "defend", "factors", "the course's", "yours"], "stopped": "error",
           "note": "The chart is the first case of the check: 10 vulnerabilities, an attacker's effort of 2 and a "
                   "defender's effort of 1, both multiplied by each factor. A dot is the course's number of zero days "
                   "and a ring is yours. The table is every case; a dashed row is one where your scaled gives "
                   "something other than the course's. The numbers are made up."},
    "nl": {"title": "Jouw zero days als het gereedschap beter wordt, voor een aanvaller die twee keer zoveel moeite doet",
           "x": "factor van beter gereedschap", "y": "verwachte zero days",
           "head": ["n", "attack", "defend", "factors", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De grafiek is het eerste geval van de check: 10 kwetsbaarheden, een inspanning van de aanvaller van "
                   "2 en van de verdediger van 1, allebei vermenigvuldigd met elke factor. Een stip is het aantal zero "
                   "days van de course en een ring het jouwe. De tabel bevat elk geval; een gestippelde rij is er een "
                   "waar jouw scaled iets anders geeft dan die van de course. De getallen zijn verzonnen."},
}


def scaled_picture(seen) -> str:
    """The first case of the check as a line of zero days against the factor, the learner's as rings, and every case
    in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _SCALED_WORDS[seen.lang]
    got = _all(fn, SCALE_CASES)
    rows = []
    for i, case in enumerate(SCALE_CASES):
        want = scaled(*case)
        if got[i] is _STOPPED:
            shown_, off = html.escape(w["stopped"]), True
        else:
            shown_, off = _code(show_scaled(got[i])), not _same(got[i], want)
        rows.append(([(_code(_value(x)), "text-align:right") for x in case[:3]]
                     + [(_code(_value(case[3])), ""), (_code(show_scaled(want)), ""), (shown_, "font-weight:600")],
                     _OFF if off else ""))
    n, attack, defend, factors = SCALE_CASES[0]
    want = scaled(n, attack, defend, factors)
    panel = chart.Panel(x=(0.6, 10.4), y=(0, 5.8), xlabel=w["x"], ylabel=w["y"], xticks=[1, 2, 4, 6, 8, 10],
                        yticks=[0, 1, 2, 3, 4, 5])
    panel.line(factors, want, role="ink-3")
    mine = got[0] if isinstance(got[0], list) else []
    rx, ry = [], []
    for k, factor in enumerate(factors):
        f = _number(mine[k]) if k < len(mine) else None
        if f is not None and -0.2 <= f <= 5.8:
            rx.append(factor)
            ry.append(f)
    panel.points(rx, ry, role="ink", r=7, hollow=True)
    panel.points(factors, want, role="ink", r=3)
    svg = chart.figure([panel], alt=w["title"], width=380, height=240)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["zero_days", "scaled", "percent", "voice"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# exercise 1: zero_days; the first case is the even effort of the worked example's 10 vulnerabilities, then a defender
# who does more, an attacker who does more, an attacker past the number of vulnerabilities, a defender past it, twice
# as many vulnerabilities, a single one, and an attacker who does nothing
ZERO_CASES = [(N, ATTACK, 4), (10, 2, 8), (10, 8, 2), (10, 14, 3), (10, 5, 12), (20, 5, 10), (1, 1, 0), (10, 0, 5)]
_Z_START = "def zero_days(n, attack, defend):\n    ...\n"
_Z = ("def zero_days(n, attack, defend):\n"
      "    found = min(n, attack)\n"
      "    unknown = (n - min(n, defend)) / n\n"
      "    return found * unknown\n")
_Z_CAP = _Z.replace("    unknown = (n - min(n, defend)) / n\n", "    unknown = (n - defend) / n\n")
_Z_ALL = _Z.replace("    found = min(n, attack)\n", "    found = attack\n")
_Z_COUNT = _Z.replace("    unknown = (n - min(n, defend)) / n\n", "    unknown = n - min(n, defend)\n")
_Z_SWAP = (_Z.replace("    found = min(n, attack)\n", "    found = min(n, defend)\n")
           .replace("    unknown = (n - min(n, defend)) / n\n", "    unknown = (n - min(n, attack)) / n\n"))
_Z_KNOWN = _Z.replace("    unknown = (n - min(n, defend)) / n\n", "    unknown = min(n, defend) / n\n")
_Z_MINUS = _Z.replace("    return found * unknown\n", "    return found - min(n, defend)\n")

# exercise 2: scaled; the first case is an attacker who makes twice the defender's effort over ten factors, then even
# efforts, a defender who makes twice the effort with 20 vulnerabilities, no factors, and a single factor
SCALE_CASES = [(N, 2, 1, FACTORS), (10, 1, 1, [0, 5, 10]), (20, 1, 2, [1, 5, 10]), (10, 3, 1, []), (5, 1, 1, [2])]
_S_START = "def scaled(n, attack, defend, factors):\n    ...\n"
_S = ("def scaled(n, attack, defend, factors):\n"
      "    result = []\n"
      "    for factor in factors:\n"
      "        result.append(zero_days(n, attack * factor, defend * factor))\n"
      "    return result\n")
_S_ONE = _S.replace("zero_days(n, attack * factor, defend * factor)", "zero_days(n, attack, defend * factor)")
_S_PLUS = _S.replace("zero_days(n, attack * factor, defend * factor)", "zero_days(n, attack + factor, defend + factor)")
_S_SWAP = _S.replace("zero_days(n, attack * factor, defend * factor)", "zero_days(n, defend * factor, attack * factor)")
_S_FACTOR = _S.replace("zero_days(n, attack * factor, defend * factor)", "zero_days(n, factor, factor)")
_S_EARLY = _S.replace("    return result\n", "        return result\n")


EXERCISES = {
    "zero_days": FunctionBox(
        name="zero_days",
        picture=zero_days_picture,
        function="zero_days",
        start=_Z_START,
        solution=_Z,
        visible=[ZERO_CASES[0]],
        hidden=ZERO_CASES[1:],
        first=[_DOTS],
        show=show_zero_days,
        mistakes=[
            Mistake("cap", wrong=_Z_CAP,
                    en="`{call}` gives {got}, and it should give {want}: when `defend` is more than `n`, your share of unknown vulnerabilities drops below 0. The defender too finds no more vulnerabilities than there are: take the smaller of `n` and `defend` before you subtract.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: als `defend` groter is dan `n`, zakt je aandeel onbekende kwetsbaarheden onder 0. Ook de verdediger vindt niet meer kwetsbaarheden dan er zijn: neem het kleinste van `n` en `defend` voordat je aftrekt."),
            Mistake("all", wrong=_Z_ALL,
                    en="`{call}` gives {got}, and it should give {want}: when `attack` is more than `n`, your attacker finds more vulnerabilities than there are. Set `found` as in the worked example, the smaller of `n` and `attack`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: als `attack` groter is dan `n`, vindt je aanvaller meer kwetsbaarheden dan er zijn. Maak `found` zoals in het uitgewerkte voorbeeld, het kleinste van `n` en `attack`."),
            Mistake("count", wrong=_Z_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: your `unknown` is the number of vulnerabilities the defender has not found, not their share. Divide it by `n`, so that it is a share between 0 and 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `unknown` is het aantal kwetsbaarheden dat de verdediger niet heeft gevonden, niet het aandeel ervan. Deel het door `n`, zodat het een aandeel tussen 0 en 1 is."),
            Mistake("swap", wrong=_Z_SWAP,
                    en="`{call}` gives {got}, and it should give {want}: you swapped the attacker and the defender. `found` is what the attacker finds, from `attack`; `unknown` is the share the defender has not found, from `defend`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je hebt aanvaller en verdediger verwisseld. `found` is wat de aanvaller vindt, uit `attack`; `unknown` is het aandeel dat de verdediger niet heeft gevonden, uit `defend`."),
            Mistake("known", wrong=_Z_KNOWN,
                    en="`{call}` gives {got}, and it should give {want}: your `unknown` is the share the defender has found. A zero day is a vulnerability the defender does not know, so take the share the defender has not found: `n` minus what the defender finds, divided by `n`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `unknown` is het aandeel dat de verdediger heeft gevonden. Een zero day is een kwetsbaarheid die de verdediger niet kent, dus neem het aandeel dat de verdediger niet heeft gevonden: `n` min wat de verdediger vindt, gedeeld door `n`."),
            Mistake("minus", wrong=_Z_MINUS,
                    en="`{call}` gives {got}, and it should give {want}: you subtract what the defender finds from what the attacker finds, as if both find the same vulnerabilities. In Garfinkel and Dafoe's model, they search apart: each vulnerability the attacker finds is unknown to the defender with a chance of `unknown`. Return `found` times `unknown`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je trekt wat de verdediger vindt af van wat de aanvaller vindt, alsof beide dezelfde kwetsbaarheden vinden. In het model van Garfinkel en Dafoe zoeken ze los van elkaar: elke kwetsbaarheid die de aanvaller vindt, is met een kans van `unknown` onbekend voor de verdediger. Geef `found` keer `unknown` terug."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: set `found`, what the attacker finds, and `unknown`, the share of all vulnerabilities the defender has not found, and return `found` times `unknown`.",
                      "nl": "Vervang de `...`: maak `found`, wat de aanvaller vindt, en `unknown`, het aandeel van alle kwetsbaarheden dat de verdediger niet heeft gevonden, en geef `found` keer `unknown` terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Set `found` to the smaller of `n` and `attack`, and `unknown` to `n` minus the smaller of `n` and `defend`, divided by `n`; return `found` times `unknown`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Maak `found` gelijk aan het kleinste van `n` en `attack`, en `unknown` aan `n` min het kleinste van `n` en `defend`, gedeeld door `n`; geef `found` keer `unknown` terug."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `n`, `attack` and `defend` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `n`, `attack` en `defend` die ze krijgt?"},
            "correct": {"en": "Your `zero_days` is right in all {n} cases. Under the check, your rings lie on the course's line.",
                        "nl": "Je `zero_days` klopt in alle {n} gevallen. Onder de check liggen jouw ringen op de lijn van de course."},
        },
    ),
    "scaled": FunctionBox(
        name="scaled",
        picture=scaled_picture,
        function="scaled",
        names={"zero_days": zero_days},
        start=_S_START,
        solution=_S,
        visible=[SCALE_CASES[0]],
        hidden=SCALE_CASES[1:],
        first=[_DOTS],
        show=show_scaled,
        mistakes=[
            Mistake("one", wrong=_S_ONE,
                    en="`{call}` gives {got}, and it should give {want}: only your defender gets better tools. Better tools help both sides: multiply `attack` by `factor` too.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alleen je verdediger krijgt beter gereedschap. Beter gereedschap helpt beide kanten: vermenigvuldig ook `attack` met `factor`."),
            Mistake("plus", wrong=_S_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: you add `factor` to the efforts, so both sides gain the same fixed amount. Better tools multiply what each side achieves: multiply both efforts by `factor`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt `factor` op bij de inspanningen, dus beide kanten krijgen er evenveel bij. Beter gereedschap vermenigvuldigt wat elke kant bereikt: vermenigvuldig beide inspanningen met `factor`."),
            Mistake("swap", wrong=_S_SWAP,
                    en="`{call}` gives {got}, and it should give {want}: you swapped the attacker and the defender in the call. `zero_days` takes the attacker's effort first, then the defender's.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je hebt aanvaller en verdediger in de aanroep verwisseld. `zero_days` krijgt eerst de inspanning van de aanvaller, dan die van de verdediger."),
            Mistake("factor", wrong=_S_FACTOR,
                    en="`{call}` gives {got}, and it should give {want}: you pass `factor` itself as both efforts, so `attack` and `defend` play no part. Each effort is the side's own effort times `factor`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft `factor` zelf als beide inspanningen, dus `attack` en `defend` doen niet mee. Elke inspanning is de eigen inspanning van een kant keer `factor`."),
            Mistake("early", wrong=_S_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your `return` stands inside the loop, so the function stops after the first factor. Put `return result` after the loop, level with `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return` staat in de loop, dus de functie stopt na de eerste factor. Zet `return result` na de loop, op gelijke hoogte met `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: for each `factor` in `factors`, put the zero days of `zero_days` for the efforts times `factor` in a list `result`.",
                      "nl": "Vervang de `...`: zet voor elke `factor` in `factors` de zero days van `zero_days` voor de inspanningen keer `factor` in een lijst `result`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Begin `result` as an empty list. In a `for` loop over `factors`, append `zero_days` with `n`, `attack` times `factor` and `defend` times `factor`; after the loop, return `result`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Begin `result` als een lege lijst. Voeg in een `for`-loop over `factors` `zero_days` toe met `n`, `attack` keer `factor` en `defend` keer `factor`; geef na de loop `result` terug."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `n`, `attack`, `defend` and `factors` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `n`, `attack`, `defend` en `factors` die ze krijgt?"},
            "correct": {"en": "Your `scaled` is right in all {n} cases. Under the check, your rings lie on the course's line.",
                        "nl": "Je `scaled` klopt in alle {n} gevallen. Onder de check liggen jouw ringen op de lijn van de course."},
        },
    ),
}

CASES = [
    ("zero_days", _Z_START, "empty", None),
    ("zero_days", _Z, "correct", None),
    ("zero_days", "def zero_days(n, attack, defend):\n    return min(n, attack) * (1 - min(n, defend) / n)\n", "correct", None),
    ("zero_days", "def zero_days(n, attack, defend):\n    if attack > n:\n        attack = n\n    if defend > n:\n        defend = n\n    return attack * (n - defend) / n\n", "correct", None),
    ("zero_days", _Z_CAP, "wrong", "cap"),
    ("zero_days", _Z_ALL, "wrong", "all"),
    ("zero_days", _Z_COUNT, "wrong", "count"),
    ("zero_days", _Z_SWAP, "wrong", "swap"),
    ("zero_days", _Z_KNOWN, "wrong", "known"),
    ("zero_days", _Z_MINUS, "wrong", "minus"),
    ("zero_days", "def zero_days(n, attack, defend):\n    return 2.4\n", "almost", None),
    ("scaled", _S_START, "empty", None),
    ("scaled", _S, "correct", None),
    ("scaled", "def scaled(n, attack, defend, factors):\n    return [zero_days(n, attack * f, defend * f) for f in factors]\n", "correct", None),
    ("scaled", "def scaled(n, attack, defend, factors):\n    result = []\n    for factor in factors:\n        a = min(n, attack * factor)\n        d = min(n, defend * factor)\n        result.append(a * (n - d) / n)\n    return result\n", "correct", None),
    ("scaled", _S_ONE, "wrong", "one"),
    ("scaled", _S_PLUS, "wrong", "plus"),
    ("scaled", _S_SWAP, "wrong", "swap"),
    ("scaled", _S_FACTOR, "wrong", "factor"),
    ("scaled", _S_EARLY, "wrong", "early"),
    ("scaled", "def scaled(n, attack, defend, factors):\n    return " + repr(scaled(*SCALE_CASES[0])) + "\n", "almost", None),
]
