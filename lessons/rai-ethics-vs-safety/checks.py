# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-ethics-vs-safety: the arithmetic of Greaves and MacAskill's case for strong longtermism
(GPI Working Paper No. 5-2021, June 2021). Three plans lower the risk of a catastrophe in the next 100 years
(extinction, or for AI one at least as bad), each with that risk, the share of it the plan removes and the plan's
cost; three estimates give the expected number of future beings; bednets, a life saved for about 4000 dollars, are
the benchmark for good done now. The learner writes lives_per_100, the lives 100 dollars of a plan is worth on
average for each estimate, and break_even, the number of future beings from which a plan beats the bednets."""
import contextlib
import copy
import html
import io
import math
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- Greaves and MacAskill's numbers ---------------------------------------------------------------------------------

BEDNETS = 100 / 4000          # lives per 100 dollars: a life saved for about 4000 dollars (section 2, after GiveWell)
ESTIMATES = {"main": 1e24, "low": 1e18, "restricted": 1e14}   # the expected number of future beings (section 3)
# section 4: each plan's risk in the next 100 years (of extinction; for AI, of a catastrophe at least as bad),
# the share of that risk it removes, and its cost
PLANS = {
    "asteroids": {"risk": 1 / 150e6 * 0.01, "cut": 0.05, "cost": 1.2e9},   # 1 in 150 million impacts, 1% of them fatal
    "pandemics": {"risk": 1 / 22000, "cut": 0.01, "cost": 250e9},
    "AI": {"risk": 0.001, "cut": 0.01, "cost": 1e9},
}
SPACEGUARD = 5e-16            # the drop in the chance of extinction per 100 dollars of the Spaceguard Survey, as given
# the six scenarios of their table in section 3 (after Newberry, 2021a): duration in centuries and lives per century,
# as powers of ten
SCENARIOS = [("Earth", 4, 10), ("Earth, digital life", 4, 14), ("Solar System", 8, 19),
             ("Solar System, digital life", 7, 23), ("Milky Way", 11, 25), ("Milky Way, digital life", 11, 34)]
POWERS = list(range(10, 25))  # the demo: from 10^10 to 10^24 future beings
SAFE, CHANCE, PAYOFF = 1000, 1e-12, 1e18   # section 8: 1,000 lives for sure, or one in a trillion of a quintillion


def lives_per_100(plan, estimates):
    """For each estimate of the number of future beings, the lives 100 dollars of the plan is worth on average: what
    the learner writes in Exercise 1."""
    drop = plan["risk"] * plan["cut"]
    return {name: drop * estimates[name] / plan["cost"] * 100 for name in estimates}


def break_even(plan, benchmark):
    """The number of future beings at which 100 dollars of the plan is worth `benchmark` lives: what the learner writes
    in Exercise 2."""
    return benchmark * plan["cost"] / (plan["risk"] * plan["cut"] * 100)


def ahead(beings: float) -> list:
    """The plans that beat the bednets with this many future beings."""
    return [name for name, plan in PLANS.items() if lives_per_100(plan, {"x": beings})["x"] > BEDNETS]


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def power(k: int) -> str:
    return "10" + str(k).translate(_SUP)


def number(x: float, lang: str) -> str:
    """A number for a picture or a sentence: up to three significant digits from 0.001 to 100,000, else a power of ten
    with two; a decimal comma in Dutch."""
    x = float(x)
    if x == 0:
        return "0"
    if 1e-3 <= abs(x) < 1e5:
        r = float(f"{x:.3g}")
        if r >= 1000:
            text = f"{int(r):,}" if lang == "en" else (f"{int(r):,}".replace(",", ".") if r >= 10000 else str(int(r)))
            return text
        text = f"{r:.10f}".rstrip("0").rstrip(".")
    else:
        mantissa, exponent = f"{x:.1e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        text = power(int(exponent)) if mantissa == "1" else f"{mantissa} × {power(int(exponent))}"
    return text.replace(".", ",") if lang == "nl" else text


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:top")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_CODE = "font-family:var(--ea-font-code);font-size:0.78rem;white-space:nowrap"


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is (its cells, a style of the row), and each cell is (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{html.escape(h)}</th>' for h in head)
    body = "".join(f'<tr style="{style}">' + "".join(f'<td style="{_CELL}{";" + s if s else ""}">{cell}</td>'
                                                    for cell, s in cells_) + "</tr>" for cells_, style in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


def _code(text: str) -> str:
    return f'<span style="{_CODE}">{html.escape(text)}</span>'


def _axis(parts: list, x0: float, x1: float, lo: int, hi: int, y: float, every: int, lang: str):
    """A log axis from 10^lo to 10^hi between x0 and x1 at height y, with a label at every `every`-th power."""
    def at(p: float) -> float:
        return x0 + (p - lo) / (hi - lo) * (x1 - x0)
    parts.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" style="stroke:var(--ea-axis)"/>')
    for p in range(lo, hi + 1):   # a tick at every power of ten, a label at every `every`-th
        long = (p - lo) % every == 0
        parts.append(f'<line x1="{at(p):.1f}" y1="{y}" x2="{at(p):.1f}" y2="{y + (5 if long else 3)}" '
                     f'style="stroke:var(--ea-axis)"/>')
        if long:
            parts.append(f'<text x="{at(p):.1f}" y="{y + 16}" style="font-size:11px;fill:var(--ea-ink-2);'
                         f'text-anchor:middle">{power(p)}</text>')
    return at


_SIZE_WORDS = {
    "en": {"title": "How many future lives? Six scenarios and three estimates",
           "names": {"Earth": "Earth", "Earth, digital life": "Earth, digital life", "Solar System": "Solar System",
                     "Solar System, digital life": "Solar System, digital life", "Milky Way": "Milky Way",
                     "Milky Way, digital life": "Milky Way, digital life"},
           "estimates": {"main": "main", "low": "low", "restricted": "restricted"},
           "note": "A dot is the number of lives in one scenario of Greaves and MacAskill's table: how many centuries "
                   "civilization lasts times how many lives a century holds. The dashed lines are their three "
                   "estimates of the expected number of future beings. Each tick to the right is ten times as many."},
    "nl": {"title": "Hoeveel toekomstige levens? Zes scenario's en drie schattingen",
           "names": {"Earth": "Aarde", "Earth, digital life": "Aarde, digitaal leven", "Solar System": "Zonnestelsel",
                     "Solar System, digital life": "Zonnestelsel, digitaal leven", "Milky Way": "Melkweg",
                     "Milky Way, digital life": "Melkweg, digitaal leven"},
           "estimates": {"main": "hoofd", "low": "laag", "restricted": "beperkt"},
           "note": "Een stip is het aantal levens in een scenario uit de tabel van Greaves en MacAskill: hoeveel eeuwen "
                   "de beschaving duurt maal hoeveel levens er in een eeuw passen. De gestippelde lijnen zijn hun drie "
                   "schattingen van het verwachte aantal toekomstige levens. Elk streepje naar rechts is tien keer zoveel."},
}


def size_picture(lang: str) -> str:
    w = _SIZE_WORDS[lang]
    W, left, right, top, row = 600, 170, 585, 34, 24
    lo, hi = 10, 46
    H = top + row * len(SCENARIOS) + 30
    parts = []
    at = _axis(parts, left, right, lo, hi, H - 22, 4, lang)
    for name, k in ESTIMATES.items():
        x = at(math.log10(k))
        parts.append(f'<line x1="{x:.1f}" y1="{top - 14}" x2="{x:.1f}" y2="{H - 22}" '
                     f'style="stroke:var(--ea-ink-3);stroke-dasharray:3 3"/>')
        parts.append(f'<text x="{x:.1f}" y="{top - 18}" style="font-size:11px;fill:var(--ea-ink-2);text-anchor:middle">'
                     f'{html.escape(w["estimates"][name])}</text>')
    for i, (name, centuries, per) in enumerate(SCENARIOS):
        y = top + row * i + row / 2
        x = at(centuries + per)
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" style="font-size:12px;fill:var(--ea-ink);text-anchor:end">'
                     f'{html.escape(w["names"][name])}</text>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{x:.1f}" y2="{y:.1f}" style="stroke:var(--ea-line)"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" style="fill:var(--ea-ink)"/>')
        side = f'x="{x - 9:.1f}" text-anchor="end"' if x > right - 40 else f'x="{x + 9:.1f}"'
        parts.append(f'<text {side} y="{y + 4:.1f}" style="font-size:11px;fill:var(--ea-ink-2)">'
                     f'{power(centuries + per)}</text>')
    svg = (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(w["title"])}" style="width:{W}px;'
           f'max-width:100%;height:auto;display:block;font-family:var(--ea-font-ui)">{"".join(parts)}</svg>')
    return yours.figure(w["title"], svg + _note(w["note"]), alt=w["title"] + ". " + w["note"])


_DEMO_WORDS = {
    "en": {"title": "Lives per 100 dollars with {n} future beings",
           "bednets": "bednets: 0.025",
           "names": {"asteroids": "asteroids", "pandemics": "pandemics", "AI": "AI"},
           "plans": {"asteroids": "the asteroid plan", "pandemics": "the pandemic plan", "AI": "the AI plan"},
           "none": "With {n} future beings, none of the three plans beats the bednets.",
           "some": "With {n} future beings, {names} beat{s} the bednets, with {values} against 0.025 lives per 100 dollars.",
           "and": " and ", "sep": ", ",
           "note": "Each tick to the right is ten times as many lives. A bar in the color of the text beats the "
                   "bednets, a gray one does not."},
    "nl": {"title": "Levens per 100 dollar bij {n} toekomstige levens",
           "bednets": "klamboes: 0,025",
           "names": {"asteroids": "asteroïden", "pandemics": "pandemieën", "AI": "AI"},
           "plans": {"asteroids": "het asteroïdenplan", "pandemics": "het pandemieplan", "AI": "het AI-plan"},
           "none": "Bij {n} toekomstige levens doet geen van de drie plannen het beter dan de klamboes.",
           "some": "Bij {n} toekomstige levens {doet} {names} het beter dan de klamboes, met {values} tegen 0,025 "
                   "levens per 100 dollar.",
           "and": " en ", "sep": "; ",   # the numbers have decimal commas
           "note": "Elk streepje naar rechts is tien keer zoveel levens. Een balk in de kleur van de tekst doet het beter "
                   "dan de klamboes, een grijze niet."},
}


def _joined(items: list, word: str, sep: str = ", ") -> str:
    return items[0] if len(items) == 1 else sep.join(items[:-1]) + word + items[-1]


def demo_frame(k, lang: str) -> tuple[str, str]:
    """The demo: with 10^k future beings, the lives 100 dollars of each plan is worth, against the bednets."""
    k = int(min(POWERS, key=lambda p: abs(p - float(k))))
    w = _DEMO_WORDS[lang]
    beings = 10.0 ** k
    values = {name: lives_per_100(plan, {"x": beings})["x"] for name, plan in PLANS.items()}
    W, left, right, top, row = 600, 100, 585, 26, 30
    lo, hi = -9, 13
    H = top + row * len(PLANS) + 32
    parts = []
    at = _axis(parts, left, right, lo, hi, H - 22, 2, lang)
    for i, (name, v) in enumerate(values.items()):
        y = top + row * i
        x = at(max(lo, min(hi, math.log10(v))))
        fill = "var(--ea-ink)" if v > BEDNETS else "var(--ea-ink-3)"
        parts.append(f'<text x="{left - 8}" y="{y + row / 2 + 4:.1f}" style="font-size:12px;fill:var(--ea-ink);'
                     f'text-anchor:end">{html.escape(w["names"][name])}</text>')
        parts.append(f'<rect x="{left}" y="{y + 6}" width="{max(1.0, x - left):.1f}" height="{row - 12}" '
                     f'style="fill:{fill}"/>')
        if x > right - 70:   # no room right of the bar: the value goes inside it, in the colour of the page
            place = f'x="{x - 6:.1f}" text-anchor="end" style="font-size:11px;fill:var(--ea-page)"'
        else:
            place = (f'x="{x + 6:.1f}" style="font-size:11px;fill:var(--ea-ink-2);paint-order:stroke;'
                     f'stroke:var(--ea-page);stroke-width:4px;stroke-linejoin:round"')
        parts.append(f'<text {place} y="{y + row / 2 + 4:.1f}">{number(v, lang)}</text>')
    x = at(math.log10(BEDNETS))
    parts.append(f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{H - 22}" '
                 f'style="stroke:var(--ea-ink-2);stroke-width:1.5;stroke-dasharray:4 3"/>')
    parts.append(f'<text x="{x + 4:.1f}" y="{top - 10}" style="font-size:11px;fill:var(--ea-ink-2)">'
                 f'{html.escape(w["bednets"])}</text>')
    title = w["title"].format(n=power(k))
    svg = (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}" style="width:{W}px;'
           f'max-width:100%;height:auto;display:block;font-family:var(--ea-font-ui)">{"".join(parts)}</svg>')
    winners = [name for name in PLANS if values[name] > BEDNETS]
    if not winners:
        said = w["none"].format(n=power(k))
    else:
        names = _joined([w["plans"][n] for n in winners], w["and"])
        vals = _joined([number(values[n], lang) for n in winners], w["and"], w["sep"])
        said = w["some"].format(n=power(k), names=names, values=vals, s="s" if len(winners) == 1 else "",
                                doet="doet" if len(winners) == 1 else "doen")
    assert winners == ahead(beings)
    return yours.figure(title, svg + _note(w["note"]), alt=title + ". " + said), said


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


def _number_like(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _close(a, b) -> bool:
    return _number_like(a) and math.isclose(a, b, rel_tol=1e-6, abs_tol=0)


_LIVES_WORDS = {
    "en": {"title": "Lives per 100 dollars, plan by plan and estimate by estimate",
           "head": ["plan", "estimate", "the course's lives_per_100", "yours", "yours beats the bednets (0.025)"],
           "yes": "yes", "no": "no", "stopped": "error", "missing": "missing",
           "names": {"main": "main", "low": "low", "restricted": "restricted"},
           "note": "A dashed row is one where your lives_per_100 gives something other than the course's."},
    "nl": {"title": "Levens per 100 dollar, per plan en per schatting",
           "head": ["plan", "schatting", "lives_per_100 van de course", "de jouwe", "de jouwe beter dan de klamboes (0,025)"],
           "yes": "ja", "no": "nee", "stopped": "fout", "missing": "ontbreekt",
           "names": {"main": "hoofd", "low": "laag", "restricted": "beperkt"},
           "note": "Een gestippelde rij is er een waar jouw lives_per_100 iets anders geeft dan die van de course."},
}


def lives_picture(seen) -> str:
    """For each plan and each estimate, the course's lives per 100 dollars beside the learner's, and whether it beats
    the bednets."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w, lang = _LIVES_WORDS[seen.lang], seen.lang
    got = _all(fn, [(plan, ESTIMATES) for plan in PLANS.values()])
    rows = []
    for i, (name, plan) in enumerate(PLANS.items()):
        want = lives_per_100(plan, ESTIMATES)
        mine = None if got is _STOPPED else got[i]
        for estimate in ESTIMATES:
            beats = "–"
            if mine is None:
                shown, off = w["stopped"], True
            elif not isinstance(mine, dict) or estimate not in mine:
                shown, off = w["missing"], True
            elif _number_like(mine[estimate]):
                shown, off = number(mine[estimate], lang), not _close(mine[estimate], want[estimate])
                beats = w["yes"] if mine[estimate] > BEDNETS else w["no"]
            else:
                text = repr(mine[estimate])
                shown, off = (text if len(text) <= 30 else text[:27] + "..."), True
            rows.append(([(_code(name), ""), (html.escape(w["names"][estimate]), ""),
                          (number(want[estimate], lang), ""), (html.escape(shown), "font-weight:600"),
                          (beats, "")], _OFF if off else ""))
    return yours.figure(w["title"], _table(rows, w["head"]) + _note(w["note"]), alt=w["title"])


_EVEN_WORDS = {
    "en": {"title": "From how many future beings does a plan beat the bednets?",
           "head": ["plan", "the course's break_even", "yours", "estimates above yours"],
           "stopped": "error", "none": "none",
           "names": {"main": "main", "low": "low", "restricted": "restricted"},
           "note": "A dashed row is one where your break_even gives something other than the course's. The estimates "
                   "are 10²⁴ (main), 10¹⁸ (low) and 10¹⁴ (restricted)."},
    "nl": {"title": "Vanaf hoeveel toekomstige levens doet een plan het beter dan de klamboes?",
           "head": ["plan", "break_even van de course", "de jouwe", "schattingen boven de jouwe"],
           "stopped": "fout", "none": "geen",
           "names": {"main": "hoofd", "low": "laag", "restricted": "beperkt"},
           "note": "Een gestippelde rij is er een waar jouw break_even iets anders geeft dan die van de course. De "
                   "schattingen zijn 10²⁴ (hoofd), 10¹⁸ (laag) en 10¹⁴ (beperkt)."},
}


def break_even_picture(seen) -> str:
    """For each plan, the course's break-even number of future beings beside the learner's, and the estimates above
    it."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w, lang = _EVEN_WORDS[seen.lang], seen.lang
    got = _all(fn, [(plan, BEDNETS) for plan in PLANS.values()])
    rows = []
    for i, (name, plan) in enumerate(PLANS.items()):
        want = break_even(plan, BEDNETS)
        above = "–"
        if got is _STOPPED:
            shown, off = w["stopped"], True
        elif _number_like(got[i]):
            shown, off = number(got[i], lang), not _close(got[i], want)
            above = ", ".join(w["names"][e] for e, k in ESTIMATES.items() if k > got[i]) or w["none"]
        else:
            text = repr(got[i])
            shown, off = (text if len(text) <= 30 else text[:27] + "..."), True
        rows.append(([(_code(name), ""), (number(want, lang), ""), (html.escape(shown), "font-weight:600"),
                      (html.escape(above), "")], _OFF if off else ""))
    return yours.figure(w["title"], _table(rows, w["head"]) + _note(w["note"]), alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["lives_per_100", "break_even", "label", "scales"]

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
    return isinstance(e, NameError) and re.search(r"'(PLANS|ESTIMATES|BEDNETS|C)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `PLANS`, `ESTIMATES` or `C`, and the check does not give it that name: use the names between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `PLANS`, `ESTIMATES` of `C`, en de check geeft die naam niet mee: gebruik de namen tussen de haakjes van de `def`-regel.")


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


_AI = Named('PLANS["AI"]', PLANS["AI"])
_EST = Named("ESTIMATES", ESTIMATES)

# exercise 1: lives_per_100; the first case is the AI plan with the three estimates
LIVES_CASES = [(_AI, _EST),
               (Named('PLANS["asteroids"]', PLANS["asteroids"]), _EST),
               (Named('PLANS["pandemics"]', PLANS["pandemics"]), _EST),
               ({"risk": 0.5, "cut": 0.1, "cost": 200}, {"few": 10, "many": 1000}),
               ({"risk": 0.2, "cut": 0.5, "cost": 50}, {"one": 100}),
               ({"risk": 0.3, "cut": 1, "cost": 10}, {})]
_L_START = "def lives_per_100(plan, estimates):\n    ...\n"
_L = ("def lives_per_100(plan, estimates):\n"
      "    drop = plan[\"risk\"] * plan[\"cut\"]\n"
      "    result = {}\n"
      "    for name in estimates:\n"
      "        result[name] = drop * estimates[name] / plan[\"cost\"] * 100\n"
      "    return result\n")
_L_NO_100 = _L.replace(" / plan[\"cost\"] * 100\n", " / plan[\"cost\"]\n")
_L_DIV_100 = _L.replace(" / plan[\"cost\"] * 100\n", " / plan[\"cost\"] / 100\n")
_L_NO_CUT = _L.replace("    drop = plan[\"risk\"] * plan[\"cut\"]\n", "    drop = plan[\"risk\"]\n")
_L_TIMES_COST = _L.replace(" / plan[\"cost\"] * 100\n", " * plan[\"cost\"] * 100\n")

# exercise 2: break_even; the first case is the AI plan against the bednets
EVEN_CASES = [(_AI, Named("BEDNETS", BEDNETS)),
              (Named('PLANS["asteroids"]', PLANS["asteroids"]), Named("BEDNETS", BEDNETS)),
              (Named('PLANS["pandemics"]', PLANS["pandemics"]), Named("BEDNETS", BEDNETS)),
              ({"risk": 0.5, "cut": 0.1, "cost": 200}, 1.0),
              ({"risk": 0.2, "cut": 0.5, "cost": 50}, 4.0)]
_E_START = "def break_even(plan, benchmark):\n    ...\n"
_E = ("def break_even(plan, benchmark):\n"
      "    drop = plan[\"risk\"] * plan[\"cut\"]\n"
      "    return benchmark * plan[\"cost\"] / (drop * 100)\n")
_E_NO_100 = _E.replace("(drop * 100)", "drop")
_E_FLIP = _E.replace("benchmark * plan[\"cost\"] / (drop * 100)", "drop * 100 / (benchmark * plan[\"cost\"])")
_E_TIMES_DROP = _E.replace("benchmark * plan[\"cost\"] / (drop * 100)", "benchmark * plan[\"cost\"] * drop * 100")
_E_NO_PARENS = _E.replace("(drop * 100)", "drop * 100")


EXERCISES = {
    "lives_per_100": FunctionBox(
        name="lives_per_100",
        picture=lives_picture,
        function="lives_per_100",
        start=_L_START,
        solution=_L,
        visible=[LIVES_CASES[0]],
        hidden=LIVES_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("no-100", wrong=_L_NO_100,
                    en="`{call}` gives {got}, and it should give {want}: those are lives per dollar. Multiply by 100 for the lives per 100 dollars.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat zijn levens per dollar. Vermenigvuldig met 100 voor de levens per 100 dollar."),
            Mistake("div-100", wrong=_L_DIV_100,
                    en="`{call}` gives {got}, and it should give {want}: 100 dollars buy 100 times what one dollar buys, so multiply by 100 instead of dividing.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: 100 dollar koopt 100 keer zoveel als één dollar, dus vermenigvuldig met 100 in plaats van te delen."),
            Mistake("no-cut", wrong=_L_NO_CUT,
                    en="`{call}` gives {got}, and it should give {want}: the plan does not take the whole risk away, only the share `plan[\"cut\"]` of it. The drop is the risk times the cut.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het plan haalt niet het hele risico weg, alleen het deel `plan[\"cut\"]` ervan. De daling is het risico maal dat deel."),
            Mistake("times-cost", wrong=_L_TIMES_COST,
                    en="`{call}` gives {got}, and it should give {want}: the cost buys the whole drop, so one dollar buys the drop divided by the cost, not times it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de kosten kopen de hele daling, dus één dollar koopt de daling gedeeld door de kosten, niet maal."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: the risk times the cut is the drop; for every estimate, the drop times the number of future beings, divided by the cost and times 100.",
                      "nl": "Vervang de `...`: het risico maal het deel dat het plan weghaalt is de daling; voor elke schatting de daling maal het aantal toekomstige levens, gedeeld door de kosten en maal 100."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. For each estimate: the risk times the cut, times that number of future beings, divided by the cost, times 100.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Voor elke schatting: het risico maal het deel dat het plan weghaalt, maal dat aantal toekomstige levens, gedeeld door de kosten, maal 100."},
            "almost": {"en": "Right for the AI plan. But `{call}` gives {got}, and it should give {want}. Does your function use the `plan` and `estimates` it gets?",
                       "nl": "Goed voor het AI-plan. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie het `plan` en de `estimates` die ze krijgt?"},
            "correct": {"en": "Your `lives_per_100` is right in all {n} cases. Under the check, the table compares every plan with the bednets.",
                        "nl": "Je `lives_per_100` klopt in alle {n} gevallen. Onder de check zet de tabel elk plan naast de klamboes."},
        },
    ),
    "break_even": FunctionBox(
        name="break_even",
        picture=break_even_picture,
        function="break_even",
        start=_E_START,
        solution=_E,
        visible=[EVEN_CASES[0]],
        hidden=EVEN_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("no-100", wrong=_E_NO_100,
                    en="`{call}` gives {got}, and it should give {want}: the benchmark is in lives per 100 dollars, so the drop per dollar has to be multiplied by 100 as well.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de benchmark is in levens per 100 dollar, dus de daling per dollar moet ook met 100 worden vermenigvuldigd."),
            Mistake("flip", wrong=_E_FLIP,
                    en="`{call}` gives {got}, and it should give {want}: that is upside down. More future beings are needed when the plan costs more and when the benchmark is higher, so those go on top.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is omgekeerd. Er zijn meer toekomstige levens nodig als het plan meer kost en als de benchmark hoger is, dus die staan boven de deelstreep."),
            Mistake("times-drop", wrong=_E_TIMES_DROP,
                    en="`{call}` gives {got}, and it should give {want}: a larger drop means fewer future beings are needed, so divide by the drop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een grotere daling betekent dat er minder toekomstige levens nodig zijn, dus deel door de daling."),
            Mistake("no-parens", wrong=_E_NO_PARENS,
                    en="`{call}` gives {got}, and it should give {want}: without parentheses, Python divides by the drop and then multiplies by 100. Put the product of the drop and 100 between parentheses.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: zonder haakjes deelt Python door de daling en vermenigvuldigt daarna met 100. Zet het product van de daling en 100 tussen haakjes."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: the benchmark times the cost, divided by the drop times 100.",
                      "nl": "Vervang de `...`: de benchmark maal de kosten, gedeeld door de daling maal 100."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The number of future beings at which the lives per 100 dollars equal the benchmark is the benchmark times the cost, divided by the drop times 100.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Het aantal toekomstige levens waarbij de levens per 100 dollar gelijk zijn aan de benchmark, is de benchmark maal de kosten, gedeeld door de daling maal 100."},
            "almost": {"en": "Right for the AI plan. But `{call}` gives {got}, and it should give {want}. Does your function use the `plan` and `benchmark` it gets?",
                       "nl": "Goed voor het AI-plan. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie het `plan` en de `benchmark` die ze krijgt?"},
            "correct": {"en": "Your `break_even` is right in all {n} cases. Under the check, each plan stands beside the estimates above its break-even.",
                        "nl": "Je `break_even` klopt in alle {n} gevallen. Onder de check staat elk plan naast de schattingen boven zijn break-even."},
        },
    ),
}

CASES = [
    ("lives_per_100", _L_START, "empty", None),
    ("lives_per_100", _L, "correct", None),
    ("lives_per_100", "def lives_per_100(plan, estimates):\n    per_dollar = plan[\"risk\"] * plan[\"cut\"] / plan[\"cost\"]\n    return {name: 100 * per_dollar * estimates[name] for name in estimates}\n", "correct", None),
    ("lives_per_100", _L_NO_100, "wrong", "no-100"),
    ("lives_per_100", _L_DIV_100, "wrong", "div-100"),
    ("lives_per_100", _L_NO_CUT, "wrong", "no-cut"),
    ("lives_per_100", _L_TIMES_COST, "wrong", "times-cost"),
    ("lives_per_100", f"def lives_per_100(plan, estimates):\n    return {lives_per_100(PLANS['AI'], ESTIMATES)!r}\n", "almost", None),
    ("lives_per_100", "def lives_per_100(plan, estimates):\n    return {name: 0 for name in ESTIMATES}\n", "error", "lesson-names"),
    ("break_even", _E_START, "empty", None),
    ("break_even", _E, "correct", None),
    ("break_even", "def break_even(plan, benchmark):\n    per_100 = plan[\"risk\"] * plan[\"cut\"] / plan[\"cost\"] * 100\n    return benchmark / per_100\n", "correct", None),
    ("break_even", _E_NO_100, "wrong", "no-100"),
    ("break_even", _E_FLIP, "wrong", "flip"),
    ("break_even", _E_TIMES_DROP, "wrong", "times-drop"),
    ("break_even", _E_NO_PARENS, "wrong", "no-parens"),
    ("break_even", f"def break_even(plan, benchmark):\n    return {break_even(PLANS['AI'], BEDNETS)!r}\n", "almost", None),
    ("break_even", "def break_even(plan, benchmark):\n    return BEDNETS * plan[\"cost\"]\n", "error", "lesson-names"),
]
