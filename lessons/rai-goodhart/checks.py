# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-goodhart: the course's contests, in which options with a true value get a score with
noise and the option with the highest score wins, as in the optimizer's curse of Smith and Winkler (2006) and the
regressional Goodhart of Manheim and Garrabrant (2018); average_gap, how far the winner's score lies above its true
value, and skeptical, the shrunk estimate of Smith and Winkler, which the learner writes."""
import contextlib
import copy
import html
import io
import math
import re

import numpy as np

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Timeout, kept, with_time_limit

# --- the contests ----------------------------------------------------------------------------------------------------

OPTIONS = 10        # options in a contest, as in the ten alternatives of Smith and Winkler's example
CONTESTS = 2000     # contests the course draws for every setting
SEED = 0
MEAN, SPREAD, NOISE = 0.0, 1.0, 1.0   # true values around 0 with standard deviation 1; noise with standard deviation 1
NS = [1, 2, 3, 5, 10, 20, 50, 100, 1000]   # the numbers of options in the demo


def contests(options: int = OPTIONS, mean: float = MEAN, spread: float = SPREAD, noise: float = NOISE,
             count: int = CONTESTS, seed: int = SEED):
    """`count` contests of `options` options: the true values (mean plus spread times standard normal noise) and the
    scores (true value plus noise times standard normal noise), one row per contest. Spread 0: all options equal."""
    rng = np.random.default_rng(seed)
    trues = mean + spread * rng.normal(size=(count, options))
    scores = trues + noise * rng.normal(size=(count, options))
    return trues, scores


def winners(trues, scores):
    """The true value and the score of the winner of every contest: the option with the highest score."""
    rows = np.arange(len(scores))
    best = np.argmax(scores, axis=1)
    return trues[rows, best], scores[rows, best]


def average_gap(trues, scores):
    """The course's solution of Exercise 1, without a loop."""
    t, s = winners(trues, scores)
    return float(np.mean(s - t))


def skeptical(scores, mean, spread, noise):
    """The course's solution of Exercise 2: Smith and Winkler's equation (6), the posterior mean."""
    alpha = 1 / (1 + noise**2 / spread**2)
    return alpha * scores + (1 - alpha) * mean


def setting(options: int, spread: float):
    """Averages over the course's contests: the winner's score, its true value and the gap. Kept across languages."""
    def make():
        t, s = winners(*contests(options, spread=spread))
        return {"score": float(np.mean(s)), "true": float(np.mean(t)), "gap": float(np.mean(s - t))}
    return kept(("rai-goodhart", "setting", options, spread), make)


TRUES, SCORES = contests()                           # options that really differ
EQUAL_TRUES, EQUAL_SCORES = contests(spread=0.0)     # options that are all equally good
EXAMPLE = 6                                          # the contest of the worked example: its winner is not the best


def expected_max(n: int) -> float:
    """The expected maximum of n standard normal draws, by numerical integration: Smith and Winkler's expected
    disappointment when n equally good options get unbiased noise with standard deviation 1."""
    x = np.linspace(-12, 12, 120001)
    phi = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))
    return float(np.trapezoid(x * n * phi * cdf ** (n - 1), x))


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


TRUES_N, SCORES_N = Named("TRUES", TRUES), Named("SCORES", SCORES)
EQUAL_TRUES_N, EQUAL_SCORES_N = Named("EQUAL_TRUES", EQUAL_TRUES), Named("EQUAL_SCORES", EQUAL_SCORES)


def number(x: float, lang: str, places: int | None = None) -> str:
    """A number for running text: whole numbers without a decimal point (or `places` decimals), a decimal comma in
    Dutch, a real minus sign; −0.00 is shown as 0.00."""
    x = float(x)
    if places is not None:
        text = f"{x:.{places}f}"
        if float(text) == 0:
            text = text.lstrip("-")
    else:
        text = f"{int(x):d}" if x.is_integer() else f"{x:g}"
    return (text.replace(".", ",") if lang == "nl" else text).replace("-", "−")


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:top")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_TEXT = "font-size:11px;fill:var(--ea-ink-2)"


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is (its cells, a style of the row), and each cell is (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{html.escape(h)}</th>' for h in head)
    body = "".join(f'<tr style="{style}">' + "".join(f'<td style="{_CELL}{";" + s if s else ""}">{cell}</td>'
                                                    for cell, s in cells_) + "</tr>" for cells_, style in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


def _side(blocks: list) -> str:
    return '<div style="display:flex;flex-wrap:wrap;gap:0.5rem 1.6rem;align-items:flex-start">' + "".join(
        f'<div>{b}</div>' for b in blocks) + '</div>'


def _svg(width: int, height: int, parts: list, label: str = "") -> str:
    named = f'role="img" aria-label="{html.escape(label)}"' if label else 'aria-hidden="true"'
    return (f'<svg viewBox="0 0 {width} {height}" {named} style="width:{width}px;max-width:100%;height:auto;'
            f'display:block;font-family:var(--ea-font-ui)">{"".join(parts)}</svg>')


_CONTEST_WORDS = {
    "en": {"title": "Contest {k} of the course: 10 options that really differ, each with its true value and its score",
           "true": "true value", "score": "score", "winner": "winner", "gap": "gap {g}",
           "note": "Sorted by score. The winner has the highest score, {s}, and a true value of {t}: its noise was "
                   "{g} points in its favor. {better}",
           "better": ["None of the other 9 options is truly better than it.",
                      "Of the other 9 options, 1 is truly better than it.",
                      "Of the other 9 options, {r} are truly better than it."]},
    "nl": {"title": "Wedstrijd {k} van de course: 10 opties die echt verschillen, elk met zijn ware waarde en zijn score",
           "true": "ware waarde", "score": "score", "winner": "winnaar", "gap": "verschil {g}",
           "note": "Gesorteerd op score. De winnaar heeft de hoogste score, {s}, en een ware waarde van {t}: zijn ruis "
                   "viel {g} punt in zijn voordeel uit. {better}",
           "better": ["Geen van de andere 9 opties is echt beter dan hij.",
                      "Van de andere 9 opties is er 1 echt beter dan hij.",
                      "Van de andere 9 opties zijn er {r} echt beter dan hij."]},
}


def contest_picture(lang: str, k: int = EXAMPLE) -> str:
    """One contest: per option a row, sorted by score, with a hollow dot at its true value and a filled dot at its
    score; the winner's row washed, with the gap between its two dots in the color of a mistake."""
    w = _CONTEST_WORDS[lang]
    true, score = TRUES[k], SCORES[k]
    order = np.argsort(-score, kind="stable")
    lo, hi = np.floor(min(true.min(), score.min())) - 0.5, np.ceil(max(true.max(), score.max())) + 0.5
    left, width, row, top = 78, 420, 21, 26
    X = lambda v: left + (v - lo) / (hi - lo) * width   # noqa: E731
    H = top + row * len(order) + 30
    parts = []
    for i, o in enumerate(order):
        y = top + i * row + row / 2
        if i == 0:
            parts.append(f'<rect x="{left - 74}" y="{y - row / 2:.1f}" width="{width + 78}" height="{row}" '
                         f'style="fill:var(--ea-data-wash)"/>')
            parts.append(f'<text x="{left - 70}" y="{y + 4:.1f}" style="{_TEXT};font-weight:600;fill:var(--ea-ink)">'
                         f'{html.escape(w["winner"])}</text>')
            a, b = sorted((X(true[o]), X(score[o])))
            parts.append(f'<line x1="{a:.1f}" y1="{y:.1f}" x2="{b:.1f}" y2="{y:.1f}" '
                         f'style="stroke:var(--ea-loss);stroke-width:3"/>')
        else:
            parts.append(f'<line x1="{X(true[o]):.1f}" y1="{y:.1f}" x2="{X(score[o]):.1f}" y2="{y:.1f}" '
                         f'style="stroke:var(--ea-axis);stroke-width:1.5"/>')
        parts.append(f'<circle cx="{X(true[o]):.1f}" cy="{y:.1f}" r="4.5" '
                     f'style="fill:var(--ea-page);stroke:var(--ea-ink);stroke-width:1.5"/>')
        parts.append(f'<circle cx="{X(score[o]):.1f}" cy="{y:.1f}" r="4.5" style="fill:var(--ea-data)"/>')
    base = top + row * len(order) + 6
    parts.append(f'<line x1="{left}" y1="{base}" x2="{left + width}" y2="{base}" style="stroke:var(--ea-axis)"/>')
    for v in range(int(lo + 0.5), int(hi) + 1):
        parts.append(f'<line x1="{X(v):.1f}" y1="{base}" x2="{X(v):.1f}" y2="{base + 4}" style="stroke:var(--ea-axis)"/>')
        parts.append(f'<text x="{X(v):.1f}" y="{base + 16}" style="{_TEXT};text-anchor:middle">{number(v, lang)}</text>')
    # the legend, above the rows
    parts.append(f'<circle cx="{left + 6}" cy="10" r="4.5" style="fill:var(--ea-page);stroke:var(--ea-ink);stroke-width:1.5"/>')
    parts.append(f'<text x="{left + 15}" y="14" style="{_TEXT}">{html.escape(w["true"])}</text>')
    parts.append(f'<circle cx="{left + 116}" cy="10" r="4.5" style="fill:var(--ea-data)"/>')
    parts.append(f'<text x="{left + 125}" y="14" style="{_TEXT}">{html.escape(w["score"])}</text>')
    o = order[0]
    g = score[o] - true[o]
    parts.append(f'<line x1="{left + 186}" y1="10" x2="{left + 204}" y2="10" style="stroke:var(--ea-loss);stroke-width:3"/>')
    parts.append(f'<text x="{left + 210}" y="14" style="font-size:11px;fill:var(--ea-loss-text)">'
                 f'{html.escape(w["gap"].format(g=number(g, lang, 2)))}</text>')
    better = int(np.sum(true > true[o]))
    note = w["note"].format(s=number(score[o], lang, 2), t=number(true[o], lang, 2), g=number(g, lang, 2),
                            better=w["better"][min(better, 2)].format(r=better))
    title = w["title"].format(k=k)
    return yours.figure(title, _svg(left + width + 12, H, parts, title) + _note(note), alt=title + ". " + note)


_DEMO_WORDS = {
    "en": {"equal": "All options equally good", "differ": "Options that really differ",
           "rows": ["winner's score", "winner's true value", "gap"],
           "title": "The winners of {c} contests of {n}, on average",
           "said_one": "With 1 option there is nothing to pick, and its score is right on average: the gap is {ge} "
                       "when all options are equally good, and {gd} when they really differ.",
           "said": "With {n} options, all equally good, the winner's score lies {ge} above its true value on average. "
                   "When the options really differ, the winner is really better, {td}, but its score says {sd}: "
                   "{gd} too high.",
           "alt": "Bars of the winner's score, true value and gap, for options that are all equally good and for "
                  "options that really differ."},
    "nl": {"equal": "Alle opties even goed", "differ": "Opties die echt verschillen",
           "rows": ["score van de winnaar", "ware waarde van de winnaar", "verschil"],
           "title": "De winnaars van {c} wedstrijden van {n}, gemiddeld",
           "said_one": "Met 1 optie valt er niets te kiezen, en de score klopt gemiddeld: het verschil is {ge} als alle "
                       "opties even goed zijn, en {gd} als ze echt verschillen.",
           "said": "Met {n} opties die allemaal even goed zijn, ligt de score van de winnaar gemiddeld {ge} boven zijn "
                   "ware waarde. Als de opties echt verschillen, is de winnaar echt beter, {td}, maar zijn score zegt "
                   "{sd}: {gd} te hoog.",
           "alt": "Balken van de score, de ware waarde en het verschil van de winnaar, voor opties die allemaal even goed "
                  "zijn en voor opties die echt verschillen."},
}

_LOW, _HIGH = -0.5, 5.0   # the scale of the demo's bars


def _bars(values: dict, words: list, heading: str, lang: str) -> str:
    """Three bars: the winner's score (data), its true value (ink) and the gap from one to the other (loss)."""
    left, width, row, top = 150, 230, 24, 24
    X = lambda v: left + (min(max(v, _LOW), _HIGH) - _LOW) / (_HIGH - _LOW) * width   # noqa: E731
    parts = [f'<text x="0" y="14" style="font-size:12.5px;font-weight:600;fill:var(--ea-ink)">{html.escape(heading)}</text>']
    spans = [(0.0, values["score"], "var(--ea-data)"), (0.0, values["true"], "var(--ea-ink)"),
             (values["true"], values["score"], "var(--ea-loss)")]
    shown = [values["score"], values["true"], values["gap"]]
    for i, ((a, b, fill), word, v) in enumerate(zip(spans, words, shown)):
        y = top + i * row
        parts.append(f'<text x="0" y="{y + 13}" style="{_TEXT}">{html.escape(word)}</text>')
        x0, x1 = sorted((X(a), X(b)))
        parts.append(f'<rect x="{x0:.1f}" y="{y + 3}" width="{max(x1 - x0, 1):.1f}" height="13" style="fill:{fill}"/>')
        ink = "var(--ea-loss-text)" if i == 2 else "var(--ea-ink)"
        parts.append(f'<text x="{x1 + 5:.1f}" y="{y + 13}" style="font-size:11px;font-weight:600;fill:{ink}">'
                     f'{number(v, lang, 2)}</text>')
    zero = X(0)
    parts.append(f'<line x1="{zero:.1f}" y1="{top}" x2="{zero:.1f}" y2="{top + 3 * row}" style="stroke:var(--ea-axis)"/>')
    return _svg(left + width + 44, top + 3 * row + 4, parts)


def demo_frame(n, lang: str) -> tuple[str, str]:
    """The demo: for n options, the averages over the course's contests of the winner's score, true value and gap,
    for options that are all equally good and for options that really differ."""
    n = min(NS, key=lambda x: abs(x - float(n)))
    w = _DEMO_WORDS[lang]
    equal, differ = setting(n, 0.0), setting(n, SPREAD)
    shown = f"{n:,}" if lang == "en" else str(n)   # Dutch: a thousands dot only from five digits
    title = w["title"].format(c=_count(lang), n=shown)
    picture = (f'<div role="img" aria-label="{html.escape(w["alt"])}">'
               f'<div style="font-family:var(--ea-font-ui);font-size:0.86rem;font-weight:600;color:var(--ea-ink);'
               f'margin-bottom:0.3rem">{html.escape(title)}</div>'
               + _side([_bars(equal, w["rows"], w["equal"], lang), _bars(differ, w["rows"], w["differ"], lang)])
               + '</div>')
    if n == 1:
        said = w["said_one"].format(ge=number(equal["gap"], lang, 2), gd=number(differ["gap"], lang, 2))
    else:
        said = w["said"].format(n=shown, ge=number(equal["gap"], lang, 2), td=number(differ["true"], lang, 2),
                                sd=number(differ["score"], lang, 2), gd=number(differ["gap"], lang, 2))
    return picture, said


# --- the pictures under the checks -----------------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error


def _call(fn, *args):
    """The learner's function on copies of args within the time limit of a check; _STOPPED when it stops with an
    error, and the Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(lambda: fn(*(copy.deepcopy(a) for a in args)))
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _number_like(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _shown(v, lang: str) -> str:
    if _number_like(v):
        return number(v, lang, 2)
    text = repr(v)
    return text if len(text) <= 40 else text[:37] + "..."


_GAP_ROWS = [(10, SPREAD), (10, 0.0), (100, SPREAD)]   # (options, spread) of the rows under Exercise 1

_GAP_WORDS = {
    "en": {"title": "The average gap of the winner, over {c} contests", "differ": "really differ", "equal": "equally good",
           "head": ["options", "the options", "the course's average_gap", "yours"],
           "note": "A dashed row is one where your average_gap gives something other than the course's.",
           "stopped": "error"},
    "nl": {"title": "Het gemiddelde verschil bij de winnaar, over {c} wedstrijden", "differ": "verschillen echt",
           "equal": "even goed", "head": ["opties", "de opties", "average_gap van de course", "de jouwe"],
           "note": "Een gestippelde rij is er een waar jouw average_gap iets anders geeft dan die van de course.",
           "stopped": "fout"},
}


def _count(lang: str) -> str:
    return f"{CONTESTS:,}" if lang == "en" else str(CONTESTS)   # Dutch: a thousands dot only from five digits


def average_gap_picture(seen) -> str:
    """Three settings, with the learner's average_gap of each beside the course's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _GAP_WORDS[seen.lang]
    rows, said = [], []
    for options, spread in _GAP_ROWS:
        trues, scores = contests(options, spread=spread)
        want = average_gap(trues, scores)
        got = _call(fn, trues, scores)
        shown = w["stopped"] if got is _STOPPED else _shown(got, seen.lang)
        off = got is _STOPPED or not (_number_like(got) and abs(float(got) - want) < 1e-9)
        kind = w["differ"] if spread else w["equal"]
        rows.append(([(str(options), ""), (kind, ""), (number(want, seen.lang, 2), ""),
                      (html.escape(shown), "font-weight:600")], _OFF if off else ""))
        said.append(f"{options}, {kind}: {number(want, seen.lang, 2)}, {shown}.")
    title = w["title"].format(c=_count(seen.lang))
    return yours.figure(title, _table(rows, w["head"]) + _note(w["note"]), alt=title + ". " + " ".join(said))


_SKEPTIC_ROWS = [(MEAN, SPREAD, NOISE), (5.0, 2.0, 1.0), (0.0, 1.0, 3.0)]   # (mean, spread, noise) under Exercise 2

_SKEPTIC_WORDS = {
    "en": {"title": "The winners of {c} contests of 10, on average: score, skeptical estimate and true value",
           "head": ["mean, spread, noise", "score", "the course's skeptical", "yours", "true value"],
           "note": "The skeptical estimate of the winner lands, on average, on its true value; its score does not. A "
                   "dashed row is one where your skeptical gives something other than the course's.",
           "stopped": "error"},
    "nl": {"title": "De winnaars van {c} wedstrijden van 10, gemiddeld: score, sceptische schatting en ware waarde",
           "head": ["mean, spread, noise", "score", "skeptical van de course", "de jouwe", "ware waarde"],
           "note": "De sceptische schatting van de winnaar komt gemiddeld uit op zijn ware waarde; zijn score niet. Een "
                   "gestippelde rij is er een waar jouw skeptical iets anders geeft dan die van de course.",
           "stopped": "fout"},
}


def skeptic_setting(mean: float, spread: float, noise: float):
    """The winners' true values and scores in contests of 10 with this mean, spread and noise. Kept."""
    def make():
        return winners(*contests(OPTIONS, mean, spread, noise))
    return kept(("rai-goodhart", "skeptic", mean, spread, noise), make)


def skeptical_picture(seen) -> str:
    """Three kinds of contest, with the winners' average score, the course's skeptical estimate, the learner's and
    the average true value."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _SKEPTIC_WORDS[seen.lang]
    rows, said = [], []
    for mean, spread, noise in _SKEPTIC_ROWS:
        t, s = skeptic_setting(mean, spread, noise)
        want = float(np.mean(skeptical(s, mean, spread, noise)))
        got = _call(fn, s, mean, spread, noise)
        if got is _STOPPED:
            shown, off = w["stopped"], True
        else:
            try:
                value = float(np.mean(np.asarray(got, dtype=float)))
                shown, off = number(value, seen.lang, 2), not (np.shape(got) == np.shape(s) and abs(value - want) < 1e-9)
            except (TypeError, ValueError):
                shown, off = _shown(got, seen.lang), True
        setting_ = ", ".join(number(x, seen.lang) for x in (mean, spread, noise))
        if seen.lang == "nl":
            setting_ = setting_.replace(", ", "; ")
        cells_ = [(setting_, ""), (number(np.mean(s), seen.lang, 2), ""), (number(want, seen.lang, 2), ""),
                  (html.escape(shown), "font-weight:600"), (number(np.mean(t), seen.lang, 2), "")]
        rows.append((cells_, _OFF if off else ""))
        said.append(f"{setting_}: {number(np.mean(s), seen.lang, 2)}, {number(want, seen.lang, 2)}, {shown}, "
                    f"{number(np.mean(t), seen.lang, 2)}.")
    title = w["title"].format(c=_count(seen.lang))
    return yours.figure(title, _table(rows, w["head"]) + _note(w["note"]), alt=title + ". " + " ".join(said))


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["average_gap", "skeptical", "prompts", "fraud"]

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
    return isinstance(e, NameError) and re.search(r"'(TRUES|SCORES|EQUAL_TRUES|EQUAL_SCORES|C)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `TRUES`, `SCORES` or `C`, and the check does not give it that name: use the names between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `TRUES`, `SCORES` of `C`, en de check geeft die naam niet mee: gebruik de namen tussen de haakjes van de `def`-regel.")


# exercise 1: average_gap; the third case is one contest, the fourth two, the fifth one option, the sixth a tie
GAP_CASES = [(TRUES_N, SCORES_N), (EQUAL_TRUES_N, EQUAL_SCORES_N),
             (np.array([[0.0, 0.0, 0.0]]), np.array([[1.0, 3.0, 2.0]])),
             (np.array([[5.0, 1.0], [0.0, 2.0]]), np.array([[4.0, 2.0], [1.0, 4.0]])),
             (np.array([[2.0], [1.0]]), np.array([[2.5], [0.0]])),
             (np.array([[1.0, 0.0]]), np.array([[3.0, 3.0]]))]
_G_START = "import numpy as np\n\n\ndef average_gap(trues, scores):\n    ...\n"
_G = ("import numpy as np\n\n\n"
      "def average_gap(trues, scores):\n"
      "    gaps = []\n"
      "    for i in range(len(scores)):\n"
      "        best = np.argmax(scores[i])\n"
      "        gaps.append(scores[i][best] - trues[i][best])\n"
      "    return np.mean(gaps)\n")
_G_TRUE_BEST = _G.replace("best = np.argmax(scores[i])", "best = np.argmax(trues[i])")
_G_MAX = _G.replace("gaps.append(scores[i][best] - trues[i][best])", "gaps.append(np.max(scores[i]) - np.max(trues[i]))")
_G_ALL = _G.replace("gaps.append(scores[i][best] - trues[i][best])", "gaps.append(np.mean(scores[i] - trues[i]))")
_G_FLIP = _G.replace("scores[i][best] - trues[i][best]", "trues[i][best] - scores[i][best]")
_G_SUM = _G.replace("return np.mean(gaps)", "return np.sum(gaps)")

# exercise 2: skeptical; the first case is contest 6 of the course, the others change the mean, the spread and the noise
SKEPTIC_CASES = [(Named("SCORES[6]", SCORES[EXAMPLE]), 0.0, 1.0, 1.0),
                 (np.array([2.0, 4.0]), 1.0, 2.0, 1.0),
                 (np.array([10.0, 6.0, 5.0]), 5.0, 1.0, 3.0),
                 (np.array([1.0, -1.0]), 0.0, 3.0, 1.0),
                 (np.array([3.0]), 1.0, 1.0, 1.0)]
_S_START = "def skeptical(scores, mean, spread, noise):\n    ...\n"
_S = ("def skeptical(scores, mean, spread, noise):\n"
      "    alpha = 1 / (1 + noise**2 / spread**2)\n"
      "    return alpha * scores + (1 - alpha) * mean\n")
_S_PLAIN = _S.replace("noise**2 / spread**2", "noise / spread")
_S_SWAP = _S.replace("alpha * scores + (1 - alpha) * mean", "alpha * mean + (1 - alpha) * scores")
_S_NO_MEAN = _S.replace("alpha * scores + (1 - alpha) * mean", "alpha * scores")


EXERCISES = {
    "average_gap": FunctionBox(
        name="average_gap",
        picture=average_gap_picture,
        function="average_gap",
        start=_G_START,
        solution=_G,
        visible=[GAP_CASES[0]],
        hidden=GAP_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("true-best", wrong=_G_TRUE_BEST,
                    en="`{call}` gives {got}, and it should give {want}: the winner is the option with the highest score, not the highest true value. The one who picks sees only the scores: take `np.argmax` of the row of `scores`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de winnaar is de optie met de hoogste score, niet met de hoogste ware waarde. Wie kiest, ziet alleen de scores: neem `np.argmax` van de rij van `scores`."),
            Mistake("max", wrong=_G_MAX,
                    en="`{call}` gives {got}, and it should give {want}: the highest true value of a contest can belong to another option than the highest score. Take both numbers of the winner, at the same index `best`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de hoogste ware waarde van een wedstrijd kan bij een andere optie horen dan de hoogste score. Neem beide getallen van de winnaar, bij dezelfde index `best`."),
            Mistake("all", wrong=_G_ALL,
                    en="`{call}` gives {got}, and it should give {want}: that is the average gap over all options, and over all options the noise cancels out. The gap that grows is that of the winner alone.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is het gemiddelde verschil over alle opties, en over alle opties valt de ruis tegen elkaar weg. Het verschil dat groeit, is dat van de winnaar alleen."),
            Mistake("flip", wrong=_G_FLIP,
                    en="`{call}` gives {got}, and it should give {want}: the gap is how far the winner's score lies above its true value, so the score comes first.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het verschil is hoe ver de score van de winnaar boven zijn ware waarde ligt, dus de score komt eerst."),
            Mistake("sum", wrong=_G_SUM,
                    en="`{call}` gives {got}, and it should give {want}: that is the sum of the gaps. The average is the sum divided by the number of contests: `np.mean`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is de som van de verschillen. Het gemiddelde is de som gedeeld door het aantal wedstrijden: `np.mean`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: for every contest, find the winner by its score, and keep its score minus its true value; return the average.",
                      "nl": "Vervang de `...`: zoek voor elke wedstrijd de winnaar op zijn score, en bewaar zijn score min zijn ware waarde; geef het gemiddelde terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. For each row, the winner is the option with the highest score; its gap is its score minus its true value, and the answer is the average of those gaps.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. In elke rij is de winnaar de optie met de hoogste score; zijn verschil is zijn score min zijn ware waarde, en het antwoord is het gemiddelde van die verschillen."},
            "almost": {"en": "Right for the course's 2,000 contests. But `{call}` gives {got}, and it should give {want}. Does your function use the `trues` and `scores` it gets?",
                       "nl": "Goed voor de 2000 wedstrijden van de course. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `trues` en `scores` die ze krijgt?"},
            "correct": {"en": f"Your `average_gap` is right in all {{n}} cases. When the options really differ, the winner's score lies {number(average_gap(TRUES, SCORES), 'en', 2)} above its true value on average; when they are all equally good, {number(average_gap(EQUAL_TRUES, EQUAL_SCORES), 'en', 2)}.",
                        "nl": f"Je `average_gap` klopt in alle {{n}} gevallen. Als de opties echt verschillen, ligt de score van de winnaar gemiddeld {number(average_gap(TRUES, SCORES), 'nl', 2)} boven zijn ware waarde; als ze allemaal even goed zijn, {number(average_gap(EQUAL_TRUES, EQUAL_SCORES), 'nl', 2)}."},
        },
    ),
    "skeptical": FunctionBox(
        name="skeptical",
        picture=skeptical_picture,
        function="skeptical",
        start=_S_START,
        solution=_S,
        visible=[SKEPTIC_CASES[0]],
        hidden=SKEPTIC_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("plain", wrong=_S_PLAIN,
                    en="`{call}` gives {got}, and it should give {want}: the weight compares variances, the squares of the standard deviations: square both `noise` and `spread`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het gewicht vergelijkt varianties, de kwadraten van de standaardafwijkingen: neem het kwadraat van zowel `noise` als `spread`."),
            Mistake("swap", wrong=_S_SWAP,
                    en="`{call}` gives {got}, and it should give {want}: with more noise, the score should count less. `alpha`, which falls as the noise grows, is the weight of the score, and `1 - alpha` that of the mean.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: met meer ruis moet de score minder tellen. `alpha`, dat kleiner wordt als de ruis groeit, is het gewicht van de score, en `1 - alpha` dat van `mean`."),
            Mistake("no-mean", wrong=_S_NO_MEAN,
                    en="`{call}` gives {got}, and it should give {want}: the estimate shrinks toward the mean, not toward 0. Add `(1 - alpha) * mean`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de schatting schuift op naar `mean`, niet naar 0. Tel `(1 - alpha) * mean` erbij op."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: work out the weight `alpha` from the noise and the spread, and return the weighted average of the scores and the mean.",
                      "nl": "Vervang de `...`: reken het gewicht `alpha` uit met `noise` en `spread`, en geef het gewogen gemiddelde van de scores en `mean` terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. `alpha` is 1 divided by (1 plus the noise variance over the spread variance), and the estimate is `alpha` times the scores plus `1 - alpha` times the mean.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. `alpha` is 1 gedeeld door (1 plus de variantie van de ruis gedeeld door die van de spreiding), en de schatting is `alpha` keer de scores plus `1 - alpha` keer `mean`."},
            "almost": {"en": "Right for contest 6 of the course. But `{call}` gives {got}, and it should give {want}. Does your function use the `scores`, `mean`, `spread` and `noise` it gets?",
                       "nl": "Goed voor wedstrijd 6 van de course. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `scores`, `mean`, `spread` en `noise` die ze krijgt?"},
            "correct": {"en": "Your `skeptical` is right in all {n} cases. With as much noise as spread, it halves the distance of every score to the mean; with three times as much noise, it keeps a tenth.",
                        "nl": "Je `skeptical` klopt in alle {n} gevallen. Met evenveel ruis als spreiding halveert hij de afstand van elke score tot het gemiddelde; met drie keer zoveel ruis houdt hij een tiende over."},
        },
    ),
}

CASES = [
    ("average_gap", _G_START, "empty", None),
    ("average_gap", _G, "correct", None),
    ("average_gap", "import numpy as np\n\n\ndef average_gap(trues, scores):\n    best = np.argmax(scores, axis=1)\n    rows = np.arange(len(scores))\n    return np.mean(scores[rows, best] - trues[rows, best])\n", "correct", None),
    ("average_gap", _G_TRUE_BEST, "wrong", "true-best"),
    ("average_gap", _G_MAX, "wrong", "max"),
    ("average_gap", _G_ALL, "wrong", "all"),
    ("average_gap", _G_FLIP, "wrong", "flip"),
    ("average_gap", _G_SUM, "wrong", "sum"),
    ("average_gap", f"def average_gap(trues, scores):\n    return {average_gap(TRUES, SCORES)!r}\n", "almost", None),
    ("average_gap", "def average_gap(trues, scores):\n    return len(TRUES)\n", "error", "lesson-names"),
    ("skeptical", _S_START, "empty", None),
    ("skeptical", _S, "correct", None),
    ("skeptical", "def skeptical(scores, mean, spread, noise):\n    ratio = (noise / spread) ** 2\n    return mean + (scores - mean) / (1 + ratio)\n", "correct", None),
    ("skeptical", _S_PLAIN, "wrong", "plain"),
    ("skeptical", _S_SWAP, "wrong", "swap"),
    ("skeptical", _S_NO_MEAN, "wrong", "no-mean"),
    ("skeptical", "def skeptical(scores, mean, spread, noise):\n    return 0.5 * scores\n", "almost", None),
    ("skeptical", "def skeptical(scores, mean, spread, noise):\n    return SCORES[6]\n", "error", "lesson-names"),
]
