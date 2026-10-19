# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson res-figures: a chart with every run of two settings as points, side by side, with a title and
a label on the y-axis. And the figures of the lesson: a misleading bar chart of two means and a clear chart of the
same runs, and three ways to draw the 2-layer and the 4-layer GPT."""
import re
from collections import Counter

import numpy as np

from everything_ai.check import Mistake, PlotBox, same_value

# The course's runs of the small GPT of res-hypothesis-control with and without warmup, seeds 0 to 9: the validation
# loss after 6 epochs, the same runs as in res-significance.
WARMUP = [2.3902, 2.3924, 2.4174, 2.3938, 2.4012, 2.4034, 2.3864, 2.3989, 2.4313, 2.4005]
NO_WARMUP = [2.3866, 2.3841, 2.4115, 2.3916, 2.3962, 2.3899, 2.3816, 2.3936, 2.4242, 2.3969]
# The 2-layer and the 4-layer GPT of res-fair-baselines, each at its best learning rate, seeds 0 to 2.
TWO_LAYERS = [1.7859, 1.7773, 1.7738]
FOUR_LAYERS = [1.7745, 1.7776, 1.7776]


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def _bars(groups, labels, y0, y1, width=330, height=250, alt=""):
    """Bars of the means of the groups on a y-axis from y0 to y1: a loss, so purple. The name of each group stands
    just above its bar: a chart clips what lies outside its axes, so names under the axis would not show."""
    from everything_ai import chart
    n = len(groups)
    panel = chart.Panel((-0.6, n - 0.4), (y0, y1), title=labels["title"], xlabel="", ylabel=labels["y"], xticks=[])
    for i, (name, vs) in enumerate(groups):
        m = float(np.mean(vs))
        panel.rect(i - 0.3, y0, i + 0.3, m, "loss")
        panel.text(i, m + (y1 - y0) * 0.08, name, "ink-2")
    return chart.figure([panel], alt=alt, width=width, height=height)


def _points(groups, labels, y0, y1, width=330, height=250, alt=""):
    """Every run of each group as a point, side by side on one y-axis, with the mean as a short line: purple. The
    name of each group stands under its points, inside the axes, so y0 leaves room for it."""
    from everything_ai import chart
    n = len(groups)
    panel = chart.Panel((-0.6, n - 0.4), (y0, y1), title=labels["title"], xlabel="", ylabel=labels["y"], xticks=[])
    for i, (name, vs) in enumerate(groups):
        xs = [i + (k - (len(vs) - 1) / 2) * 0.05 for k in range(len(vs))]
        panel.points(xs, vs, "loss", r=3)
        m = float(np.mean(vs))
        panel.line([i - 0.3, i + 0.3], [m, m], "ink", 2)
        panel.text(i, y0 + (y1 - y0) * 0.04, name, "ink-2")
    return chart.figure([panel], alt=alt, width=width, height=height)


def _wrap(figs, lang) -> str:
    """Figures side by side that wrap below each other in a narrow window."""
    from everything_ai import chart
    if lang == "nl":
        figs = [chart.with_decimal_comma(f) for f in figs]
    return '<div style="display:flex;flex-wrap:wrap;gap:0 1.5rem">' + "".join(f'<div style="flex:1 1 260px;max-width:330px">{f}</div>' for f in figs) + "</div>"


def before_after(labels: dict) -> str:
    """The same runs twice: as two bars of the means on an axis that starts at 2.39, and as every run as a point."""
    groups = [(labels["with"], WARMUP), (labels["without"], NO_WARMUP)]
    bad = _bars(groups, {"title": labels["bad"], "y": labels["y"]}, 2.39, 2.404, alt=labels["alt_bad"])
    good = _points(groups, {"title": labels["good"], "y": labels["y"]}, 2.365, 2.44, alt=labels["alt_good"])
    return _wrap([bad, good], labels.get("lang"))


def three_ways(labels: dict) -> str:
    """Three figures of the 2-layer and the 4-layer GPT at their best learning rate, for the choice question."""
    from everything_ai import chart
    groups = [(labels["two"], TWO_LAYERS), (labels["four"], FOUR_LAYERS)]
    a = _bars(groups, {"title": "A", "y": labels["y"]}, 1.776, 1.780, width=260, height=230, alt=labels["alt_a"])
    pb = []
    for name, vs in groups:   # B: each model in its own panel, each with its own y-axis
        lo, hi = min(vs), max(vs)
        pad = (hi - lo) * 0.3
        p = chart.Panel((-0.6, 0.6), (lo - pad, hi + pad), title="B" if not pb else "", xlabel=name, ylabel=labels["y"] if not pb else "", xticks=[])
        p.points([(k - (len(vs) - 1) / 2) * 0.1 for k in range(len(vs))], vs, "loss", r=3)
        pb.append(p)
    b = chart.figure(pb, alt=labels["alt_b"], width=130, height=230)
    c = _points(groups, {"title": "C", "y": labels["y"]}, 1.765, 1.795, width=260, height=230, alt=labels["alt_c"])
    return _wrap([a, b, c], labels.get("lang"))


# --- exercise 1: seed-points --------------------------------------------------------------------------------------------------

_DATA = ("import matplotlib.pyplot as plt\n\n"
         f"with_warmup = {WARMUP}\n"
         f"without_warmup = {NO_WARMUP}\n")
_START = (_DATA + "# every run of both settings as a point, each setting at its own name\n...\n"
          "plt.title(\"\")\nplt.ylabel(\"\")\nplt.show()\n")
_SOLUTION = (_DATA + "plt.scatter([\"with warmup\"] * len(with_warmup), with_warmup)\n"
             "plt.scatter([\"without warmup\"] * len(without_warmup), without_warmup)\n"
             "plt.title(\"Without warmup, the runs end a little lower\")\nplt.ylabel(\"validation loss after 6 epochs\")\nplt.show()\n")


def _with(draw: str = None, title: str = "\"Without warmup, the runs end a little lower\"", ylabel: str = "\"validation loss after 6 epochs\"") -> str:
    draw = draw if draw is not None else ("plt.scatter([\"with warmup\"] * len(with_warmup), with_warmup)\n"
                                          "plt.scatter([\"without warmup\"] * len(without_warmup), without_warmup)\n")
    return _DATA + draw + f"plt.title({title})\nplt.ylabel({ylabel})\nplt.show()\n"


def _kind(kind: str):
    return lambda got, text: bool(got and got["charts"]) and all(c["kind"] == kind for c in got["charts"])


def _counts(values) -> Counter:
    """Numbers as counts, rounded, so the same run drawn from a list is the same number."""
    return Counter(round(v, 6) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool))


def _holds(counts: Counter, runs) -> bool:
    """Every run is among the counted numbers, as often as it is in `runs`."""
    return not (_counts(runs) - counts)


def _at_names(got) -> dict:
    """The y values of the points at each name along the x-axis, as counts: {"with warmup": Counter(...), ...}."""
    out = {}
    for c in got["charts"] if got else []:
        if c["kind"] == "scatter":
            for x, y in zip(c["x"], c["y"]):
                if isinstance(x, str):
                    out.setdefault(x, Counter()).update(_counts([y]))
    return out


def _homes(got):
    """The name that holds every run with warmup, and the one that holds every run without, or None."""
    at = _at_names(got)
    return (next((n for n, c in at.items() if _holds(c, WARMUP)), None),
            next((n for n, c in at.items() if _holds(c, NO_WARMUP)), None))


def _all_runs_named(got) -> bool:
    return _holds(sum(_at_names(got).values(), Counter()), WARMUP + NO_WARMUP)


def _only_means(got, text) -> bool:
    """Points or a line, but only one or two values in all: the means, not the runs."""
    if not got or not got["charts"]:
        return False
    values = [y for c in got["charts"] if c["kind"] in ("scatter", "line") for y in c["y"]]
    return 0 < len(values) <= 2


_PAIRS = {frozenset(_counts([a, b])) for a, b in zip(WARMUP, NO_WARMUP)}


def _pairs(got, text) -> bool:
    """Only lines, each from the run of a seed with warmup to its run without, as in the first picture of
    res-significance."""
    charts = got["charts"] if got else []
    return (len(charts) >= 2 and all(c["kind"] == "line" and len(c["y"]) == 2 for c in charts)
            and {frozenset(_counts(c["y"])) for c in charts} >= _PAIRS)


def _bars_too(got, text) -> bool:
    """Bars, with every run as a point as well: the bars start at 0, so the points shrink to a dot per setting."""
    charts = got["charts"] if got else []
    points = Counter()
    for c in charts:
        if c["kind"] == "scatter":
            points.update(_counts(c["y"]))
    return any(c["kind"] == "bar" for c in charts) and _holds(points, WARMUP + NO_WARMUP)


def _sideways(got, text) -> bool:
    """Every run along the x-axis, and the names up the y-axis: the chart lies on its side."""
    xs = Counter()
    for c in got["charts"] if got else []:
        if c["kind"] == "scatter":
            xs.update(_counts(c["x"]))
    return _holds(xs, WARMUP + NO_WARMUP)


def _one_name(got, text) -> bool:
    """All twenty runs at numbers along the x-axis, such as the seeds 0 to 9: no name says which setting is which."""
    if not got or not got["charts"]:
        return False
    xs = [x for c in got["charts"] if c["kind"] == "scatter" for x in c["x"]]
    return len(xs) >= 20 and not any(isinstance(x, str) for x in xs)


def _one_place(got, text) -> bool:
    """Every run at a name, but the two settings not each at their own name: all at one, or mixed."""
    with_name, without_name = _homes(got)
    return _all_runs_named(got) and (with_name is None or without_name is None or with_name == without_name)


# A name that says "without", in English or Dutch: the runs without warmup belong there.
_WITHOUT = re.compile(r"without|zonder|w/o|\bno\b|\bgeen\b|\bniet\b|\bnone\b|\boff\b", re.IGNORECASE)


def _swapped(got, text) -> bool:
    """The runs with warmup at a name that says without, and the runs without at one that does not."""
    with_name, without_name = _homes(got)
    return (with_name is not None and without_name is not None and with_name != without_name
            and bool(_WITHOUT.search(with_name)) and not _WITHOUT.search(without_name))


def _same(a, b) -> bool:
    """The check compares the learner's points with the solution's one by one (PlotBox with merge). Any name along
    the x-axis counts, such as "met warmup": which runs stand at which name, the mistakes `one-place` and `swapped`
    look at. Numbers count as everywhere."""
    if a and b and all(isinstance(v, str) for v in list(a) + list(b)):
        return True
    return same_value(a, b)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

EXERCISES = {
    "seed-points": PlotBox(
        name="seed-points",
        start=_START,
        solution=_SOLUTION,
        merge=True,
        first=[_DOTS],
        compare=_same,
        mistakes=[
            Mistake("bars", matches=_kind("bar"),
                    en="Your chart draws bars. A bar is read by its length from 0, so next to a loss of 2.4 a difference of 0.006 disappears, and a bar of a mean hides the ten runs behind it. Draw every run as a point: `plt.scatter([\"with warmup\"] * len(with_warmup), with_warmup)`, and the same for the runs without warmup.",
                    nl="Je grafiek tekent staven. Een staaf lees je af aan zijn lengte vanaf 0, dus naast een loss van 2,4 verdwijnt een verschil van 0,006, en een staaf van een gemiddelde verbergt de tien runs erachter. Teken elke run als een punt: `plt.scatter([\"with warmup\"] * len(with_warmup), with_warmup)`, en hetzelfde voor de runs zonder warmup."),
            Mistake("only-means", matches=_only_means,
                    en="Your chart has only one or two points: the means, not the runs. Draw all ten runs of each setting, so a reader sees how far they spread.",
                    nl="Je grafiek heeft maar één of twee punten: de gemiddelden, niet de runs. Teken alle tien runs van elke instelling, zodat een lezer ziet hoe ver ze spreiden."),
            Mistake("pairs", matches=_pairs, state="almost",
                    en="Your lines join the two runs of each seed, as the first picture of Is the difference real? did: a good figure when the runs come in pairs. This exercise asks for the other one, every run as a loose point: draw each setting with one `plt.scatter`.",
                    nl="Je lijnen verbinden de twee runs van elke seed, zoals het eerste plaatje van Is het verschil echt? deed: een goede figuur als de runs in paren komen. Deze oefening vraagt om de andere, elke run als een los punt: teken elke instelling met één `plt.scatter`."),
            Mistake("line", matches=_kind("line"),
                    en="Your chart draws lines, and a line joins runs that have nothing to do with each other: seed 3 does not lead to seed 4. Draw every run as a loose point with `plt.scatter`.",
                    nl="Je grafiek tekent lijnen, en een lijn verbindt runs die niets met elkaar te maken hebben: seed 3 leidt niet naar seed 4. Teken elke run als een los punt met `plt.scatter`."),
            Mistake("bars-too", matches=_bars_too, state="almost",
                    en="The points are right, but the bars under them start at 0, so the y-axis runs from 0 and the twenty runs shrink to one dot per setting: the spread and the difference are gone. Leave out `plt.bar`; a mean drawn with `plt.scatter` may stay.",
                    nl="De punten kloppen, maar de staven eronder beginnen bij 0, dus loopt de y-as vanaf 0 en krimpen de twintig runs tot één stip per instelling: de spreiding en het verschil zijn weg. Laat `plt.bar` weg; een gemiddelde dat je met `plt.scatter` tekent, mag blijven staan."),
            Mistake("sideways", matches=_sideways,
                    en="Your chart lies on its side: the runs stand along the x-axis and the names up the y-axis. That can show the same, but this exercise asks for the names along the x-axis and the loss up the y-axis: swap the two lists in each `plt.scatter`.",
                    nl="Je grafiek ligt op zijn kant: de runs staan langs de x-as en de namen langs de y-as. Dat kan hetzelfde laten zien, maar deze oefening vraagt om de namen langs de x-as en de loss langs de y-as: draai de twee lijsten in elke `plt.scatter` om."),
            Mistake("numbers", matches=_one_name,
                    en="Your points stand at numbers along the x-axis, such as the seeds 0 to 9, and a number does not say which setting a point belongs to. Give each setting its own name along the x-axis: `[\"with warmup\"] * len(with_warmup)` as the x values.",
                    nl="Je punten staan bij getallen langs de x-as, zoals de seeds 0 tot en met 9, en een getal zegt niet bij welke instelling een punt hoort. Geef elke instelling een eigen naam langs de x-as: `[\"with warmup\"] * len(with_warmup)` als x-waarden."),
            Mistake("one-place", matches=_one_place,
                    en="All twenty runs are there, but the two settings do not each stand at their own name, so a reader cannot tell the runs with warmup from those without. Put all of `with_warmup` at one name along the x-axis and all of `without_warmup` at another.",
                    nl="Alle twintig runs staan erin, maar de twee instellingen staan niet elk bij een eigen naam, dus kan een lezer de runs met warmup niet van die zonder onderscheiden. Zet heel `with_warmup` bij één naam langs de x-as en heel `without_warmup` bij een andere."),
            Mistake("swapped", matches=_swapped,
                    en="Your names are swapped: the runs of `with_warmup` stand at a name that says without, and those of `without_warmup` at the other. Put `with_warmup` at `\"with warmup\"` and `without_warmup` at `\"without warmup\"`.",
                    nl="Je namen zijn omgewisseld: de runs van `with_warmup` staan bij een naam die zonder zegt, en die van `without_warmup` bij de andere. Zet `with_warmup` bij `\"with warmup\"` en `without_warmup` bij `\"without warmup\"`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with two lines: one `plt.scatter` with the runs with warmup, one with the runs without, each at its own name along the x-axis. Then fill in the title and the label of the y-axis.",
                      "nl": "Vervang de `...` door twee regels: één `plt.scatter` met de runs met warmup, één met de runs zonder, elk bij een eigen naam langs de x-as. Vul daarna de titel en het label van de y-as in."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "nothing": {"en": "Your code draws nothing yet. Draw the runs with `plt.scatter`, before `plt.show()`.",
                        "nl": "Je code tekent nog niets. Teken de runs met `plt.scatter`, vóór `plt.show()`."},
            "kind": {"en": "Your code draws {kind}, but this exercise asks for {want_kind}: every run as a loose point. Draw it with `{draw}`.",
                     "nl": "Je code tekent {kind}, maar deze oefening vraagt om {want_kind}: elke run als een los punt. Teken die met `{draw}`."},
            "values": {"en": "Not every run is a point at its setting yet. Draw all ten runs of `with_warmup` at one name along the x-axis and all ten of `without_warmup` at another, such as `\"with warmup\"` and `\"without warmup\"`.",
                       "nl": "Nog niet elke run is een punt bij zijn instelling. Teken alle tien runs van `with_warmup` bij één naam langs de x-as en alle tien van `without_warmup` bij een andere, zoals `\"with warmup\"` en `\"without warmup\"`."},
            "title": {"en": "The points are right. Now give the chart a title that says its one message, such as what happens without warmup.",
                      "nl": "De punten kloppen. Geef de grafiek nu een titel die zijn ene boodschap zegt, zoals wat er zonder warmup gebeurt."},
            "ylabel": {"en": "The points are right. Now say along the y-axis what it shows: `plt.ylabel(\"validation loss after 6 epochs\")`.",
                       "nl": "De punten kloppen. Zeg nu langs de y-as wat die laat zien: `plt.ylabel(\"validation loss after 6 epochs\")`."},
            "xlabel": {"en": "", "nl": ""},
            "correct": {"en": "Your chart shows every run, the two settings side by side on one axis, and says what it shows. A reader sees at once that the two groups overlap, and that the runs without warmup lie a little lower.",
                        "nl": "Je grafiek laat elke run zien, de twee instellingen naast elkaar op één as, en zegt wat hij laat zien. Een lezer ziet meteen dat de twee groepen overlappen, en dat de runs zonder warmup iets lager liggen."},
        },
    ),
}

# The lesson is done when these are correct. "fair-figure" is the choice question, "caption" the open question.
REQUIRED = ["seed-points", "fair-figure", "caption"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("seed-points", _START, "empty", None),
    ("seed-points", _SOLUTION, "correct", None),
    ("seed-points", _with(draw="plt.scatter([\"without warmup\"] * 10, without_warmup)\nplt.scatter([\"with warmup\"] * 10, with_warmup)\n"), "correct", None),
    ("seed-points", _with(draw="plt.scatter([\"with warmup\"] * 10 + [\"without warmup\"] * 10, with_warmup + without_warmup)\n"), "correct", None),
    ("seed-points", _with(draw="plt.scatter([\"with warmup\"] * 10, with_warmup)\nplt.scatter([\"without warmup\"] * 10, without_warmup)\n"
                          "plt.scatter([\"with warmup\", \"without warmup\"], [sum(with_warmup) / 10, sum(without_warmup) / 10], marker=\"_\", s=400)\n"), "correct", None),
    ("seed-points", _with(draw="plt.bar([\"with warmup\", \"without warmup\"], [sum(with_warmup) / 10, sum(without_warmup) / 10])\n"), "wrong", "bars"),
    ("seed-points", _with(draw="plt.plot(with_warmup)\nplt.plot(without_warmup)\n"), "wrong", "line"),
    ("seed-points", _with(draw="plt.scatter([\"with warmup\", \"without warmup\"], [sum(with_warmup) / 10, sum(without_warmup) / 10])\n"), "wrong", "only-means"),
    ("seed-points", _with(draw="plt.scatter(range(10), with_warmup)\nplt.scatter(range(10), without_warmup)\n"), "wrong", "numbers"),
    ("seed-points", _with(draw="plt.scatter([0] * 10, with_warmup)\nplt.scatter([1] * 10, without_warmup)\n"), "wrong", "numbers"),
    # other names count, such as Dutch ones, and plt.plot with "o" draws points too
    ("seed-points", _with(draw="plt.scatter([\"met warmup\"] * 10, with_warmup)\nplt.scatter([\"zonder warmup\"] * 10, without_warmup)\n"), "correct", None),
    ("seed-points", _with(draw="plt.plot([\"with warmup\"] * 10, with_warmup, \"o\")\nplt.plot([\"without warmup\"] * 10, without_warmup, \"o\")\n"), "correct", None),
    ("seed-points", _with(draw="plt.scatter([\"without warmup\"] * 10, with_warmup)\nplt.scatter([\"with warmup\"] * 10, without_warmup)\n"), "wrong", "swapped"),
    ("seed-points", _with(draw="plt.scatter([\"runs\"] * 20, with_warmup + without_warmup)\n"), "wrong", "one-place"),
    ("seed-points", _with(draw="plt.scatter(with_warmup, [\"with warmup\"] * 10)\nplt.scatter(without_warmup, [\"without warmup\"] * 10)\n"), "wrong", "sideways"),
    ("seed-points", _with(draw="plt.bar([\"with warmup\", \"without warmup\"], [sum(with_warmup) / 10, sum(without_warmup) / 10])\n"
                          "plt.scatter([\"with warmup\"] * 10, with_warmup)\nplt.scatter([\"without warmup\"] * 10, without_warmup)\n"), "almost", "bars-too"),
    ("seed-points", _with(draw="for i in range(10):\n    plt.plot([\"with warmup\", \"without warmup\"], [with_warmup[i], without_warmup[i]], \"o-\")\n"), "almost", "pairs"),
    ("seed-points", _with(draw="plt.plot([\"with warmup\", \"without warmup\"], [sum(with_warmup) / 10, sum(without_warmup) / 10])\n"), "wrong", "only-means"),
    ("seed-points", _with(title="\"\""), "almost", "no-title"),
    ("seed-points", _with(ylabel="\"\""), "almost", "no-ylabel"),
    ("seed-points", _with(draw="plt.scatter([\"with warmup\"] * 10, with_warmup)\n"), "wrong", "values"),
]
