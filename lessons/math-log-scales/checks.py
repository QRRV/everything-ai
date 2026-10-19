# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson math-log-scales: how often words appear in tiny Shakespeare, by their rank, drawn on a log-log
plot, and the exponent of that power law from two points. And the picture of y = x^k on normal and on log-log axes.
The counts are those of data/tiny-shakespeare.txt, in small letters, with words as runs of letters and apostrophes."""
import math
import re

from everything_ai.check import CodeBox, Mistake, PlotBox

RANKS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
COUNTS = [6283, 5690, 3759, 2375, 1507, 592, 306, 139, 50, 21, 9, 3]


def picture(k: float, labels: dict[str, str]) -> str:
    """y = x^k for x from 1 to 100, on normal axes and on log-log axes, with points at x = 1, 10 and 100. Drawn as
    SVG text, for every stop of the slider once (L.explore)."""
    from everything_ai import chart
    xs = [10 ** (2 * i / 100) for i in range(101)]
    ys = [x ** k for x in xs]
    marks = [1, 10, 100]
    top = max(ys) * 1.08
    normal = chart.Panel((0, 100), (0, top), title=labels["normal"], xlabel="x", ylabel="y")
    normal.line(xs, ys, "data").points(marks, [x ** k for x in marks], "data", 5)
    loglog = chart.Panel((1, 100), (0.01, 10000), title=labels["loglog"], xlabel="x", ylabel="y", xlog=True, ylog=True)
    loglog.line(xs, ys, "data").points(marks, [x ** k for x in marks], "data", 5)
    return chart.figure([normal, loglog], alt=labels["alt"], width=330, height=300)


# The lesson is done when these are correct. "halfway" is the choice question in the lesson.
REQUIRED = ["halfway", "zipf-plot", "zipf-exponent"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_NO_PLT = Mistake("no-plt", syntax=r"(?s)\A(?!.*import\s+matplotlib\.pyplot\s+as\s+plt).*\bplt\s*\.",
                  en="Python does not know `plt` yet: the box needs the line `import matplotlib.pyplot as plt` at the top. Put it back.",
                  nl="Python kent `plt` nog niet: het vak heeft de regel `import matplotlib.pyplot as plt` bovenaan nodig. Zet hem terug.")

_PLOT_DATA = f"import matplotlib.pyplot as plt\n\nranks = {RANKS}\ncounts = {COUNTS}\n"
_PLOT_START = _PLOT_DATA + "...\nplt.show()"
_PLOT = _PLOT_DATA + "plt.loglog(ranks, counts)\nplt.show()"


def _scales(x: str, y: str):
    return lambda got, text: bool(got) and bool(got["charts"]) and (got["xscale"], got["yscale"]) == (x, y)


_K_HEAD = "import numpy as np\n\nrank_a = 10\ncount_a = 2375\nrank_b = 1000\ncount_b = 21\n"
_K_START = _K_HEAD + "k = ...\nk\n"
_K = _K_HEAD + "k = (np.log10(count_b) - np.log10(count_a)) / (np.log10(rank_b) - np.log10(rank_a))\nk"


def _k(new: str) -> str:
    return _K.replace("k = (np.log10(count_b) - np.log10(count_a)) / (np.log10(rank_b) - np.log10(rank_a))", new)


def _any_base(wrong):
    """A known mistake with logs in base 10, e or 2: the right answer may take any base, so its mistakes may too.
    `wrong(log)` works the mistake out on the two points of the box (word 10 and word 1000) with that log."""
    def matches(got, text):
        try:
            got = float(got)
        except (TypeError, ValueError):
            return False
        return any(math.isclose(got, wrong(log), rel_tol=1e-6) for log in (math.log10, math.log, math.log2))
    return matches


_PLOT_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "empty": {"en": "Replace the `...` with a line that draws `counts` against `ranks`, with both axes on a log scale.",
              "nl": "Vervang de `...` door een regel die `counts` tegen `ranks` tekent, met beide assen op een logaritmische schaal."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
    "nothing": {"en": "Your code draws nothing yet. Draw the line with `plt.loglog`, or with `plt.plot` and a log scale on both axes, before `plt.show()`.",
                "nl": "Je code tekent nog niets. Teken de lijn met `plt.loglog`, of met `plt.plot` en een logaritmische schaal op beide assen, vóór `plt.show()`."},
    "kind": {"en": "Your code draws {kind}, but this exercise asks for {want_kind}: points joined by a line. Draw it with `plt.loglog`, or with `{draw}` and two log scales.",
             "nl": "Je code tekent {kind}, maar deze oefening vraagt om {want_kind}: punten verbonden door een lijn. Teken die met `plt.loglog`, of met `{draw}` en twee logaritmische schalen."},
    "values": {"en": "Your line goes through the x values `{got_x}` and the y values `{got}`, and it should go through `{want_x}` and `{want}`. Give the ranks first, along the bottom, then the counts.",
               "nl": "Je lijn gaat door de x-waarden `{got_x}` en de y-waarden `{got}`, en dat moet `{want_x}` en `{want}` zijn. Geef eerst de rangnummers, langs de onderkant, dan de aantallen."},
    "title": {"en": "", "nl": ""}, "xlabel": {"en": "", "nl": ""}, "ylabel": {"en": "", "nl": ""},
    "correct": {"en": "Your log-log plot is right: from word 10 on, the counts fall in an almost straight line, to 3 times for word 5,000; only the few most common words lie flatter. A straight line on a log-log plot is a power law.",
                "nl": "Je log-log plot klopt: vanaf woord 10 dalen de aantallen in een bijna rechte lijn, tot 3 keer voor woord 5000; alleen de paar vaakste woorden liggen vlakker. Een rechte lijn op een log-log plot is een machtswet."},
}

EXERCISES = {
    "zipf-plot": PlotBox(
        name="zipf-plot",
        start=_PLOT_START,
        solution=_PLOT,
        first=[_DOTS],
        mistakes=[
            _NO_PLT,
            Mistake("normal-axes", matches=_scales("linear", "linear"),
                    en="Your chart has normal axes: the line drops straight down at the left and then lies flat along the bottom, so almost every point squeezes into the corner at the bottom left. Put both axes on a log scale: `plt.loglog(ranks, counts)`, or `plt.xscale(\"log\")` and `plt.yscale(\"log\")`.",
                    nl="Je grafiek heeft gewone assen: de lijn valt links recht naar beneden en ligt dan plat langs de onderkant, dus bijna alle punten zitten opgepropt in de hoek linksonder. Zet beide assen op een logaritmische schaal: `plt.loglog(ranks, counts)`, of `plt.xscale(\"log\")` en `plt.yscale(\"log\")`."),
            Mistake("y-only", matches=_scales("linear", "log"),
                    en="Only the y-axis has a log scale, so the line still curves. A power law needs both axes on a log scale: add `plt.xscale(\"log\")`, or draw with `plt.loglog`.",
                    nl="Alleen de y-as heeft een logaritmische schaal, dus de lijn is nog krom. Een machtswet heeft beide assen op een logaritmische schaal nodig: voeg `plt.xscale(\"log\")` toe, of teken met `plt.loglog`."),
            Mistake("x-only", matches=_scales("log", "linear"),
                    en="Only the x-axis has a log scale, so the line still curves. A power law needs both axes on a log scale: add `plt.yscale(\"log\")`, or draw with `plt.loglog`.",
                    nl="Alleen de x-as heeft een logaritmische schaal, dus de lijn is nog krom. Een machtswet heeft beide assen op een logaritmische schaal nodig: voeg `plt.yscale(\"log\")` toe, of teken met `plt.loglog`."),
        ],
        texts=_PLOT_TEXTS,
    ),
    "zipf-exponent": CodeBox(
        name="zipf-exponent",
        start=_K_START,
        solution=_K,
        keep=["rank_a = 10", "count_a = 2375", "rank_b = 1000"],
        # The check also runs the box with another count: a number typed in by hand does not change with it.
        others=[{"count_b": 30}],
        first=[_DOTS],
        mistakes=[
            Mistake("no-logs", wrong=_k("k = (count_b - count_a) / (rank_b - rank_a)"),
                    en="Your last line gives `{got}`: the slope on normal axes, in counts per rank. The exponent is the slope on the log-log plot: take the log of each number first.",
                    nl="Je laatste regel geeft `{got}`: de helling op gewone assen, in aantallen per rangnummer. De exponent is de helling op de log-log plot: neem eerst van elk getal de log."),
            Mistake("swapped", wrong=_k("k = (np.log10(rank_b) - np.log10(rank_a)) / (np.log10(count_b) - np.log10(count_a))"),
                    en="Your last line gives `{got}`: the change in the ranks divided by the change in the counts, the wrong way round. The slope is the change up, the counts, divided by the change along, the ranks.",
                    nl="Je laatste regel geeft `{got}`: de verandering van de rangnummers gedeeld door die van de aantallen, andersom. De helling is de verandering omhoog, de aantallen, gedeeld door die opzij, de rangnummers."),
            Mistake("sign", wrong=_k("k = (np.log10(count_a) - np.log10(count_b)) / (np.log10(rank_b) - np.log10(rank_a))"),
                    en="Your last line gives `{got}`: a slope up, but the line goes down, because the counts get smaller as the rank grows. Subtract in the same order up and along: b minus a, for the counts and for the ranks.",
                    nl="Je laatste regel geeft `{got}`: een helling omhoog, maar de lijn daalt, want de aantallen worden kleiner als het rangnummer groeit. Trek omhoog en opzij in dezelfde volgorde af: b min a, bij de aantallen en bij de rangnummers."),
            Mistake("y-log-only", wrong=_k("k = (np.log10(count_b) - np.log10(count_a)) / (rank_b - rank_a)"),
                    matches=_any_base(lambda log: (log(21) - log(2375)) / (1000 - 10)),
                    en="Your last line gives `{got}`: only the counts have a log. On a log-log plot both axes do: take the log of the ranks too.",
                    nl="Je laatste regel geeft `{got}`: alleen de aantallen hebben een log. Op een log-log plot hebben beide assen dat: neem ook de log van de rangnummers."),
            Mistake("no-divide", wrong=_k("k = np.log10(count_b) - np.log10(count_a)"),
                    matches=_any_base(lambda log: log(21) - log(2375)),
                    en="Your last line gives `{got}`: how far the line goes down. A slope is that divided by how far it goes along: the change in the log of the ranks.",
                    nl="Je laatste regel geeft `{got}`: hoe ver de lijn daalt. Een helling is dat gedeeld door hoe ver hij opzij gaat: de verandering van de log van de rangnummers."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the slope on the log-log plot: the change in the log of the counts, divided by the change in the log of the ranks.",
                      "nl": "Vervang de `...` door de helling op de log-log plot: de verandering van de log van de aantallen, gedeeld door de verandering van de log van de rangnummers."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another count by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een ander aantal."},
            "no_value": {"en": "The last line of the box shows nothing. Put `k` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `k` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you divide the change in the log of the counts by the change in the log of the ranks?",
                      "nl": "Je laatste regel geeft `{got}`. Deel je de verandering van de log van de aantallen door de verandering van de log van de rangnummers?"},
            "other": {"en": "Right for these counts. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out from the four names themselves?",
                      "nl": "Goed voor deze aantallen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met de vier namen zelf?"},
            "other_error": {"en": "Right for these counts. But with {where}, your code stops with an error. Do you work it out from the four names themselves?",
                            "nl": "Goed voor deze aantallen. Maar met {where} stopt je code met een fout. Reken je het uit met de vier namen zelf?"},
            "correct": {"en": "The exponent is about −1.03: about −1. A word twice as far down the list appears about half as often. This is Zipf's law, and it holds roughly for almost every language. The check also tried another count.",
                        "nl": "De exponent is ongeveer −1,03: ongeveer −1. Een woord dat twee keer zo ver op de lijst staat, komt ongeveer half zo vaak voor. Dit is de wet van Zipf, en die geldt ongeveer voor bijna elke taal. De check probeerde ook een ander aantal."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("zipf-plot", _PLOT_START, "empty", None),
    ("zipf-plot", _PLOT, "correct", None),
    ("zipf-plot", _PLOT_DATA + "plt.plot(ranks, counts)\nplt.xscale(\"log\")\nplt.yscale(\"log\")\nplt.show()", "correct", None),
    ("zipf-plot", _PLOT_DATA + "plt.loglog(ranks, counts)\nplt.title(\"Zipf\")\nplt.xlabel(\"rank\")\nplt.ylabel(\"count\")\nplt.show()", "correct", None),
    ("zipf-plot", _PLOT_DATA + "plt.plot(ranks, counts)\nplt.show()", "wrong", "normal-axes"),
    ("zipf-plot", _PLOT_DATA + "plt.plot(ranks, counts)\nplt.yscale(\"log\")\nplt.show()", "wrong", "y-only"),
    ("zipf-plot", _PLOT_DATA + "plt.semilogx(ranks, counts)\nplt.show()", "wrong", "x-only"),
    ("zipf-plot", _PLOT_DATA + "plt.loglog(counts, ranks)\nplt.show()", "wrong", "values"),
    ("zipf-plot", _PLOT_DATA + "plt.scatter(ranks, counts)\nplt.xscale(\"log\")\nplt.yscale(\"log\")\nplt.show()", "wrong", "kind"),
    ("zipf-plot", _PLOT.replace("import matplotlib.pyplot as plt\n\n", ""), "error", "no-plt"),
    ("zipf-exponent", _K_START, "empty", None),
    ("zipf-exponent", _K, "correct", None),
    ("zipf-exponent", _k("k = np.log(count_b / count_a) / np.log(rank_b / rank_a)"), "correct", None),
    ("zipf-exponent", _k("k = (np.log10(count_a) - np.log10(count_b)) / (np.log10(rank_a) - np.log10(rank_b))"), "correct", None),
    ("zipf-exponent", _k("k = (count_b - count_a) / (rank_b - rank_a)"), "wrong", "no-logs"),
    ("zipf-exponent", _k("k = (np.log10(rank_b) - np.log10(rank_a)) / (np.log10(count_b) - np.log10(count_a))"), "wrong", "swapped"),
    ("zipf-exponent", _k("k = (np.log10(count_b) - np.log10(count_a)) / (rank_b - rank_a)"), "wrong", "y-log-only"),
    ("zipf-exponent", _k("k = np.log10(count_b) - np.log10(count_a)"), "wrong", "no-divide"),
    ("zipf-exponent", _k("k = np.log(count_b) - np.log(count_a)"), "wrong", "no-divide"),
    ("zipf-exponent", _k("k = (np.log(count_b) - np.log(count_a)) / (rank_b - rank_a)"), "wrong", "y-log-only"),
    ("zipf-exponent", _k("k = (np.log10(count_a) - np.log10(count_b)) / (np.log10(rank_b) - np.log10(rank_a))"), "wrong", "sign"),
    ("zipf-exponent", _k("k = np.log(count_b / count_a) / np.log(rank_a / rank_b)"), "wrong", "sign"),
    ("zipf-exponent", _k("k = (np.log10(21) - np.log10(2375)) / 2"), "almost", "other-value"),
]
