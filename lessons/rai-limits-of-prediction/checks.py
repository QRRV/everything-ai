# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-limits-of-prediction: a function that gives the R² of a model's predictions, used on two
made-up models of a child's school grade, sorting tasks into perception, automating judgment and predicting social
outcomes, and the picture of the predictions against the real grades."""
import html
import re

from everything_ai.check import Code, CodeBox, Mistake, Sort
from everything_ai import yours

# Ten children, made up by the course: the grade point average at 15 (1 to 4), and what two models predicted from
# what was known about them when they were younger. `BEST` stands for a careful model, `SIMPLE` for one with four
# variables.
ACTUAL = [2.25, 3.5, 2.75, 4.0, 1.75, 3.0, 2.5, 3.25, 2.0, 3.0]
BEST = [2.6, 3.1, 2.9, 3.1, 2.6, 2.9, 2.8, 2.4, 2.9, 3.1]
SIMPLE = [2.5, 2.8, 2.8, 3.0, 2.9, 3.1, 2.6, 2.7, 2.7, 2.9]


def r2(y: list, pred: list) -> float:
    """1 minus the MSE divided by the variance of y."""
    n = len(y)
    mean = sum(y) / n
    mse = sum((a - b) ** 2 for a, b in zip(y, pred)) / n
    variance = sum((a - mean) ** 2 for a in y) / n
    return 1 - mse / variance


def picture(labels: dict) -> str:
    """Each child's real grade, the careful model's prediction, a line between them, and the mean as a dashed line.
    `labels` gives "title", "xlabel", "ylabel", "mean" and "alt"."""
    from everything_ai import chart
    n = len(ACTUAL)
    mean = sum(ACTUAL) / n
    panel = chart.Panel(x=(0.4, n + 3.7), y=(1.5, 4.25), title=labels["title"], xlabel=labels["xlabel"],
                        ylabel=labels["ylabel"], xticks=list(range(1, n + 1)), yticks=[2, 3, 4])
    panel.line([0.5, n + 0.5], [mean, mean], "ink", width=1.4, dashed=True)
    panel.text(n + 0.65, mean - 0.06, labels["mean"], "ink-2", "start")
    for i in range(n):
        panel.line([i + 1, i + 1], [ACTUAL[i], BEST[i]], "residual", width=2)
        panel.points([i + 1], [ACTUAL[i]], "data", r=5)
        panel.points([i + 1], [BEST[i]], "pred", r=5, hollow=True)
    return chart.figure([panel], alt=labels["alt"], width=600, height=280)


def _r2_picture(seen) -> str:
    """The learner's R² for both models next to the course's, as bars from 0 to 1 with the course's value marked."""
    mine = seen.mine
    if mine is None or not mine.has_value or not isinstance(mine.value, list):
        return ""
    names = seen.word({"en": ["careful model", "simple model"], "nl": ["zorgvuldig model", "eenvoudig model"]})
    right = [r2(ACTUAL, BEST), r2(ACTUAL, SIMPLE)]
    rows = []
    for k, name in enumerate(names):
        got = mine.value[k] if k < len(mine.value) else None
        try:
            shown = seen.word({"en": f"{float(got):.2f}", "nl": f"{float(got):.2f}".replace(".", ",")})
            width = max(0.0, min(1.0, float(got))) * 100
        except (TypeError, ValueError):
            shown, width = html.escape(str(got)), 0.0
        bar = (f'<div style="position:relative;width:12rem;height:0.8rem;background:var(--ea-inset);border-radius:var(--ea-radius-small)">'
               f'<div style="width:{width:.1f}%;height:100%;background:var(--ea-data);border-radius:var(--ea-radius-small)"></div>'
               f'<div style="position:absolute;top:-0.15rem;left:{right[k] * 100:.1f}%;width:2px;height:1.1rem;background:var(--ea-ink)"></div></div>')
        course = seen.word({"en": f"the course: {right[k]:.2f}", "nl": f"de course: {right[k]:.2f}".replace(".", ",")})
        rows.append(f'<tr><td style="padding:0.2rem 0.8rem 0.2rem 0">{name}</td>'
                    f'<td style="padding:0.2rem 0.8rem;font-weight:600;color:var(--ea-ink)">{shown}</td>'
                    f'<td style="padding:0.2rem 0.8rem">{bar}</td>'
                    f'<td style="padding:0.2rem 0;color:var(--ea-ink-2)">{course}</td></tr>')
    title = seen.word({"en": "Your R² for both models, from 0 to 1", "nl": "Jouw R² voor beide modellen, van 0 tot 1"})
    table = f'<table style="font-family:var(--ea-font-ui);font-size:0.9rem;border-collapse:collapse">{"".join(rows)}</table>'
    return yours.figure(title, table)


# The lesson is done when these are correct. "target" is the choice question in the lesson.
REQUIRED = ["r2", "kinds", "target"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_ARRAYS = ("actual = np.array(" + str(ACTUAL) + ")\nbest = np.array(" + str(BEST) + ")\nsimple = np.array(" + str(SIMPLE) + ")")
_HEAD = "import numpy as np\n\n# grade point average at 15, made up by the course\n" + _ARRAYS + "\n\ndef r2(y, pred):\n"
_LAST = "\n[r2(actual, best), r2(actual, simple)]"
_START = _HEAD + "    ...\n" + _LAST + "\n"
_BODY = ("    mse = np.mean((y - pred) ** 2)\n"
         "    variance = np.mean((y - y.mean()) ** 2)\n"
         "    return 1 - mse / variance\n")
_SOLUTION = _HEAD + _BODY + _LAST


def _f(body: str) -> str:
    return _HEAD + body + _LAST


EXERCISES = {
    "r2": CodeBox(
        name="r2",
        picture=_r2_picture,
        start=_START,
        solution=_SOLUTION,
        digits=2,
        keep=["actual = np.array(" + str(ACTUAL) + ")", "best = np.array(" + str(BEST) + ")", "def r2(y, pred):",
              "[r2(actual, best), r2(actual, simple)]"],
        # The check also runs the box with another simple model: one that always predicts the mean grade.
        others=[{"simple": Code("np.array([2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8])")}],
        first=[_DOTS],
        mistakes=[
            Mistake("no-return", matches=lambda got, text: isinstance(got, list) and None in got,
                    en="Your last line gives `{got}`: your function reaches its end without a `return`, so it gives `None`. End it with `return` and R².",
                    nl="Je laatste regel geeft `{got}`: je functie komt aan het eind zonder `return`, dus ze geeft `None`. Sluit af met `return` en R²."),
            Mistake("pred-variance", wrong=_f(_BODY.replace("np.mean((y - y.mean()) ** 2)", "np.mean((pred - pred.mean()) ** 2)")),
                    en="Your last line gives `{got}`: far below 0. R² divides by the variance of the real grades `y`, not of the predictions: those hardly spread at all.",
                    nl="Je laatste regel geeft `{got}`: ver onder 0. R² deelt door de variantie van de echte cijfers `y`, niet van de voorspellingen: die spreiden nauwelijks."),
            Mistake("flipped", wrong=_f(_BODY.replace("return 1 - mse / variance", "return 1 - variance / mse")),
                    en="Your last line gives `{got}`: the division is the wrong way round. R² is 1 minus the MSE divided by the variance.",
                    nl="Je laatste regel geeft `{got}`: de deling is omgekeerd. R² is 1 min de MSE gedeeld door de variantie."),
            Mistake("mse", wrong=_f(_BODY.replace("return 1 - mse / variance", "return mse")),
                    en="Your last line gives `{got}`: those are the MSEs, the mean squared misses. R² compares the MSE with the variance of `y`.",
                    nl="Je laatste regel geeft `{got}`: dat zijn de MSE's, de gemiddelde gekwadrateerde missers. R² vergelijkt de MSE met de variantie van `y`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the body of `r2`: the MSE, the variance of `y`, and a `return` with R².",
                      "nl": "Vervang de `...` door de body van `r2`: de MSE, de variantie van `y`, en een `return` met R²."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another simple model by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een ander eenvoudig model."},
            "no_value": {"en": "The last line of the box shows nothing. Keep `[r2(actual, best), r2(actual, simple)]` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Houd `[r2(actual, best), r2(actual, simple)]` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does your function give back 1 minus the MSE of `pred` divided by the variance of `y`?",
                      "nl": "Je laatste regel geeft `{got}`. Geeft je functie 1 min de MSE van `pred` gedeeld door de variantie van `y` terug?"},
            "other": {"en": "Right for these models. But with {where}, your last line gives `{got}`, and it should give `{want}`. Does your function work out R² from `y` and `pred` themselves?",
                      "nl": "Goed voor deze modellen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Rekent je functie R² uit `y` en `pred` zelf uit?"},
            "other_error": {"en": "Right for these models. But with {where}, your code stops with an error. Does your function work out R² from `y` and `pred` themselves?",
                            "nl": "Goed voor deze modellen. Maar met {where} stopt je code met een fout. Rekent je functie R² uit `y` en `pred` zelf uit?"},
            "correct": {"en": "`{got}`: the careful model explains a fifth of how far the grades spread, and the simple one a little less. For 4 of the 10 children, its prediction is still almost a whole grade point off. The check also tried a model that always predicts the mean: it gets 0.",
                        "nl": "`{got}`: het zorgvuldige model verklaart een vijfde van hoe ver de cijfers spreiden, en het eenvoudige iets minder. Voor 4 van de 10 kinderen zit de voorspelling er nog bijna een heel punt naast. De check probeerde ook een model dat altijd het gemiddelde voorspelt: dat krijgt 0."},
        },
    ),
    "kinds": Sort(
        name="kinds",
        groups={
            "perception": {"en": "Perception", "nl": "Waarneming"},
            "judgment": {"en": "Automating judgment", "nl": "Oordelen automatiseren"},
            "social": {"en": "Predicting social outcomes", "nl": "Sociale uitkomsten voorspellen"},
        },
        items={
            "speech": {"en": "**Speech to text.** Writing down what someone says in a recording.",
                       "nl": "**Spraak naar tekst.** Opschrijven wat iemand in een opname zegt."},
            "scan": {"en": "**A scan.** Spotting a broken bone on an X-ray.",
                     "nl": "**Een scan.** Een gebroken bot zien op een röntgenfoto."},
            "spam": {"en": "**Spam.** Moving junk mail out of the inbox.",
                     "nl": "**Spam.** Ongewenste mail uit de inbox halen."},
            "essay": {"en": "**An essay.** Giving a grade to a school essay.",
                      "nl": "**Een opstel.** Een cijfer geven voor een opstel op school."},
            "again": {"en": "**Again?** Whether someone who was released will commit a crime again.",
                      "nl": "**Opnieuw?** Of iemand die vrijkwam opnieuw een misdrijf pleegt."},
            "hire": {"en": "**A good hire?** How well an applicant will do the job in two years.",
                     "nl": "**Een goede keuze?** Hoe goed een sollicitant het werk over twee jaar doet."},
        },
        solution={"speech": "perception", "scan": "perception", "spam": "judgment", "essay": "judgment", "again": "social", "hire": "social"},
        nudge={
            "speech": {"en": "Is the answer already there, in the recording?", "nl": "Zit het antwoord er al, in de opname?"},
            "scan": {"en": "Could an expert check the answer by looking at the same picture?", "nl": "Kan een deskundige het antwoord controleren door naar dezelfde foto te kijken?"},
            "spam": {"en": "Is there one right answer, or does it take judgment?", "nl": "Is er één goed antwoord, of vraagt het een oordeel?"},
            "essay": {"en": "Would two teachers always give the same grade?", "nl": "Zouden twee docenten altijd hetzelfde cijfer geven?"},
            "again": {"en": "Is this about the future of a person's life?", "nl": "Gaat dit over de toekomst van iemands leven?"},
            "hire": {"en": "Can the answer be known today?", "nl": "Kan het antwoord vandaag al bekend zijn?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the tasks again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de taken opnieuw uit."},
            "empty": {"en": "Put each task with the kind it belongs to.",
                      "nl": "Zet elke taak bij de soort waar ze bij hoort."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The task *{item}* belongs to another kind.",
                      "nl": "De taak *{item}* hoort bij een andere soort."},
            "correct": {"en": "All {n} are right. In perception, the answer is already in the data, and AI has become good at it. Judgment can be learned from many examples, but people disagree about some answers. A social outcome lies in the future of a life, and depends on much that no data holds.",
                        "nl": "Alle {n} kloppen. Bij waarneming zit het antwoord al in de data, en daar is AI goed in geworden. Een oordeel kun je leren uit veel voorbeelden, maar over sommige antwoorden zijn mensen het oneens. Een sociale uitkomst ligt in de toekomst van een leven, en hangt af van veel wat in geen enkele data staat."},
        },
    ),
}

_SORTED = {"speech": "perception", "scan": "perception", "spam": "judgment", "essay": "judgment", "again": "social", "hire": "social"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("r2", _START, "empty", None),
    ("r2", _SOLUTION, "correct", None),
    ("r2", _f("    return 1 - np.mean((y - pred) ** 2) / np.var(y)\n"), "correct", None),
    ("r2", _f("    mse = np.mean((y - pred) ** 2)\n    variance = np.mean((y - y.mean()) ** 2)\n"), "wrong", "no-return"),
    ("r2", _f(_BODY.replace("np.mean((y - y.mean()) ** 2)", "np.mean((pred - pred.mean()) ** 2)")), "wrong", "pred-variance"),
    ("r2", _f(_BODY.replace("return 1 - mse / variance", "return 1 - variance / mse")), "wrong", "flipped"),
    ("r2", _f(_BODY.replace("return 1 - mse / variance", "return mse")), "wrong", "mse"),
    ("r2", _f(_BODY.replace("return 1 - mse / variance", "return mse / variance")), "wrong", None),
    ("r2", _HEAD + "    return 0\n\n[0.2, 0.15]", "wrong", "changed-line"),
    ("r2", _f("    if pred[0] == 2.6:\n        return 1 - 0.348 / 0.435\n    return 1 - 0.37 / 0.435\n"), "almost", "other-value"),
    ("r2", _f("    ...\n    return 0\n"), "wrong", "dots-left"),
    ("kinds", {}, "empty", None),
    ("kinds", _SORTED, "correct", None),
    ("kinds", {"speech": "perception", "again": "social"}, "wrong", "unfinished"),
    ("kinds", {**_SORTED, "essay": "social"}, "wrong", "wrong:essay"),
    ("kinds", {**_SORTED, "hire": "judgment"}, "wrong", "wrong:hire"),
    ("kinds", {**_SORTED, "scan": "judgment"}, "wrong", "wrong:scan"),
]
