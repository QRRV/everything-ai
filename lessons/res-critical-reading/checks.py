# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson res-critical-reading: sorting the sentences of an invented abstract into what its table shows
and what goes beyond it, and fake_win_rate, how often the best of several equally good variants beats one baseline run."""
import html
import math
import re

from everything_ai.check import FunctionBox, Mistake, Sort


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def paper_card(labels: dict) -> str:
    """The invented paper as a card: its title, a note that it is invented, the setup and the table of results. HTML,
    not an SVG, so that the text wraps in a narrow window. Ink only."""
    rows = "".join(f'<tr><td style="padding:0.3rem 0.5rem;border-top:1px solid var(--ea-line)">{html.escape(name)}</td>'
                   f'<td style="padding:0.3rem 0.5rem;border-top:1px solid var(--ea-line);text-align:right;font-variant-numeric:tabular-nums">{html.escape(a)}</td>'
                   f'<td style="padding:0.3rem 0.5rem;border-top:1px solid var(--ea-line);text-align:right;font-variant-numeric:tabular-nums">{html.escape(b)}</td></tr>'
                   for name, a, b in labels["rows"])
    return (f'<div style="border:1.2px solid var(--ea-ink-3);border-radius:8px;padding:0.8rem 0.9rem;background:var(--ea-inset);'
            f'max-width:34rem;font-size:0.9rem;line-height:1.45;color:var(--ea-ink)">'
            f'<div style="font-weight:650;font-size:1rem">{html.escape(labels["title"])}</div>'
            f'<div style="color:var(--ea-ink-3);font-size:0.8rem;margin-bottom:0.5rem">{html.escape(labels["invented"])}</div>'
            f'<div style="margin-bottom:0.5rem"><b>{html.escape(labels["setup_name"])}</b> {html.escape(labels["setup"])}</div>'
            f'<table style="border-collapse:collapse;width:100%">'
            f'<tr><th></th><th style="text-align:right;padding:0.3rem 0.5rem">LayerNorm</th>'
            f'<th style="text-align:right;padding:0.3rem 0.5rem">WarmNorm</th></tr>{rows}</table></div>')


def chance_picture(labels: dict) -> str:
    """The chance that the best of n equally good variants, each run once, beats one run of the baseline: n / (n + 1),
    for n from 1 to 20, with dots at 1, 4, 9 and 19. Nothing here is learned or a loss: ink."""
    from everything_ai import chart
    p = chart.Panel((0, 20.5), (0.4, 1.0), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xticks=[1, 5, 10, 15, 20], yticks=[0.5, 0.75, 1.0])
    ns = list(range(1, 21))
    p.line(ns, [n / (n + 1) for n in ns], "ink", 2.2)
    marks = [1, 4, 9, 19]
    p.points(marks, [n / (n + 1) for n in marks], "ink", r=4.5)
    for n, dx, dy in ((1, 0.6, -0.035), (4, 0.6, -0.035), (9, 0.5, -0.035), (19, -0.4, -0.05)):
        p.text(n + dx, n / (n + 1) + dy, number(n / (n + 1), 2, labels.get("lang", "en")), "ink-2", "start" if dx > 0 else "end")
    svg = chart.figure([p], alt=labels["alt"], width=330, height=250)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# --- exercise 1: overclaims ---------------------------------------------------------------------------------------------------

_SORT_SOLUTION = {"loss": "shown", "lower": "beyond", "slower": "shown", "sota": "beyond", "params": "shown", "every-lr": "beyond"}

# --- exercise 2: fake_win_rate ------------------------------------------------------------------------------------------------

_HEAD = ("import numpy as np\n\n\ndef fake_win_rate(n_variants, n_trials, seed):\n"
         "    # n_trials experiments in which every variant and the baseline are equally good\n"
         "    rng = np.random.default_rng(seed)\n")
_DRAW = "    variants = rng.normal(size=(n_trials, n_variants))\n    baseline = rng.normal(size=n_trials)\n"
_WIN = "    return (variants.max(axis=1) > baseline).mean()\n"
_START = (_HEAD + "    # one score for every variant, and one for the baseline, in every experiment\n    ...\n"
          "    # the share of experiments in which the best variant scores higher than the baseline\n    ...\n")


def _w(draw: str = _DRAW, win: str = _WIN) -> str:
    return _HEAD + draw + win


def _value(v):
    try:
        if isinstance(v, (str, bytes, bool)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _close(got, want) -> bool:
    """A share of random experiments: close to the course's own, within a little more than its random wobble. A larger
    number, such as a count of wins, within the same share of itself."""
    g, w = _value(got), _value(want)
    return g is not None and w is not None and abs(g - w) < 0.02 * max(1.0, abs(w))


def _show(v):
    x = _value(v)
    if x is None and isinstance(v, (list, tuple)) or getattr(v, "ndim", 0) > 0:
        n = len(v)
        kind = {"en": "a list" if isinstance(v, list) else "a tuple" if isinstance(v, tuple) else "an array",
                "nl": "een list" if isinstance(v, list) else "een tuple" if isinstance(v, tuple) else "een array"}
        return {lang: f"{kind[lang]} of {number(n, 0, lang)} value{'' if n == 1 else 's'}" if lang == "en"
                else f"{kind[lang]} van {number(n, 0, lang)} {'waarde' if n == 1 else 'waarden'}" for lang in ("en", "nl")}
    if x is None:
        return repr(v)
    if x != x or math.isinf(x):
        return str(x)
    digits = 0 if abs(x) >= 100 else 3
    return {"en": number(x, digits, "en"), "nl": number(x, digits, "nl")}


def _broadcast(e: BaseException) -> bool:
    return isinstance(e, ValueError) and "could not be broadcast" in str(e)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_FORMULA = Mistake("formula", matches=lambda got, text: re.search(
                       r"\.(normal|standard_normal|random|uniform|integers|choice|permutation|shuffle|gauss|normalvariate)\s*\(", text) is None,
                   en="Your function gives the right shares, but it draws no scores: it works them out with the formula of The idea. The exercise checks that formula: draw the scores with `rng`, and count how often the best variant wins.",
                   nl="Je functie geeft de goede aandelen, maar trekt geen scores: hij rekent ze uit met de formule uit Het idee. De oefening controleert die formule: trek de scores met `rng`, en tel hoe vaak de beste variant wint.")

_WRONG = {
    "own-baseline": _w(draw="    variants = rng.normal(size=(n_trials, n_variants))\n    baseline = rng.normal(size=(n_trials, n_variants))\n",
                       win="    return (variants > baseline).any(axis=1).mean()\n"),
    "one-variant": _w(win="    return (variants[:, 0] > baseline).mean()\n"),
    "count": _w(win="    return (variants.max(axis=1) > baseline).sum()\n"),
}

EXERCISES = {
    "overclaims": Sort(
        name="overclaims",
        groups={
            "shown": {"en": "Shown by the table", "nl": "Laat de tabel zien"},
            "beyond": {"en": "Claims more than the table shows", "nl": "Claimt meer dan de tabel laat zien"},
        },
        items={
            "loss": {"en": "In one run each, the validation loss was 1.58 with WarmNorm and 1.62 with LayerNorm.",
                     "nl": "In één run per methode was de validation loss 1,58 met WarmNorm en 1,62 met LayerNorm."},
            "lower": {"en": "WarmNorm gives a small GPT a lower validation loss than LayerNorm does.",
                      "nl": "WarmNorm geeft een kleine GPT een lagere validation loss dan LayerNorm."},
            "slower": {"en": "A training step with WarmNorm takes 3% longer.",
                       "nl": "Een trainingsstap met WarmNorm duurt 3% langer."},
            "sota": {"en": "WarmNorm sets a new state of the art for character-level language models.",
                     "nl": "WarmNorm is de nieuwe state of the art voor language models op tekenniveau."},
            "params": {"en": "WarmNorm learns 2 numbers more than LayerNorm, in each place it is used.",
                       "nl": "WarmNorm leert 2 getallen meer dan LayerNorm, op elke plek waar het gebruikt wordt."},
            "every-lr": {"en": "WarmNorm makes training more stable at every learning rate.",
                         "nl": "WarmNorm maakt trainen stabieler bij elke learning rate."},
        },
        solution=_SORT_SOLUTION,
        nudge={
            "loss": {"en": "Which row of the table has these two numbers, and how many runs does the setup give each method?",
                     "nl": "Welke rij van de tabel heeft deze twee getallen, en hoeveel runs geeft de opzet elke methode?"},
            "lower": {"en": "Did LayerNorm get the same five learning rates to choose from? And how many runs is each number?",
                      "nl": "Kreeg LayerNorm dezelfde vijf learning rates om uit te kiezen? En uit hoeveel runs komt elk getal?"},
            "slower": {"en": "Compare the two times per step in the table.",
                       "nl": "Vergelijk de twee tijden per stap in de tabel."},
            "sota": {"en": "Which methods besides LayerNorm were compared?",
                     "nl": "Met welke methoden naast LayerNorm is vergeleken?"},
            "params": {"en": "Look at the row with the numbers to learn.",
                       "nl": "Kijk naar de rij met de te leren getallen."},
            "every-lr": {"en": "At how many learning rates was the comparison run?",
                         "nl": "Bij hoeveel learning rates is de vergelijking gedraaid?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the sentences again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de zinnen opnieuw uit."},
            "empty": {"en": "Put each sentence in a group: shown by the table, or claims more than the table shows.",
                      "nl": "Zet elke zin in een groep: laat de tabel zien, of claimt meer dan de tabel laat zien."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The sentence “{item}” belongs in the other group.",
                      "nl": "De zin ‘{item}’ hoort in de andere groep."},
            "correct": {"en": "All {n} are right. The loss of one run each, the time per step and the extra numbers are in the table. A lower loss in general needs LayerNorm tuned the same way and more than one run each, \"state of the art\" needs more methods than one baseline, and \"every learning rate\" needs more than one.",
                        "nl": "Alle {n} kloppen. De loss van één run per methode, de tijd per stap en de extra getallen staan in de tabel. Een lagere loss in het algemeen vraagt LayerNorm op dezelfde manier getuned en meer dan één run per methode, \"state of the art\" vraagt meer methoden dan één baseline, en \"elke learning rate\" vraagt er meer dan één."},
        },
    ),
    "fake-wins": FunctionBox(
        name="fake-wins",
        function="fake_win_rate",
        start=_START,
        solution=_w(),
        visible=[(4, 20000, 0)],
        hidden=[(1, 20000, 1), (9, 20000, 2), (19, 20000, 3)],
        show=_show,
        compare=_close,
        first=[_DOTS],
        form=[_FORMULA],
        mistakes=[
            Mistake("axis", error=_broadcast,
                    en="With `{call}`, NumPy stops: it cannot compare the two arrays, their shapes do not fit. The best variants and `baseline` should both have one number per experiment. With one row per experiment, `rng.normal(size=(n_trials, n_variants))`, the best variant of each is the largest number in its row: `variants.max(axis=1)`.",
                    nl="Met `{call}` stopt NumPy: hij kan de twee arrays niet vergelijken, hun shapes passen niet. De beste varianten en `baseline` moeten allebei één getal per experiment hebben. Met één rij per experiment, `rng.normal(size=(n_trials, n_variants))`, is de beste variant van elk het grootste getal in zijn rij: `variants.max(axis=1)`."),
            Mistake("own-baseline", wrong=_WRONG["own-baseline"],
                    en="`{call}` gives {got}, and it should give about {want}: every variant gets its own run of the baseline. In a paper, all variants are compared with one run of the baseline: draw one baseline score per experiment, and compare the best variant with it.",
                    nl="`{call}` geeft {got}, en dat moet ongeveer {want} zijn: elke variant krijgt een eigen run van de baseline. In een paper worden alle varianten met één run van de baseline vergeleken: trek één baselinescore per experiment, en vergelijk de beste variant daarmee."),
            Mistake("one-variant", wrong=_WRONG["one-variant"],
                    en="`{call}` gives {got}, and it should give about {want}: your share stays near 0.5, whatever the number of variants, as for one variant against one baseline run. Compare the best of the variants, `variants.max(axis=1)`, with one baseline score per experiment, as a paper that reports its best variant does.",
                    nl="`{call}` geeft {got}, en dat moet ongeveer {want} zijn: je aandeel blijft rond 0,5, hoeveel varianten er ook zijn, zoals bij één variant tegen één run van de baseline. Vergelijk de beste van de varianten, `variants.max(axis=1)`, met één baselinescore per experiment, zoals een paper doet die zijn beste variant laat zien."),
            Mistake("count", wrong=_WRONG["count"],
                    en="`{call}` gives {got}, and it should give about {want}: that is the number of wins, not their share. Divide by the number of experiments, or take `.mean()` of the comparison.",
                    nl="`{call}` geeft {got}, en dat moet ongeveer {want} zijn: dat is het aantal overwinningen, niet hun aandeel. Deel door het aantal experimenten, of neem `.mean()` van de vergelijking."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: the scores of the variants and the baseline, and the share of experiments in which the best variant wins.",
                      "nl": "Vervang de twee `...`: de scores van de varianten en de baseline, en het aandeel experimenten waarin de beste variant wint."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the number of variants, the number of experiments and a seed, and the `def` line of your function has another number of names between its parentheses. Keep `(n_variants, n_trials, seed)`.",
                          "nl": "De check roept `{call}` aan, met het aantal varianten, het aantal experimenten en een seed, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(n_variants, n_trials, seed)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give about {want}. Do you draw one score per variant and one baseline score per experiment, and count how often the best variant is higher?",
                      "nl": "`{call}` geeft {got}, en dat moet ongeveer {want} zijn. Trek je per experiment één score per variant en één baselinescore, en tel je hoe vaak de beste variant hoger is?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give about {want}. Does your function work for 1 variant, and for many?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet ongeveer {want} zijn. Werkt je functie ook voor 1 variant, en voor veel?"},
            "correct": {"en": "Your simulation is right in all {n} cases: with 1 variant a fake win comes half of the time, with 4 about 4 times in 5, with 19 about 19 times in 20. The baseline only wins when it happens to be the best of all the runs.",
                        "nl": "Je simulatie klopt in alle {n} gevallen: met 1 variant komt een nepoverwinning de helft van de keren, met 4 ongeveer 4 keer op 5, met 19 ongeveer 19 keer op 20. De baseline wint alleen als hij toevallig de beste van alle runs is."},
        },
    ),
}

# The lesson is done when these are correct. "first-ask" is the choice question, "carried-claim" the open question.
REQUIRED = ["overclaims", "fake-wins", "first-ask", "carried-claim"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("overclaims", {}, "empty", None),
    ("overclaims", dict(_SORT_SOLUTION), "correct", None),
    ("overclaims", {"loss": "shown", "lower": "beyond"}, "wrong", "unfinished"),
    ("overclaims", {**_SORT_SOLUTION, "lower": "shown"}, "wrong", "wrong:lower"),
    ("overclaims", {**_SORT_SOLUTION, "every-lr": "shown"}, "wrong", "wrong:every-lr"),
    ("fake-wins", _START, "empty", None),
    ("fake-wins", _w(), "correct", None),
    ("fake-wins", _w(win="    best = variants.min(axis=1)\n    return np.mean(best < baseline)\n"), "correct", None),
    ("fake-wins", _HEAD + "    wins = 0\n    for _ in range(n_trials):\n        scores = rng.normal(size=n_variants)\n"
     "        if scores.max() > rng.normal():\n            wins = wins + 1\n    return wins / n_trials\n", "correct", None),
    ("fake-wins", _w(win="    return (variants.max(axis=0) > baseline).mean()\n"), "error", "axis"),
    ("fake-wins", _w(draw="    variants = rng.normal(size=(n_variants, n_trials))\n    baseline = rng.normal(size=n_trials)\n"), "error", "axis"),
    ("fake-wins", _HEAD + "    baseline = rng.normal(size=n_trials)\n    variants = rng.normal(size=(n_trials, n_variants))\n"
     "    return (variants.max(axis=1) > baseline).sum()\n", "wrong", "count"),
    ("fake-wins", "def fake_win_rate(n_variants, n_trials, seed):\n    return n_variants / (n_variants + 1)\n", "wrong", "formula"),
] + [("fake-wins", code, "wrong", key) for key, code in _WRONG.items()]
