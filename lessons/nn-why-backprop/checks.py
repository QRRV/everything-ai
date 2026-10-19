# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-why-backprop: the FLOPs of a forward pass, counted from the sizes of the layers, and
sorting calculations by how they grow with the number of parameters. And the picture: the FLOPs of one forward pass
and of measuring every slope by moving, as the number of parameters N moves on a slider."""
import re

from everything_ai.check import FunctionBox, Sort, Mistake


def forward(n: int) -> int:
    """About 2 FLOPs per parameter for one forward pass."""
    return 2 * n


def moving(n: int) -> int:
    """One forward pass where you are, and one for each parameter moved."""
    return (n + 1) * 2 * n


def whole(n: int, lang: str) -> str:
    """A whole number with its thousands marked, as each language writes it: 20,100 and 20.100."""
    text = f"{n:,}"
    return text.replace(",", ".") if lang == "nl" else text


def picture(n: int, labels: dict[str, str]) -> str:
    """The FLOPs over N from 0 to 1000, in millions: one forward pass dashed, every slope by moving solid, with a dot
    on each at the N of the slider. Drawn as SVG text, once for every stop of the slider (L.explore)."""
    from everything_ai import chart
    ns = list(range(0, 1001, 25))
    p = chart.Panel((0, 1040), (0, 2.2), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])   # room for the tick 1000
    p.line(ns, [moving(k) / 1e6 for k in ns], "ink", 2.2)
    p.line(ns, [forward(k) / 1e6 for k in ns], "ink-3", 2.2, dashed=True)
    p.points([n, n], [moving(n) / 1e6, forward(n) / 1e6], "ink", 4.5)
    p.text(720, 1.3, labels["move"], "ink", anchor="end")
    p.text(990, 0.12, labels["forward"], "ink-3", anchor="end")
    return chart.figure([p], alt=labels["alt"], width=440, height=280, drag=0)   # dragging across it sets N


# The lesson is done when these are correct. "bigger" is the choice question in the lesson.
REQUIRED = ["forward-flops", "growth", "bigger"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_F_HEAD = "def forward_flops(sizes):\n    total = 0\n    for i in range(len(sizes) - 1):\n"
_F_START = _F_HEAD + "        ...\n    return total\n"
_F_BODY = "        total = total + 2 * sizes[i] * sizes[i + 1]"
_F = _F_HEAD + _F_BODY + "\n    return total"


def _f(body: str) -> str:
    return _F_HEAD + body + "\n    return total"


EXERCISES = {
    "forward-flops": FunctionBox(
        name="forward-flops",
        function="forward_flops",
        start=_F_START,
        solution=_F,
        visible=[([3, 4, 2],)],
        hidden=[([784, 128, 10],), ([5, 5],), ([1, 1, 1, 1],), ([10, 20, 30, 40],), ([7],)],
        first=[_DOTS],
        mistakes=[
            Mistake("past-end", error=lambda e: isinstance(e, IndexError),
                    en="With `{call}`, Python stops: `sizes[i + 1]` goes past the end of the list. The last layer has no layer after it, so the loop stops one place earlier: `range(len(sizes) - 1)`.",
                    nl="Met `{call}` stopt Python: `sizes[i + 1]` gaat voorbij het eind van de lijst. Na de laatste layer komt geen layer meer, dus de loop stopt één plek eerder: `range(len(sizes) - 1)`."),
            Mistake("no-two", wrong=_f("        total = total + sizes[i] * sizes[i + 1]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `{got}` is the number of weights. Each weight costs 2 FLOPs, a multiplication and an addition.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `{got}` is het aantal weights. Elke weight kost 2 FLOPs, een vermenigvuldiging en een optelling."),
            Mistake("sum", wrong=_f("        total = total + 2 * (sizes[i] + sizes[i + 1])"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you add the sizes of two layers. Each of the `sizes[i + 1]` neurons has a weight for each of its `sizes[i]` inputs: multiply them.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt de grootte van twee layers op. Elk van de `sizes[i + 1]` neuronen heeft een weight voor elk van zijn `sizes[i]` inputs: vermenigvuldig ze."),
            Mistake("with-bias", wrong=_f("        total = total + 2 * sizes[i] * sizes[i + 1] + sizes[i + 1]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you also count the bias terms. Adding a bias term is the first addition of each sum, so 2 FLOPs per weight already includes it.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt ook de bias terms. Een bias term optellen is de eerste optelling van elke som, dus 2 FLOPs per weight telt die al mee."),
            Mistake("square-neurons", wrong=_f("        total = total + 2 * sizes[i + 1] * sizes[i + 1]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you multiply the number of neurons by itself. With `[3, 4, 2]` that happens to give 40 as well. The weights of layer `i + 1` are its neurons, `sizes[i + 1]`, times its inputs, `sizes[i]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je vermenigvuldigt het aantal neuronen met zichzelf. Bij `[3, 4, 2]` geeft dat toevallig ook 40. De weights van layer `i + 1` zijn zijn neuronen, `sizes[i + 1]`, keer zijn inputs, `sizes[i]`."),
            Mistake("per-parameter", wrong=_f("        total = total + 2 * (sizes[i] + 1) * sizes[i + 1]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you count 2 FLOPs for each bias term too. Adding a bias term is the first addition of each sum, so count 2 FLOPs per weight only.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt ook 2 FLOPs voor elke bias term. Een bias term optellen is de eerste optelling van elke som, dus tel alleen 2 FLOPs per weight."),
            Mistake("square", wrong=_f("        total = total + 2 * sizes[i] * sizes[i]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you multiply the size of a layer by itself. The weights of layer `i + 1` are its neurons, `sizes[i + 1]`, times its inputs, `sizes[i]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je vermenigvuldigt de grootte van een layer met zichzelf. De weights van layer `i + 1` zijn zijn neuronen, `sizes[i + 1]`, keer zijn inputs, `sizes[i]`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with what layer `i + 1` adds to `total`: 2 FLOPs for each of its weights.",
                      "nl": "Vervang de `...` door wat layer `i + 1` bij `total` optelt: 2 FLOPs voor elk van zijn weights."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with one list, and the `def` line of your function has a different number of names between its parentheses. Keep `(sizes)`.",
                          "nl": "De check roept `{call}` aan, met één lijst, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(sizes)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. How many weights does layer `i + 1` have, with `sizes[i + 1]` neurons and `sizes[i]` inputs? Each costs 2 FLOPs.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Hoeveel weights heeft layer `i + 1`, met `sizes[i + 1]` neuronen en `sizes[i]` inputs? Elke kost 2 FLOPs."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Do you work it out from `sizes` itself, for any number of layers?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken je het uit met `sizes` zelf, voor elk aantal layers?"},
            "correct": {"en": "Your function is right in all {n} cases. For `[784, 128, 10]` it gives 203,264 FLOPs, as in the worked example.",
                        "nl": "Je functie klopt in alle {n} gevallen. Voor `[784, 128, 10]` geeft hij 203.264 FLOPs, zoals in het uitgewerkte voorbeeld."},
        },
    ),
    "growth": Sort(
        name="growth",
        groups={
            "n": {"en": "Grows like N: O(N)", "nl": "Groeit zoals N: O(N)"},
            "n2": {"en": "Grows like N²: O(N²)", "nl": "Groeit zoals N²: O(N²)"},
        },
        items={
            "forward": {"en": "**One forward pass**, for one example.", "nl": "**Eén forward pass**, voor één voorbeeld."},
            "moving": {"en": "**Every slope by moving** each parameter in turn.", "nl": "**Elke helling door te verschuiven**, elke parameter om de beurt."},
            "half": {"en": "**Half of the slopes by moving**: you move only half of the parameters, each in turn.",
                     "nl": "**De helft van de hellingen door te verschuiven**: je verschuift maar de helft van de parameters, elk om de beurt."},
            "save": {"en": "**Saving the parameters** to a file.", "nl": "**De parameters bewaren** in een bestand."},
            "pairs": {"en": "**Comparing every pair** of parameters, to see which two are closest.",
                      "nl": "**Elk paar parameters vergelijken**, om te zien welke twee het dichtst bij elkaar liggen."},
            "step": {"en": "**One step of gradient descent**, once every slope is known: each parameter minus η times its slope.",
                     "nl": "**Eén stap van gradient descent**, als elke helling al bekend is: elke parameter min η keer zijn helling."},
        },
        solution={"forward": "n", "moving": "n2", "half": "n2", "save": "n", "pairs": "n2", "step": "n"},
        nudge={
            "forward": {"en": "A forward pass uses each weight once: about 2N FLOPs.", "nl": "Een forward pass gebruikt elke weight één keer: ongeveer 2N FLOPs."},
            "step": {"en": "With the slopes known, each parameter changes once. How many changes is that?",
                     "nl": "Als de hellingen bekend zijn, verandert elke parameter één keer. Hoeveel veranderingen zijn dat?"},
            "save": {"en": "Each parameter is written once. Twice as many parameters, twice as much to write?",
                     "nl": "Elke parameter wordt één keer weggeschreven. Twee keer zoveel parameters, twee keer zoveel te schrijven?"},
            "moving": {"en": "One forward pass of about 2N FLOPs for each of the N parameters: what does that grow like?",
                       "nl": "Eén forward pass van ongeveer 2N FLOPs voor elk van de N parameters: hoe groeit dat?"},
            "half": {"en": "Half the forward passes, but each still costs about 2N FLOPs. What does big-O leave out?",
                     "nl": "De helft van de forward passes, maar elke kost nog steeds ongeveer 2N FLOPs. Wat laat de grote-O-notatie weg?"},
            "pairs": {"en": "Each of the N parameters is compared with each of the others, like the handshakes at the party.",
                      "nl": "Elk van de N parameters wordt vergeleken met elk van de andere, zoals de handdrukken op het feest."},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the calculations again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de berekeningen opnieuw uit."},
            "empty": {"en": "Put each calculation in a group: it grows like N, or like N².",
                      "nl": "Zet elke berekening in een groep: hij groeit zoals N, of zoals N²."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The calculation *{item}* belongs in the other group.", "nl": "De berekening *{item}* hoort in de andere groep."},
            "correct": {"en": "All {n} are right. A fixed bit of work per parameter is O(N); a forward pass per parameter, or a step per pair of parameters, is O(N²), also when you only do half of them.",
                        "nl": "Alle {n} kloppen. Een vast beetje werk per parameter is O(N); een forward pass per parameter, of een stap per paar parameters, is O(N²), ook als je er maar de helft van doet."},
        },
    ),
}

_SOLVED = EXERCISES["growth"].solution

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("forward-flops", _F_START, "empty", None),
    ("forward-flops", _F, "correct", None),
    ("forward-flops", _f("        total = total + sizes[i] * sizes[i + 1] * 2"), "correct", None),
    ("forward-flops", "def forward_flops(sizes):\n    total = 0\n    for a, b in zip(sizes, sizes[1:]):\n        total = total + 2 * a * b\n    return total", "correct", None),
    ("forward-flops", "def forward_flops(sizes):\n    total = 0\n    for i in range(len(sizes)):\n        total = total + 2 * sizes[i] * sizes[i + 1]\n    return total", "error", "past-end"),
    ("forward-flops", _f("        total = total + sizes[i] * sizes[i + 1]"), "wrong", "no-two"),
    ("forward-flops", _f("        total = total + 2 * (sizes[i] + sizes[i + 1])"), "wrong", "sum"),
    ("forward-flops", _f("        total = total + 2 * sizes[i] * sizes[i + 1] + sizes[i + 1]"), "wrong", "with-bias"),
    ("forward-flops", _f("        total = total + 2 * sizes[i] * sizes[i]"), "wrong", "square"),
    ("forward-flops", _f("        total = total + 2 * sizes[i + 1] * sizes[i + 1]"), "wrong", "square-neurons"),
    ("forward-flops", _f("        total = total + 2 * (sizes[i] + 1) * sizes[i + 1]"), "wrong", "per-parameter"),
    ("forward-flops", "def forward_flops(sizes):\n    return 40", "almost", None),
    ("growth", {}, "empty", None),
    ("growth", _SOLVED, "correct", None),
    ("growth", {"forward": "n", "moving": "n2"}, "wrong", "unfinished"),
    ("growth", {**_SOLVED, "half": "n"}, "wrong", "wrong:half"),
    ("growth", {**_SOLVED, "step": "n2"}, "wrong", "wrong:step"),
]
