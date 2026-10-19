# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson repro-guided-claim: verdict, the rule written down before any run that says, from the
accuracy logs of one run, whether it shows the claim of the grokking paper. Saved as repro/plan.py, the first part of
the learner's reproduction kit."""
import math
import re

from everything_ai.check import FunctionBox, Mistake

THRESHOLD, DELAY = 0.99, 10


def sketch_picture(labels: dict) -> str:
    """A sketch of the claim, not the paper's data: training accuracy, solid, reaches 99% at step t; validation
    accuracy, dashed, stays low and reaches 99% at step v, more than 10 times later. Accuracy is a metric that is not
    optimised, so both are ink (style guide 1.3) and differ by line style and label. Log x-axis of steps, a thin line
    at 99%."""
    from everything_ai import chart
    p = chart.Panel((10, 1e6), (0, 1.12), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xlog=True, yticks=[0, 0.5, 1.0])

    def s_curve(x, middle, width):
        return 1 / (1 + math.exp(-(math.log10(x) - middle) / width))

    def reaches(level, middle, width):   # the step at which s_curve reaches `level`
        return 10 ** (middle + width * math.log(level / (1 - level)))

    xs = [10 ** (1 + i / 40) for i in range(201)]
    p.line([10, 1e6], [THRESHOLD, THRESHOLD], "ink-3", 1.0)
    p.line(xs, [min(1.0, 1.02 * s_curve(x, 2.4, 0.18)) for x in xs], "ink", 2.2)
    p.line(xs, [0.02 + 0.98 * s_curve(x, 4.6, 0.14) for x in xs], "ink", 2.2, dashed=True)
    t, v = reaches(THRESHOLD / 1.02, 2.4, 0.18), reaches((THRESHOLD - 0.02) / 0.98, 4.6, 0.14)   # about 1,070 and 174,000
    p.points([t, v], [THRESHOLD, THRESHOLD], "ink", r=4)
    p.text(t, 1.07, "t", "ink", "middle")
    p.text(v, 1.07, "v", "ink", "middle")
    p.text(10 ** 2.45, 0.72, labels["train"], "ink", "end")   # left of the training curve, which is at 0.72 near step 350
    p.text(10 ** 4.5, 0.6, labels["val"], "ink", "end")       # left of the validation curve, which is at 0.6 near step 45,000
    return chart.figure([p], alt=labels["alt"], width=330, height=260)


# --- exercise 1: verdict --------------------------------------------------------------------------------------------------------

_HEAD = ("# Your reproduction plan: when one run shows the claim, decided before any run.\n"
         "THRESHOLD = 0.99   # an accuracy counts as reached from 99% on\n"
         "DELAY = 10         # \"long after\": validation reaches it at a step at least 10 times as late as training\n\n\n")
_FIRST = ("def first_step(steps, accuracy, threshold):\n"
          "    # the first step at which the accuracy reaches the threshold, or None when it never does\n"
          "    for i in range(len(steps)):\n        if accuracy[i] >= threshold:\n            return steps[i]\n    return None\n\n\n")
_VERDICT = ("def verdict(steps, train_acc, val_acc):\n"
            "    # \"did not fit\", \"not reached\", \"no delay\" or \"grokked\", for the logs of one run\n"
            "    t = first_step(steps, train_acc, THRESHOLD)\n    v = first_step(steps, val_acc, THRESHOLD)\n"
            "    if t is None:\n        return \"did not fit\"\n    if v is None:\n        return \"not reached\"\n"
            "    if v < DELAY * t:\n        return \"no delay\"\n    return \"grokked\"\n")
_START = (_HEAD + "def first_step(steps, accuracy, threshold):\n"
          "    # the first step at which the accuracy reaches the threshold, or None when it never does\n    ...\n\n\n"
          "def verdict(steps, train_acc, val_acc):\n"
          "    # \"did not fit\", \"not reached\", \"no delay\" or \"grokked\", for the logs of one run\n    ...\n")


def _p(first: str = _FIRST, verdict: str = _VERDICT) -> str:
    return _HEAD + first + verdict


_STEPS = "[10, 100, 1000, 10000, 100000]"
_VISIBLE = (_STEPS, "[0.3, 0.95, 1.0, 1.0, 1.0]", "[0.0, 0.01, 0.02, 0.4, 1.0]")
_HIDDEN = [
    (_STEPS, "[0.3, 0.95, 1.0, 1.0, 1.0]", "[0.0, 0.5, 0.99, 1.0, 1.0]"),     # validation along with training: no delay
    ("[100, 200, 500, 1000, 2000, 5000]", "[0.995, 1.0, 1.0, 1.0, 1.0, 1.0]",
     "[0.0, 0.2, 0.995, 1.0, 1.0, 1.0]"),                                        # validation 5 times as late: still no delay
    (_STEPS, "[0.3, 0.95, 1.0, 1.0, 1.0]", "[0.0, 0.01, 0.02, 0.05, 0.3]"),   # validation never reaches 99%
    (_STEPS, "[0.2, 0.5, 0.8, 0.9, 0.95]", "[0.0, 0.01, 0.02, 0.05, 0.3]"),    # training never reaches 99%
    (_STEPS, "[0.3, 0.99, 1.0, 1.0, 1.0]", "[0.0, 0.0, 0.0, 0.99, 1.0]"),      # 99% exactly, 100 times later
    (_STEPS, "[0.3, 0.95, 1.0, 1.0, 1.0]", "[0.0, 0.0, 0.0, 0.99, 1.0]"),      # exactly 10 times later
    (_STEPS, "[0.3, 0.95, 1.0, 1.0, 1.0]", "[0.0, 0.0, 0.0, 0.5, 0.99]"),      # validation reaches 99% exactly, at the last step
]


class _Named(list):
    """A list of a test case, written in the check's messages the way the lesson writes it."""

    def __init__(self, shown: str):
        super().__init__(eval(shown))
        self.shown = shown

    def __repr__(self):
        return self.shown

    def __deepcopy__(self, memo):
        return list(self)


def _case(steps: str, train: str, val: str) -> tuple:
    return (_Named(steps), _Named(train), _Named(val))


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_WRONG = {
    "strict": _p(first=_FIRST.replace("accuracy[i] >= threshold", "accuracy[i] > threshold")),
    "index": _p(first=_FIRST.replace("return steps[i]", "return i")),
    "delay-strict": _p(verdict=_VERDICT.replace("if v < DELAY * t:", "if v <= DELAY * t:")),
    "no-delay-check": _p(verdict=_VERDICT.replace("    if v < DELAY * t:\n        return \"no delay\"\n", "")),
    "later": _p(verdict=_VERDICT.replace("if v < DELAY * t:", "if v <= t:")),
    "order": _p(verdict=_VERDICT.replace("    if t is None:\n        return \"did not fit\"\n    if v is None:\n        return \"not reached\"\n",
                                         "    if v is None:\n        return \"not reached\"\n    if t is None:\n        return \"did not fit\"\n")),
}

EXERCISES = {
    "verdict": FunctionBox(
        name="verdict",
        function="verdict",
        start=_START,
        solution=_p(),
        visible=[_case(*_VISIBLE)],
        hidden=[_case(*c) for c in _HIDDEN],
        python_values=True,
        save_as=("repro", "plan"),
        first=[_DOTS],
        mistakes=[
            Mistake("strict", wrong=_WRONG["strict"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: an accuracy of exactly 0.99 does not count for you. The plan says reached from 99% on: `>=`, not `>`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: een accuracy van precies 0,99 telt bij jou niet. Het plan zegt bereikt vanaf 99%: `>=`, niet `>`."),
            Mistake("index", wrong=_WRONG["index"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: your `first_step` gives back the place in the list, not the step. With steps 10, 100, 1000, … place 4 is step 100,000: give back `steps[i]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je `first_step` geeft de plek in de lijst terug, niet de stap. Met stappen 10, 100, 1000, … is plek 4 stap 100.000: geef `steps[i]` terug."),
            Mistake("delay-strict", wrong=_WRONG["delay-strict"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: validation reaches 99% exactly 10 times as late as training, and the plan says at least 10 times. That is a delay: `v < DELAY * t` is no delay.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de validatie haalt 99% precies 10 keer zo laat als de training, en het plan zegt minstens 10 keer. Dat is een vertraging: `v < DELAY * t` is geen vertraging."),
            Mistake("no-delay-check", wrong=_WRONG["no-delay-check"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: validation reaches 99% at the same step as training, and you call that grokked. The claim is that it happens long after: compare `v` with `DELAY * t`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de validatie haalt 99% op dezelfde stap als de training, en jij noemt dat grokked. De claim is dat het lang daarna gebeurt: vergelijk `v` met `DELAY * t`."),
            Mistake("later", wrong=_WRONG["later"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: validation reaches 99% at step 500, only 5 times as late as training at step 100, and you call that grokked. The plan says at least 10 times as late: compare `v` with `DELAY * t`, with `DELAY = 10`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de validatie haalt 99% op stap 500, maar 5 keer zo laat als de training op stap 100, en jij noemt dat grokked. Het plan zegt minstens 10 keer zo laat: vergelijk `v` met `DELAY * t`, met `DELAY = 10`."),
            Mistake("order", wrong=_WRONG["order"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: training never reaches 99% here, so the run did not fit its training data, whatever validation does. Look at training first.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de training haalt hier nooit 99%, dus de run heeft zijn training data niet geleerd, wat de validatie ook doet. Kijk eerst naar de training."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace each `...` with your code: in `first_step`, the first step that reaches the threshold; in `verdict`, the verdict for one run.",
                      "nl": "Vervang elke `...` door je eigen code: in `first_step` de eerste stap die de drempel haalt, in `verdict` het oordeel over één run."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the steps and the two accuracy logs, and the `def` line of your `verdict` has another number of names between its parentheses. Keep `(steps, train_acc, val_acc)`.",
                          "nl": "De check roept `{call}` aan, met de stappen en de twee accuracy-logs, en de `def`-regel van je `verdict` heeft een ander aantal namen tussen de haakjes. Houd `(steps, train_acc, val_acc)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you find the first step at which each accuracy reaches `THRESHOLD`, and then check training, validation and the delay, in that order?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Zoek je de eerste stap waarop elke accuracy `THRESHOLD` haalt, en kijk je dan naar de training, de validatie en de vertraging, in die volgorde?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your verdict work for every kind of run: one without a delay, one where validation never gets there, one that did not fit, and a delay of exactly 10 times?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je oordeel voor elk soort run: een zonder vertraging, een waarin de validatie er nooit komt, een die zijn training data nooit haalt, en een vertraging van precies 10 keer?"},
            "correct": {"en": "Your verdict is right for all {n} runs: `\"grokked\"`, `\"no delay\"` at the same step and at 5 times as late, `\"not reached\"`, `\"did not fit\"`, and the edges at exactly 99% and exactly 10 times. It is the first part of your reproduction kit now, as `{saved}`: before the first run, it says what counts.",
                        "nl": "Je oordeel klopt voor alle {n} runs: `\"grokked\"`, `\"no delay\"` op dezelfde stap en op 5 keer zo laat, `\"not reached\"`, `\"did not fit\"`, en de randen bij precies 99% en precies 10 keer. Het is nu het eerste deel van je reproductiekit, als `{saved}`: vóór de eerste run zegt het wat telt."},
            "kept": {"en": "Your verdict is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je oordeel is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# The lesson is done when these are correct. "which-claim" is the choice question, "why-before" the open question.
REQUIRED = ["which-claim", "verdict", "why-before"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("verdict", _START, "empty", None),
    ("verdict", _p(), "correct", None),
    ("verdict", _HEAD + "def first_step(steps, accuracy, threshold):\n    i = 0\n    while i < len(steps):\n"
     "        if accuracy[i] >= threshold:\n            return steps[i]\n        i = i + 1\n    return None\n\n\n" + _VERDICT, "correct", None),
    ("verdict", _p(verdict=("def verdict(steps, train_acc, val_acc):\n    t = first_step(steps, train_acc, THRESHOLD)\n"
                            "    v = first_step(steps, val_acc, THRESHOLD)\n    if t is None:\n        return \"did not fit\"\n"
                            "    elif v is None:\n        return \"not reached\"\n    elif v >= DELAY * t:\n        return \"grokked\"\n"
                            "    else:\n        return \"no delay\"\n")), "correct", None),
] + [("verdict", code, "wrong", key) for key, code in _WRONG.items()]


# --- the demo: your verdict on three made-up runs --------------------------------------------------------------------------

_DEMO_STEPS = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
_DEMO_TRAIN = [0.1, 0.4, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
DEMO_RUNS = [
    (_DEMO_STEPS, _DEMO_TRAIN, [0.0, 0.01, 0.01, 0.01, 0.02, 0.05, 0.3, 0.995, 1.0]),
    (_DEMO_STEPS, _DEMO_TRAIN, [0.0, 0.1, 0.5, 0.95, 0.995, 1.0, 1.0, 1.0, 1.0]),
    (_DEMO_STEPS, _DEMO_TRAIN, [0.0, 0.01, 0.01, 0.02, 0.02, 0.05, 0.1, 0.15, 0.2]),
]


def demo():
    """The learner's own verdict, from work/projects/repro/plan.py, on the three made-up runs: a list of verdicts,
    None when the file is not there yet, or "error" when it stops with an error."""
    import importlib
    from everything_ai import projects
    projects.on_path()
    if not (projects.folder() / "repro" / "plan.py").exists():
        return None
    projects.forget()
    try:
        plan = importlib.import_module("repro.plan")
        return [plan.verdict(*[list(part) for part in run]) for run in DEMO_RUNS]
    except Exception:
        return "error"
