# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson repro-own-ablation: the ablation, a third setting "ablation" that the learner's own/ablation.py
adds to the SETTINGS of their experiment, the setting of the claim with one part of its method taken out, saved as
own/ablation.py; and kept, how much of the claim's gain the ablation keeps, seed by seed, saved as own/kept.py. The
play block is the course's ablations of its dropout experiment: without the dropout at the input, and without the
dropout in the hidden layers, three seeds each at the budget of the course's plan."""
import importlib.util
import math
import sys
import time

import torch  # noqa: F401  loaded here: a first import inside the time limit of a check would not finish in it

from everything_ai import paths, projects
from everything_ai.check import MISSING, TIME_LIMIT, CodeBox, FunctionBox, Mistake, Result, Timeout, _fill, _show_rounded, is_empty_text, run_code, with_time_limit


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F = _checks("repro-own-figure", "everything_ai_checks_repro_own_figure_for_ablation")
B, P = F.B, F.P
E, SEEDS, RUNS, number, short, whole, median = B.E, B.SEEDS, B.RUNS, B.number, B.short, B.whole, F.median
BUDGET = F.COURSE_STEPS   # the ablation runs at the budget of the plan, over its seeds
STEPS = 20                # the steps of the check's runs: the code, not the claim
RUN_LIMIT = 60.0          # seconds a run of the check gets
HALF = 0.5                # the rule: the result depends on the part when the ablation keeps less than half of the gain

BUILDS_ON = (("own", "experiment", "repro-own-build", "experiment"),)
projects.builds_on(*BUILDS_ON)

# The course's ablations of its dropout experiment, at BUDGET steps, seeds 0, 1 and 2, measured on one thread with
# data/tools/measure_dropout.py --ablation: the dropout of the input and of the hidden layers, and the test error.
ABLATIONS = {
    "no-input": {"settings": (0.0, 0.5), "runs": [13.8, 13.2, 14.1]},    # dropout in the hidden layers only
    "no-hidden": {"settings": (0.2, 0.0), "runs": [13.5, 13.3, 12.7]},   # dropout at the input only
}
COURSE_ABLATION = "no-input"   # the course's own ablation, the solution of exercise 1
ABLATION_MINUTES = 7          # its three runs of 12,800 steps on one thread, about 130 seconds each


def course_runs(name: str) -> list[float]:
    """The test error of seeds 0, 1 and 2 at BUDGET steps: "claim", "other", or one of ABLATIONS."""
    return list(RUNS[BUDGET][name]) if name in RUNS[BUDGET] else list(ABLATIONS[name]["runs"])


def signed(x: float, lang: str) -> str:
    """A number as the text writes it, with a real minus sign."""
    return short(x, lang).replace("-", "\u2212")


def gains(runs: list[float], other: list[float], direction: str = "lower") -> list[float]:
    """Seed by seed, how far `runs` is ahead of the other setting, the way the claim says."""
    return [round((o - r) if direction == "lower" else (r - o), 4) for r, o in zip(runs, other)]


def course_kept(name: str) -> float:
    """How much of the claim's gain an ablation of the course keeps: its median gain over "other", against the claim's."""
    other = course_runs("other")
    return median(gains(course_runs(name), other)) / median(gains(course_runs("claim"), other))


def depends(share: float | None) -> bool | None:
    """The rule of the lesson: the result depends on the part when the ablation keeps less than half of the gain."""
    return None if share is None else share < HALF


def ablation_picture(k: int, names: list[str], labels: dict) -> str:
    """The four settings of the play block at BUDGET steps, a row each: the claim, the two ablations, and the other
    setting; each seed a dot and the median a ring, on an axis of test error; a dashed line at the median of the
    other setting and a solid one at the claim's. Row `k`, the slider's, is marked. Purple: the test error counts
    mistakes."""
    from everything_ai import chart
    rows = ["claim", "no-input", "no-hidden", "other"]
    p = chart.Panel((10.2, 14.6), (-0.6, len(rows) - 0.4), title=labels["title"], xlabel=labels["x"], ylabel="",
                    yticks=[], xticks=[12, 13, 14])   # room on the left for the names of the rows
    top = len(rows) - 1
    p.line([median(course_runs("other"))] * 2, [-0.5, top + 0.5], "ink-3", 1.0, dashed=True)
    p.line([median(course_runs("claim"))] * 2, [-0.5, top + 0.5], "ink-3", 1.0)
    for i, name in enumerate(rows):
        y = top - i
        if i == k:
            p.rect(10.2, y - 0.42, 14.6, y + 0.42, "ink-3")
        runs = course_runs(name)
        p.points(runs, [y] * len(runs), "loss", r=3.5)
        p.points([median(runs)], [y], "loss", r=7, hollow=True)
        p.text(10.3, y - 0.08, names[i], "ink-2", "start")
    return chart.figure([p], alt=labels["alt"], width=360, height=260, drag=0)


# --- exercise 1: the ablation -------------------------------------------------------------------------------------------

_A_HEAD = ("from own import experiment   # your experiment of Your reproduction 3, from your kit\n\n"
           "# the ablation: the setting of the claim with one part of its method taken out, and\n"
           "# all the rest as it is. It adds the setting \"ablation\" to the SETTINGS of your\n"
           "# experiment while it runs; the file of your experiment stays as it is\n")
_A_SET = "experiment.SETTINGS[\"ablation\"] = (0.0, 0.5)   # the claim's (0.2, 0.5), without the input's dropout\n"
_A_PART = "PART = \"the dropout at the input\"   # the part you take out, in a few words\n"
_A_RUN = "run = experiment.run\n"
_A_START = _A_HEAD + "experiment.SETTINGS[\"ablation\"] = ...\nPART = \"...\"   # the part you take out, in a few words\n" + _A_RUN
A_CODE = _A_HEAD + _A_SET + _A_PART + _A_RUN
_A_WRONG = {
    "two-parts": A_CODE.replace("(0.0, 0.5)   #", "(0.0, 0.2)   #"),
    "same-as-claim": A_CODE.replace("(0.0, 0.5)   #", "(0.2, 0.5)   #"),
    "same-as-other": A_CODE.replace("(0.0, 0.5)   #", "(0.0, 0.0)   #"),
    "no-part": A_CODE.replace("PART = \"the dropout at the input\"", "PART = \"\""),
    "claim-changed": A_CODE.replace("experiment.SETTINGS[\"ablation\"] = (0.0, 0.5)", "experiment.SETTINGS[\"claim\"] = (0.0, 0.5)"),
}


def _parts(a, b) -> int | None:
    """In how many parts two settings differ, when they have parts: tuples or lists of one length, or dicts of the same
    keys. None when they cannot be compared part by part."""
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)) and len(a) == len(b):
        return sum(x != y for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b):
        return sum(a[k] != b[k] for k in a)
    return None


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not (isinstance(x, float) and math.isnan(x))


class AblationBox(CodeBox):
    """The learner's own/ablation.py in a code box: it imports their experiment, adds the setting "ablation" to its
    SETTINGS, names the part it takes out, and gives `run`. The check runs the box, compares the new setting with
    "claim" and "other", and runs `run("ablation", 0, STEPS)` twice, each within RUN_LIMIT seconds."""

    def check(self, text, inputs=None) -> Result:
        if text is MISSING or text is None:
            return Result("error", self.message("missing"))
        if is_empty_text(text, self.start):
            cleared = text.strip() != self.start.strip() and "cleared" in self.texts
            return Result("empty", self.message("cleared" if cleared else "empty"))
        for m in self.first:
            if m.matches(None, text):
                return Result(m.state, m.text, mistake=m.key)
        before = run_code("from own import experiment\nexperiment.SETTINGS")
        ran = run_code(text)
        if ran.timeout:
            return Result("slow", self.message("slow-top", s=f"{TIME_LIMIT:g}"))
        if ran.error is not None:
            return Result("error", self._plain_error(ran, text))
        exp = sys.modules.get("own.experiment") or ran.names.get("experiment")   # also `from own.experiment import SETTINGS`
        settings = getattr(exp, "SETTINGS", None)
        if not isinstance(settings, dict) or not isinstance(before.value, dict) or not {"claim", "other"} <= set(before.value):
            return self._mistake("no-settings")
        if any(settings.get(k) != before.value[k] for k in ("claim", "other")):
            return self._mistake("claim-changed")
        if "ablation" not in settings:
            return self._mistake("no-ablation")
        a, claim, other = settings["ablation"], settings["claim"], settings["other"]
        if a == claim or _parts(a, claim) == 0:   # also [0.2, 0.5] for the claim's (0.2, 0.5)
            return self._mistake("same-as-claim")
        if a == other or _parts(a, other) == 0:
            return self._mistake("same-as-other")
        parts = _parts(a, claim)
        if parts is not None and parts > 1:
            return self._mistake("two-parts", parts=str(parts))
        part = ran.names.get("PART")
        if not isinstance(part, str) or not part.strip() or part.strip() == "...":
            return self._mistake("no-part")
        fn = ran.names.get("run")
        if not callable(fn):
            return self._mistake("no-run")
        try:
            first = with_time_limit(lambda: fn("ablation", 0, STEPS), RUN_LIMIT)
            again = with_time_limit(lambda: fn("ablation", 0, STEPS), RUN_LIMIT)
        except Timeout:
            return self._mistake("slow", steps=str(STEPS))
        except Exception:
            return self._mistake("stops", steps=str(STEPS))
        if not _number(first):
            return self._mistake("not-a-number", got=B._shown(first), steps=str(STEPS))
        if first != again:
            return self._mistake("not-the-same-twice", a=B._shown(first), b=B._shown(again), steps=str(STEPS))
        return self._correct(text, a=B._shown(a), claim=B._shown(claim), part=part.strip(), steps=str(STEPS),
                             got={"en": short(first, "en"), "nl": short(first, "nl")})

    def _mistake(self, key: str, **values) -> Result:
        m = next(m for m in self.mistakes if m.key == key)
        return Result(m.state, _fill(m.text, **values), mistake=key)


_R3 = {"en": "[Your reproduction 3](/course/open/repro-own-build?lang=en)", "nl": "[Je reproductie 3](/course/open/repro-own-build?lang=nl)"}
_AB_EN, _AB_NL = "Your ablation", "Je ablation"

EXERCISES = {
    "ablation": AblationBox(
        name="ablation",
        start=_A_START,
        solution=A_CODE,
        save_as=("own", "ablation"),
        first=[B._DOTS],
        mistakes=[
            Mistake("no-settings", state="error",
                    en="Your experiment has no `SETTINGS` dictionary with `\"claim\"` and `\"other\"`, the way the kit keeps its settings, so the ablation has nowhere to add its own. Keep them in one, as the start of exercise 2 of " + _R3["en"] + " has it.",
                    nl="Je experiment heeft geen dictionary `SETTINGS` met `\"claim\"` en `\"other\"`, zoals de kit zijn settings bewaart, dus kan de ablation zijn eigen setting nergens toevoegen. Bewaar ze in één, zoals het begin van oefening 2 van " + _R3["nl"] + " het heeft."),
            Mistake("no-ablation",
                    en=_AB_EN + " adds no setting `\"ablation\"` to the `SETTINGS` of your experiment. Add it as a key: `experiment.SETTINGS[\"ablation\"] = ...`, with the value of `\"claim\"` without one part.",
                    nl=_AB_NL + " voegt geen setting `\"ablation\"` toe aan de `SETTINGS` van je experiment. Voeg hem toe als sleutel: `experiment.SETTINGS[\"ablation\"] = ...`, met de waarde van `\"claim\"` zonder één deel."),
            Mistake("claim-changed", wrong=_A_WRONG["claim-changed"],
                    en=_AB_EN + " changes the setting `\"claim\"` or `\"other\"` of your experiment, and those are what your main result ran. Leave them as they are, and add a third key, `\"ablation\"`.",
                    nl=_AB_NL + " verandert de setting `\"claim\"` of `\"other\"` van je experiment, en die heeft je hoofdresultaat gedraaid. Laat ze zoals ze zijn, en voeg een derde sleutel toe, `\"ablation\"`."),
            Mistake("same-as-claim", wrong=_A_WRONG["same-as-claim"],
                    en=_AB_EN + " is the setting of the claim itself: nothing is taken out, so its runs would be the claim's again. Take out one part of the method, such as one of the numbers in the value of `\"claim\"`.",
                    nl=_AB_NL + " is de setting van de claim zelf: er is niets weggehaald, dus zouden zijn runs weer die van de claim zijn. Haal één deel van de methode weg, zoals een van de getallen in de waarde van `\"claim\"`."),
            Mistake("same-as-other", wrong=_A_WRONG["same-as-other"],
                    en=_AB_EN + " is the other setting: every part of the method is out, and your main result has those runs already. An ablation takes out one part, and keeps the rest of the claim's setting.",
                    nl=_AB_NL + " is de andere setting: elk deel van de methode is eruit, en je hoofdresultaat heeft die runs al. Een ablation haalt één deel weg, en houdt de rest van de setting van de claim."),
            Mistake("two-parts", wrong=_A_WRONG["two-parts"],
                    en=_AB_EN + " differs from the setting of the claim in {parts} parts at once. A change in its result could then come from either, and the ablation could not say which part your result depends on. Take out one, and keep the others as the claim has them.",
                    nl=_AB_NL + " verschilt van de setting van de claim in {parts} delen tegelijk. Een verandering in zijn resultaat kan dan van elk komen, en de ablation kan niet zeggen van welk deel je resultaat afhangt. Haal er één weg, en houd de andere zoals de claim ze heeft."),
            Mistake("no-part", wrong=_A_WRONG["no-part"],
                    en="Say in `PART`, in a few words, which part of the method your ablation takes out, such as `\"the dropout at the input\"`. Your report names it.",
                    nl="Zeg in `PART`, in een paar woorden, welk deel van de methode je ablation weghaalt, zoals `\"de dropout bij de input\"`. Je verslag noemt het."),
            Mistake("no-run",
                    en="Your box gives no `run`. Give the run of your experiment: `run = experiment.run`.",
                    nl="Je vak geeft geen `run`. Geef de run van je experiment: `run = experiment.run`."),
            Mistake("stops", state="error",
                    en="`run(\"ablation\", 0, {steps})` stops with an error. Does `make_model` of your experiment read the value of the setting the way it reads that of `\"claim\"`? The ablation's value must have the same form.",
                    nl="`run(\"ablation\", 0, {steps})` stopt met een fout. Leest `make_model` van je experiment de waarde van de setting zoals hij die van `\"claim\"` leest? De waarde van de ablation moet dezelfde vorm hebben."),
            Mistake("slow", state="slow",
                    en="`run(\"ablation\", 0, {steps})` is not done after a minute, and the check gives each run no more. Does your experiment take as long with `\"claim\"`?",
                    nl="`run(\"ablation\", 0, {steps})` is na een minuut nog niet klaar, en meer geeft de check een run niet. Duurt je experiment met `\"claim\"` net zo lang?"),
            Mistake("not-a-number",
                    en="`run(\"ablation\", 0, {steps})` gives back `{got}`, not one number: the contract of your kit. Does your experiment give a number with `\"claim\"`?",
                    nl="`run(\"ablation\", 0, {steps})` geeft `{got}` terug, niet één getal: het contract van je kit. Geeft je experiment een getal met `\"claim\"`?"),
            Mistake("not-the-same-twice",
                    en="`run(\"ablation\", 0, {steps})` gives `{a}` and then `{b}`: not the same twice. Is something random in your experiment not set by the seed for this setting?",
                    nl="`run(\"ablation\", 0, {steps})` geeft `{a}` en daarna `{b}`: niet twee keer hetzelfde. Is er iets willekeurigs in je experiment dat voor deze setting niet door de seed wordt vastgelegd?"),
        ],
        texts={
            **F._KEEP,
            "empty": {"en": "Replace the `...`: the value of the setting `\"ablation\"`, the claim's with one part taken out, and in `PART` which part that is.",
                      "nl": "Vervang de `...`: de waarde van de setting `\"ablation\"`, die van de claim met één deel weggehaald, en in `PART` welk deel dat is."},
            "slow-top": {"en": "Your box is not done after {s} seconds, all the time the check gives the lines outside `run`. Does your experiment load data or a model when it is imported? Load them inside its functions, in " + _R3["en"] + ".",
                         "nl": "Je vak is na {s} seconden nog niet klaar, alle tijd die de check de regels buiten `run` geeft. Laadt je experiment data of een model als het geïmporteerd wordt? Laad ze in zijn functies, in " + _R3["nl"] + "."},
            "kept": {"en": "Your ablation is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je ablation klopt. Je hebt `{saved}` zelf veranderd, dus laat de check dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
            "correct": {"en": "Your ablation is {a}, the claim's {claim} without {part}; with it, `run(\"ablation\", 0, {steps})` gives {got}, the same twice. It is in your kit now, as `{saved}`.",
                        "nl": "Je ablation is {a}, die van de claim, {claim}, zonder {part}; daarmee geeft `run(\"ablation\", 0, {steps})` {got}, twee keer hetzelfde. Hij zit nu in je kit, als `{saved}`."},
        },
    ),
}


# --- exercise 2: what the ablation keeps --------------------------------------------------------------------------------

_K_HEAD = ("import numpy as np\n\n\n"
           "def kept(direction, claim, ablation, other):\n"
           "    # how much of the claim's gain the ablation keeps: the median of its gains over the\n"
           "    # other setting, seed by seed, against the median of the claim's gains; None when the\n"
           "    # claim has no gain to keep. Each list holds the metric of seed 0, 1, 2, ...\n"
           "    # direction is \"lower\" or \"higher\", the way the claim says the metric goes\n")
_K_GAINS = ("    claim_gains, ablation_gains = [], []\n"
            "    for c, a, o in zip(claim, ablation, other):\n"
            "        if direction == \"lower\":\n"
            "            claim_gains.append(o - c)\n"
            "            ablation_gains.append(o - a)\n"
            "        else:\n"
            "            claim_gains.append(c - o)\n"
            "            ablation_gains.append(a - o)\n")
_K_TAIL = ("    if np.median(claim_gains) <= 0:\n"
           "        return None\n"
           "    return float(np.median(ablation_gains) / np.median(claim_gains))\n")
_K_START = _K_HEAD + "    # 1. the gains of the claim and of the ablation over the other setting, seed by seed\n    ...\n" \
           "    # 2. None when the claim has no gain; else the share the ablation keeps\n    ...\n"
K_CODE = _K_HEAD + _K_GAINS + _K_TAIL
_K_WRONG = {
    "no-direction": _K_HEAD + "    claim_gains, ablation_gains = [], []\n    for c, a, o in zip(claim, ablation, other):\n"
                    "        claim_gains.append(o - c)\n        ablation_gains.append(o - a)\n" + _K_TAIL,
    "unpaired": _K_HEAD + _K_GAINS + "    if np.median(claim_gains) <= 0:\n        return None\n"
                "    if direction == \"lower\":\n"
                "        return float((np.median(other) - np.median(ablation)) / (np.median(other) - np.median(claim)))\n"
                "    return float((np.median(ablation) - np.median(other)) / (np.median(claim) - np.median(other)))\n",
    "per-seed": _K_HEAD + _K_GAINS + "    if np.median(claim_gains) <= 0:\n        return None\n"
                "    return float(np.median([a / c for a, c in zip(ablation_gains, claim_gains)]))\n",
    "no-none": _K_HEAD + _K_GAINS + "    return float(np.median(ablation_gains) / np.median(claim_gains))\n",
    "against-claim": _K_HEAD + _K_GAINS + _K_TAIL.replace("np.median(ablation_gains) / np.median(claim_gains)",
                                                          "1 - np.median(ablation_gains) / np.median(claim_gains)"),
}
_COURSE_CASE = ("lower", [12.5, 12.9, 12.4], [13.8, 13.2, 14.1], [12.9, 13.6, 13.6])
EXERCISES["kept"] = FunctionBox(
    name="kept",
    function="kept",
    start=_K_START,
    solution=K_CODE,
    visible=[_COURSE_CASE],
    hidden=[("lower", [12.5, 12.9, 12.4], [13.5, 13.3, 12.7], [12.9, 13.6, 13.6]),
            ("higher", [80.0, 82.0, 81.0], [79.0, 80.0, 80.5], [76.0, 77.0, 78.0]),
            ("lower", [10.0, 10.0], [10.0, 11.0], [10.0, 10.0]),
            ("higher", [70.0, 71.0, 69.0], [72.0, 70.0, 71.0], [71.0, 72.0, 70.0]),   # a claim that does worse: None too
            ("lower", [5.0, 6.2, 7.4, 8.0], [5.6, 6.2, 7.9, 8.4], [6.0, 7.0, 8.0, 9.0]),
            ("higher", [0.3, 0.9, 0.5], [0.2, 0.8, 0.4], [0.1, 0.1, 0.4])],
    save_as=("own", "kept"),
    show=lambda v: _show_rounded(v, 4),   # -0.7143, not -0.714285714285715
    first=[B._DOTS],
    mistakes=[
        Mistake("no-none", wrong=_K_WRONG["no-none"], state="almost",
                en="`{call}` gives `{got}`, and it should give `{want}`: your `kept` divides by a median gain of 0: the claim has no gain there, so there is nothing to keep, and the share has no meaning. Give back `None` first when the median of the claim's gains is 0 or less.",
                nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je `kept` deelt door een winst van 0 in de mediaan: de claim heeft daar geen winst, dus is er niets te houden, en betekent het aandeel niets. Geef eerst `None` terug als de mediaan van de winsten van de claim 0 of minder is."),
        Mistake("unpaired", wrong=_K_WRONG["unpaired"],
                en="`{call}` gives `{got}`, and it should give `{want}`: you compare the medians of the settings, not the gains seed by seed. Each seed of the claim and of the ablation has the same starting weights and batches as that seed of the other setting: the gain of a seed is its own difference. Take the median of those.",
                nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je vergelijkt de medianen van de settings, niet de winsten seed voor seed. Elke seed van de claim en van de ablation heeft dezelfde startwaarden en batches als die seed van de andere setting: de winst van een seed is zijn eigen verschil. Neem daarvan de mediaan."),
        Mistake("per-seed", wrong=_K_WRONG["per-seed"],
                en="`{call}` gives `{got}`, and it should give `{want}`: you take the median of the share of each seed. A seed whose claim gains little, say 0.1, then gives a share of 5 or of −9 that says little. Divide the median of the ablation's gains by the median of the claim's.",
                nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je neemt de mediaan van het aandeel van elke seed. Een seed waarvan de claim weinig wint, zeg 0,1, geeft dan een aandeel van 5 of van −9 dat weinig zegt. Deel de mediaan van de winsten van de ablation door de mediaan van die van de claim."),
        Mistake("against-claim", wrong=_K_WRONG["against-claim"],
                en="`{call}` gives `{got}`, and it should give `{want}`: that is the share the ablation loses, not the share it keeps. Give back what it keeps: its median gain over the claim's.",
                nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is het aandeel dat de ablation verliest, niet het aandeel dat hij houdt. Geef terug wat hij houdt: zijn winst in de mediaan, gedeeld door die van de claim."),
    ],
    texts={
        **F._KEEP,
        "empty": {"en": "Replace both `...`: first the gains of the claim and of the ablation over the other setting, seed by seed, the way `direction` says; then `None` when the claim has no gain, or the share the ablation keeps.",
                  "nl": "Vervang beide `...`: eerst de winsten van de claim en van de ablation op de andere setting, seed voor seed, zoals `direction` zegt; dan `None` als de claim geen winst heeft, of het aandeel dat de ablation houdt."},
        "unreached": {"en": "With `{call}`, your function gives back `None`, and here the claim has a gain to keep. Does every way through it end in a `return` with the share, and does it take the gains the way `direction` says: for a metric that should go down, the other minus the setting, and for one that should go up, the setting minus the other?",
                      "nl": "Met `{call}` geeft je functie `None` terug, en hier heeft de claim wel een winst om te houden. Eindigt elke weg erdoor met een `return` met het aandeel, en neemt hij de winsten zoals `direction` zegt: bij een metric die omlaag hoort te gaan, de andere min de setting, en bij een die omhoog hoort te gaan, de setting min de andere?"},
        "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does it take the gain of each seed over the other setting, the way `direction` says, and divide the median of the ablation's gains by the median of the claim's?",
                  "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Neemt hij de winst van elke seed op de andere setting, zoals `direction` zegt, en deelt hij de mediaan van de winsten van de ablation door de mediaan van die van de claim?"},
        "almost": {"en": "Your `kept` is right for the course's ablation. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for a metric that should go up, for a claim without a gain, and for another number of seeds?",
                   "nl": "Je `kept` klopt voor de ablation van de course. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor een metric die omhoog hoort te gaan, voor een claim zonder winst, en voor een ander aantal seeds?"},
        "kept": {"en": "Your `kept` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                 "nl": "Je `kept` klopt. Je hebt `{saved}` zelf veranderd, dus laat de check dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        "correct": {"en": "Your `kept` gives how much of the claim's gain the ablation keeps, seed by seed, or `None` when there is no gain to keep. It is in your kit now, as `{saved}`.",
                    "nl": "Je `kept` geeft hoeveel van de winst van de claim de ablation houdt, seed voor seed, of `None` als er geen winst te houden is. Hij zit nu in je kit, als `{saved}`."},
    },
)

REQUIRED = ["ablation", "kept", "why-ablation"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("ablation", _A_START, "empty", None),
    ("ablation", A_CODE, "correct", None),
    ("ablation", A_CODE.replace("(0.0, 0.5)   #", "(0.2, 0.0)   #"), "correct", None),   # the other ablation
    ("ablation", "from own.experiment import SETTINGS, run\n\nSETTINGS[\"ablation\"] = (0.0, 0.5)\nPART = \"the dropout at the input\"\n", "correct", None),
    ("ablation", A_CODE.replace("(0.0, 0.5)   #", "[0.2, 0.5]   #"), "wrong", "same-as-claim"),   # a list of the claim's values
    ("ablation", A_CODE.replace("experiment.SETTINGS[\"ablation\"] = (0.0, 0.5)", "experiment.SETTINGS[\"ablation\"] = \"none\""), "error", "stops"),
    ("ablation", A_CODE.replace("experiment.SETTINGS[\"ablation\"] = (0.0, 0.5)   # the claim's (0.2, 0.5), without the input's dropout\n", ""), "wrong", "no-ablation"),
    ("ablation", A_CODE.replace("run = experiment.run", "run = None"), "wrong", "no-run"),
] + [("ablation", code, "wrong", key) for key, code in _A_WRONG.items()] + [
    ("kept", _K_START, "empty", None),
    ("kept", K_CODE, "correct", None),
    ("kept", _K_WRONG["no-direction"], "wrong", "unreached"),
] + [("kept", code, "almost" if key == "no-none" else "wrong", key) for key, code in _K_WRONG.items() if key != "no-direction"]


# --- the run button and the demo ----------------------------------------------------------------------------------------

def ablation_path():
    return projects.folder() / "own" / "ablation.jsonl"


def my_ablation_log() -> list[dict] | None:
    """The lines of the learner's ablation log, work/projects/own/ablation.jsonl; None when there is none."""
    import json
    path = ablation_path()
    if not path.is_file():
        return None
    try:
        lines = [json.loads(t) for t in path.read_text(encoding="utf-8").splitlines() if t.strip()]
    except ValueError:
        return None
    ok = all(isinstance(r, dict) and r.get("setting") == "ablation" and isinstance(r.get("seed"), int) and isinstance(r.get("steps"), int)
             and _number(r.get("result")) for r in lines)
    return lines if lines and ok else None


def demo(progress=None) -> list[dict] | str | None:
    """Runs the ablation with the learner's kit: their run_plan with the run of their own/ablation.py, over the seeds of
    their plan at their budget, into work/projects/own/ablation.jsonl. None when a part of the kit is missing;
    "budget" when there is no budget; "log" when a line of the ablation log is not a run of the ablation as run_plan
    writes it; "error" when it stops with an error; otherwise the lines of the ablation log."""
    import importlib
    folder = projects.folder() / "own"
    plan, budget = P.saved_plan(), F.saved_budget()
    if not all((folder / f"{m}.py").is_file() for m in ("runner", "experiment", "ablation")) or plan is None:
        return None
    if budget is None:
        return "budget"
    if ablation_path().is_file() and ablation_path().read_text(encoding="utf-8").strip() and my_ablation_log() is None:
        return "log"
    projects.on_path()
    projects.forget()
    try:
        run_plan = importlib.import_module("own.runner").run_plan
        run = importlib.import_module("own.ablation").run
        run_plan(run, list(range(int(plan["seeds"]))), [budget["steps"]], str(ablation_path()), progress, ["ablation"])
    except Exception:
        return "error"
    finally:
        projects.forget()
    lines = my_ablation_log()
    return "log" if lines is None else lines


def my_reading(plan: dict, steps: int) -> dict | None:
    """The learner's runs at their budget, the claim's and the other's from their main log and the ablation's from its
    own, seed by seed for the seeds of the plan, with what their own/kept.py says the ablation keeps: {"claim",
    "ablation", "other", "kept"}. None when a part is missing or stops."""
    import importlib
    main, abl = F.my_log(), my_ablation_log()
    if not main or not abl:
        return None
    n = int(plan["seeds"])
    a, b = P._settings(plan)
    res = F.course_results_at(main, steps, plan)
    mine = [line["result"] for line in sorted(abl, key=lambda line: line["seed"]) if line["steps"] == steps]
    if len(res[a]) < n or len(res[b]) < n or len(mine) < n:
        return None
    out = {"claim": res[a][:n], "ablation": mine[:n], "other": res[b][:n]}
    projects.on_path()
    projects.forget()
    try:
        out["kept"] = importlib.import_module("own.kept").kept(plan["direction"], out["claim"], out["ablation"], out["other"])
        if out["kept"] is not None and not (_number(out["kept"]) and math.isfinite(out["kept"])):   # a kept.py changed by hand
            out["kept"] = "error"
    except Exception:
        out["kept"] = "error"
    finally:
        projects.forget()
    return out
